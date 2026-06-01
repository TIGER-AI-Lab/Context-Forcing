from typing import List, Optional
import torch

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class CausalLongInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            num_max_frames: int = 21,
            sink_size: int = 3,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.return_attn_map = getattr(args, "return_attn_map", False)
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True, return_attn_map=self.return_attn_map) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all causal hyperparameters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
        cache_size = getattr(args, "model_kwargs", {}).get("cache_size", 0)
        self.cache_size = cache_size
        self.kv_cache_size = cache_size * 1560 if cache_size > 0 else 32760 # 32760 # self.local_attn_size * self.frame_seq_length
        self.sink_size = sink_size


    def _predict_from_context(
        self,
        noise: torch.Tensor,
        initial_latent: Optional[torch.Tensor],
        conditional_dict: dict,
        profile: bool = False
    ): # -> (torch.Tensor, dict):
        """
        [NEW] Core autoregressive generation function.
        Handles KV cache, context priming, and the temporal denoising loop.
        This is a pure latent-in, latent-out function.
        """
        # --- Setup and Shape Calculation ---
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames
        
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )
        
        # --- Profiling Setup ---
        profiling_data = {}
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            block_events = []
            init_start.record()

        self.kv_cache1 = None
        # --- Step 1: Initialize KV Cache ---
        # (This logic is identical to your original implementation)
        # need to change with infer video length
        if self.kv_cache1 is None:
            print(self.kv_cache_size)
            self._initialize_kv_cache(batch_size, noise.dtype, noise.device)
            self._initialize_crossattn_cache(batch_size, noise.dtype, noise.device)
        else:
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["dynamic_sink_tokens"] = torch.tensor([self.sink_size * self.frame_seq_length], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["token_source_frame"] = torch.full((self.cache_size,), -1, dtype=torch.long, device=noise.device)

        # --- Step 2: Cache Context Feature (Context Priming) ---
        current_start_frame = 0
        if initial_latent is not None:
            # This logic processes the initial_latent in chunks to fill the KV cache.
            context_chunks = torch.split(initial_latent, self.num_frame_per_block, dim=1)
            for chunk in context_chunks:
                if chunk.shape[1] == 0: continue
                output[:, current_start_frame : current_start_frame + chunk.shape[1]] = chunk
                timestep_zero = torch.zeros([batch_size, chunk.shape[1]], device=noise.device, dtype=torch.int64)
                self.generator(
                    noisy_image_or_video=chunk,
                    conditional_dict=conditional_dict,
                    timestep=timestep_zero,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += chunk.shape[1]

        if profile:
            init_end = torch.cuda.Event(enable_timing=True)
            init_end.record()
            diffusion_start.record()

        # --- Step 3: Temporal Denoising Loop for New Frames ---
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
            
        for current_num_frames in all_num_frames:
            if profile:
                block_start = torch.cuda.Event(enable_timing=True)
                block_end = torch.cuda.Event(enable_timing=True)
                block_start.record()

            noise_idx_start = current_start_frame - num_input_frames
            noise_idx_end = noise_idx_start + current_num_frames
            noisy_input = noise[:, noise_idx_start:noise_idx_end]

            # Step 3.1: Spatial denoising loop for the current block
            denoised_pred = None
            for index, current_timestep in enumerate(self.denoising_step_list):
                print(f"current_timestep: {current_timestep}")
                timestep = torch.ones([batch_size, current_num_frames], device=noise.device, dtype=torch.int64) * current_timestep
                
                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred, self.kv_cache1 = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    _, denoised_pred, self.kv_cache1 = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )

            # Step 3.2: Record the model's final output for this block
            output[:, current_start_frame : current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: Rerun with the clean prediction to update KV cache for the *next* block
            context_timestep = torch.ones_like(timestep) * self.args.context_noise

            # Do KV-ReCache
            if current_start_frame % self.local_attn_size == 0 and current_start_frame > 0 and False:
                print(f"Do kv-recache, range:[{current_start_frame + current_num_frames - self.local_attn_size}, {current_start_frame + current_num_frames}]")
                # Step 2: recompute kv cache
                context_timestep = torch.ones(
                    [batch_size, self.local_attn_size],
                    device=noise.device,
                    dtype=torch.int64) * 0
                update_clip = output[:, current_start_frame + current_num_frames - self.local_attn_size:current_start_frame + current_num_frames]
                
                sink_cache = None
                if self.kv_cache1 is not None:
                    if getattr(self, 'cached_sink', None) is not None:
                        sink_cache = self.cached_sink
                    else:
                        sink_tokens_count = self.sink_size * self.frame_seq_length 
                        sink_cache = []
                        
                        for block_cache in self.kv_cache1:
                            sink_k = block_cache["k"][:, :sink_tokens_count].clone()
                            sink_v = block_cache["v"][:, :sink_tokens_count].clone()
                            
                            device = sink_k.device
                            sink_end_idx = torch.tensor([sink_tokens_count], dtype=torch.long, device=device)
                            
                            block_sink = {
                                "k": sink_k,
                                "v": sink_v,
                                "global_end_index": sink_end_idx.clone(), 
                                "local_end_index": sink_end_idx.clone()   
                            }
                            
                            sink_cache.append(block_sink)
                    
                        self.cached_sink = sink_cache

                with torch.no_grad():
                    _, _, self.kv_cache1 = self.generator(
                        noisy_image_or_video=update_clip,
                        conditional_dict=conditional_dict,
                        timestep=context_timestep,
                        kv_cache=sink_cache,  
                        crossattn_cache=self.crossattn_cache,
                        current_start=(current_start_frame + current_num_frames - self.local_attn_size) * self.frame_seq_length
                    )
            else:
                print(f"Fill the kv, range:[{current_start_frame}, {current_start_frame + current_num_frames}]")
                with torch.no_grad():
                    _, _, self.kv_cache1 = self.generator(
                        noisy_image_or_video=denoised_pred,
                        conditional_dict=conditional_dict,
                        timestep=context_timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )
                print(f"Token_source_global_idx:", self.kv_cache1[0]["token_source_frame"])

            if profile:
                block_end.record()
                block_events.append((block_start, block_end))

            # Step 3.4: Update frame pointer
            current_start_frame += current_num_frames

        if profile:
            diffusion_end = torch.cuda.Event(enable_timing=True)
            diffusion_end.record()
            profiling_data = {
                "init_events": (init_start, init_end),
                "diffusion_events": (diffusion_start, diffusion_end),
                "block_events": block_events
            }
        '''
        import numpy as np

        recorder = {} 
        for layer_idx, layer_cache in enumerate(self.kv_cache1):
            if isinstance(layer_cache, dict) and "sim_recorder" in layer_cache:
                layer_recorder = layer_cache["sim_recorder"]
                
                for step, sim_val_list in layer_recorder.items():
                    if step not in recorder:
                        recorder[step] = []
                    
                    val = sim_val_list[0] if isinstance(sim_val_list, list) and len(sim_val_list) > 0 else 0.0
                    
                    recorder[step].append(val)
        sorted_steps = sorted(recorder.keys())

        data_matrix = []
        valid_steps = []

        for step in sorted_steps:
            layer_sims = recorder[step]
            data_matrix.append(layer_sims)
            valid_steps.append(step)

        data_matrix = np.array(data_matrix) 

        print("Average Sim per Layer:", np.mean(data_matrix, axis=0))
        print("Average Sim per Frame:", np.mean(data_matrix, axis=1))
        '''

        return output, profiling_data

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False
    ) -> torch.Tensor:
        """
        [MODIFIED] High-level wrapper for causal inference.
        Handles text encoding and VAE decoding, calling the core generation function internally.
        """
        with torch.no_grad():
            # --- Step 1: Prepare Inputs ---
            conditional_dict = self.text_encoder(text_prompts=text_prompts)

            # --- Step 2: Call the Core Generation Function ---
            output_latents, profiling_data = self._predict_from_context(
                noise=noise,
                initial_latent=initial_latent,
                conditional_dict=conditional_dict,
                profile=profile
            )
            
            # --- Step 3: Decode Latents to Pixels ---
            if profile:
                vae_start = torch.cuda.Event(enable_timing=True)
                vae_end = torch.cuda.Event(enable_timing=True)
                vae_start.record()

            video = self.vae.decode_to_pixel(output_latents, use_cache=False)
            video = (video * 0.5 + 0.5).clamp(0, 1)

            if profile:
                # --- Step 4: Process and Print Profiling Results ---
                torch.cuda.synchronize()
                vae_end.record()
                vae_end.synchronize()
                
                init_time = profiling_data["init_events"][0].elapsed_time(profiling_data["init_events"][1])
                diffusion_time = profiling_data["diffusion_events"][0].elapsed_time(profiling_data["diffusion_events"][1])
                vae_time = vae_start.elapsed_time(vae_end)
                total_time = init_time + diffusion_time + vae_time
                block_times = [start.elapsed_time(end) for start, end in profiling_data["block_events"]]

                print("\n--- Profiling Results ---")
                print(f"  - Initialization/Caching Time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
                print(f"  - Diffusion Generation Time:   {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
                for i, block_time in enumerate(block_times):
                    print(f"    - Block {i+1} Generation: {block_time:.2f} ms")
                print(f"  - VAE Decoding Time:           {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
                print(f"  - Total Inference Time:        {total_time:.2f} ms")
                print("-------------------------\n")


        if return_latents:
            return video, output_latents
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache