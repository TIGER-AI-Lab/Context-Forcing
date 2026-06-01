from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
from typing import List, Optional
import torch
import torch.distributed as dist


class SelfForcingTrainingPipeline:
    def __init__(self,
                 denoising_step_list: List[int],
                 scheduler: SchedulerInterface,
                 generator: WanDiffusionWrapper,
                 num_frame_per_block=3,
                 independent_first_frame: bool = False,
                 same_step_across_blocks: bool = False,
                 last_step_only: bool = False,
                 num_max_frames: int = 21,
                 context_noise: int = 0,
                 sink_size: int = 3,
                 local_attn_size: int = 21,
                 cache_size: int = 0,
                 all_fully_denoised: bool = False,
                 context_window: int = 21,
                 real_score_length: int = 21,
                 **kwargs):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]  # remove the zero timestep for inference

        # Wan specific hyperparameters
        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.i2v = False

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.independent_first_frame = independent_first_frame
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only
        self.local_attn_size = local_attn_size
        self.kv_cache_size = cache_size * 1560 if cache_size > 0 else 32760 # 32760 # self.local_attn_size * self.frame_seq_length # self.local_attn_size * self.frame_seq_length
        self.sink_size = sink_size
        self.all_fully_denoised = all_fully_denoised
        self.context_window = context_window

        self.real_score_length = real_score_length

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Generate random indices
            indices = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(num_blocks,),
                device=device
            )
            if self.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        dist.broadcast(indices, src=0)  # Broadcast the random indices to all ranks
        return indices.tolist()

    def predict_from_context(
        self,
        context_latent: torch.Tensor,
        num_frames_to_generate,
        teacher_forcing=False,
        return_sim_step: bool = False,
        **conditional_dict
    ) -> torch.Tensor:
        """
        Autoregressively predicts and generates future video frames starting from a given context.
        This is a pure inference function and does not compute gradients.

        Args:
            context_latent (torch.Tensor): The starting context video clip of shape [B, F_ctx, C, H, W].
            num_frames_to_generate (int): The number of new frames to generate.
            **conditional_dict: A dictionary containing conditional information like text embeddings.

        Returns:
            torch.Tensor: The complete video containing the context and the newly generated frames.
        """
    
        # --- 1. Initialization and Setup ---
        # (This part remains the same as before)
        batch_size, context_frames, num_channels, height, width = context_latent.shape
        assert num_frames_to_generate % self.num_frame_per_block == 0
        num_blocks_to_generate = num_frames_to_generate // self.num_frame_per_block

        noise = torch.randn(
            [batch_size, num_frames_to_generate, num_channels, height, width],
            device=context_latent.device,
            dtype=context_latent.dtype
        )

        total_output_frames = context_frames + num_frames_to_generate
        output = torch.zeros(
            [batch_size, total_output_frames, num_channels, height, width],
            device=context_latent.device,
            dtype=context_latent.dtype
        )
        output[:, :context_frames] = context_latent
        
        # Initialize KV and cross-attention caches. 42 frame length.
        self._initialize_kv_cache(
            batch_size=batch_size, dtype=context_latent.dtype, device=context_latent.device
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=context_latent.dtype, device=context_latent.device
        )
        # Rerun with timestep zero to update the cache
        # context_timestep = torch.ones_like(timestep) * self.context_noise
        # add context noise
        # denoised_pred = self.scheduler.add_noise(
        #     denoised_pred.flatten(0, 1),
        #     torch.randn_like(denoised_pred.flatten(0, 1)),
        #     context_timestep * torch.ones(
        #         [batch_size * current_num_frames], device=context_latent.device, dtype=torch.long)
        # ).unflatten(0, denoised_pred.shape[:2])
        # with torch.no_grad():
        #     self.generator(
        #         noisy_image_or_video=denoised_pred,
        #         conditional_dict=conditional_dict,
        #         timestep=context_timestep,
        #         kv_cache=self.kv_cache1,
        #         crossattn_cache=self.crossattn_cache,
        #         current_start=current_start_frame * self.frame_seq_length
        #     )

        # --- 2. Context Priming (Always without gradients) ---
        # We never need to backpropagate through the initial context.
        current_start_frame = 0
        with torch.no_grad(): # Context priming never requires gradients
            context_blocks = torch.split(context_latent, self.num_frame_per_block, dim=1)
            for block in context_blocks:
                if block.shape[1] == 0: continue
                current_num_frames_in_block = block.shape[1]
                timestep = torch.zeros([batch_size, current_num_frames_in_block], device=context_latent.device, dtype=torch.int64)
                
                self.generator(
                    noisy_image_or_video=block,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )
                current_start_frame += current_num_frames_in_block

        # --- 3. Autoregressive Generation Loop (with Training Logic) ---
        
        # 3.1 Generate random exit flags for stochastic gradient truncation (from reference)
        if not teacher_forcing:
            num_denoising_steps = len(self.denoising_step_list)
            exit_flags = self.generate_and_sync_list(num_blocks_to_generate, num_denoising_steps, device=noise.device)

        for i in range(num_blocks_to_generate):
            current_num_frames = self.num_frame_per_block
            
            start_noise_frame = i * current_num_frames
            end_noise_frame = start_noise_frame + current_num_frames
            noisy_input = noise[:, start_noise_frame:end_noise_frame]

            # 3.2 Denoising Loop with random exit for backpropagation
            denoised_pred = None
            if teacher_forcing:
                # --- Teacher Forcing Path: Full, Deterministic Denoising ---
                # This loop is simple and always runs to completion.
                for index, current_timestep in enumerate(self.denoising_step_list):
                    timestep = torch.ones(
                        [batch_size, current_num_frames],
                        device=noise.device,
                        dtype=torch.int64) * current_timestep
                    
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )
                    
                    if index < len(self.denoising_step_list) - 1:
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])
            
            else: # Self-Forcing / DMD Path
                # --- Self-Forcing Path: Stochastic Denoising with Exit Flags ---
                # This is the complex logic you already have
                for index, current_timestep in enumerate(self.denoising_step_list):
                    if self.same_step_across_blocks:
                        exit_flag = (index == exit_flags[0])
                    else:
                        exit_flag = (index == exit_flags[i])  # Only backprop at the randomly selected timestep (consistent across all ranks)
                    timestep = torch.ones(
                        [batch_size, current_num_frames],
                        device=noise.device,
                        dtype=torch.int64) * current_timestep

                    if not exit_flag:
                        with torch.no_grad():
                            _, denoised_pred = self.generator(
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
                        # with torch.set_grad_enabled(current_start_frame >= start_gradient_frame_index):
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                        break
                

            # 3.3 Record output
            output[:, current_start_frame : current_start_frame + current_num_frames] = denoised_pred
            
            # 3.4 Update KV cache (always with no_grad)
            
            context_timestep = torch.ones(
                [batch_size, current_num_frames], 
                device=context_latent.device, 
                dtype=torch.long
            ) * self.context_noise * 0

            flat_denoised_pred_for_cache = denoised_pred.detach().flatten(0, 1)
            timestep_for_cache_update = context_timestep.flatten() # Correctly shaped

            noisy_flat_input_for_cache = self.scheduler.add_noise(
                flat_denoised_pred_for_cache,
                torch.randn_like(flat_denoised_pred_for_cache),
                timestep_for_cache_update
            )

            denoised_pred_for_cache = noisy_flat_input_for_cache.unflatten(0, (batch_size, current_num_frames))

            # denoised_pred_for_cache = self.scheduler.add_noise(
            #     denoised_pred.detach(), # detach() to cut graph before re-noising for cache
            #     torch.randn_like(denoised_pred),
            #     context_timestep * torch.ones(
            #         [batch_size * current_num_frames], device=context_latent.device, dtype=torch.long)
            # ).unflatten(0, denoised_pred.shape[:2])
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised_pred_for_cache,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )

            # 3.5 Update frame index
            current_start_frame += current_num_frames
        
        # --- 4. Calculate Return Values based on random exit_flags ---
        # This logic is now copied exactly from the reference function

        if teacher_forcing:
            if return_sim_step:
                return output, None, None, None
            else:
                return output, None, None
        else:
            if not self.same_step_across_blocks:
                denoised_timestep_from, denoised_timestep_to = None, None
            elif exit_flags[0] == len(self.denoising_step_list) - 1:
                denoised_timestep_to = 0
                denoised_timestep_from = 1000 - torch.argmin(
                    (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()
            else:
                denoised_timestep_to = 1000 - torch.argmin(
                    (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0).item()
                denoised_timestep_from = 1000 - torch.argmin(
                    (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()
            
            # --- 5. Return Results ---
            if return_sim_step:
                # Assuming same_step_across_blocks=True for simplicity
                num_simulation_steps = exit_flags[0].item() + 1
                return output, denoised_timestep_from, denoised_timestep_to, num_simulation_steps
            else:
                return output, denoised_timestep_from, denoised_timestep_to


    def inference_with_trajectory(
            self,
            noise: torch.Tensor,
            context_teacher: bool = False,
            initial_latent: Optional[torch.Tensor] = None,
            return_sim_step: bool = False,
            **conditional_dict
    ) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        if context_teacher:
            output = torch.zeros(
                [batch_size, num_output_frames, num_channels, height, width],
                device=noise.device,
                dtype=noise.dtype
            )
        else:
            output = torch.zeros(
                [batch_size, num_output_frames, num_channels, height, width],
                device=noise.device,
                dtype=noise.dtype
            )

        # Step 1: Initialize KV cache to all zeros
        self._initialize_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        # if self.kv_cache1 is None:
        #     self._initialize_kv_cache(
        #         batch_size=batch_size,
        #         dtype=noise.dtype,
        #         device=noise.device,
        #     )
        #     self._initialize_crossattn_cache(
        #         batch_size=batch_size,
        #         dtype=noise.dtype,
        #         device=noise.device
        #     )
        # else:
        #     # reset cross attn cache
        #     for block_index in range(self.num_transformer_blocks):
        #         self.crossattn_cache[block_index]["is_init"] = False
        #     # reset kv cache
        #     for block_index in range(len(self.kv_cache1)):
        #         self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
        #             [0], dtype=torch.long, device=noise.device)
        #         self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
        #             [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
            output[:, :1] = initial_latent
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )
            current_start_frame += 1

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)
        start_gradient_frame_index = num_output_frames - self.real_score_length + 3

        print("start_gradient_frame_index", start_gradient_frame_index)

        # for block_index in range(num_blocks):
        for block_index, current_num_frames in enumerate(all_num_frames):
            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                if self.same_step_across_blocks:
                    exit_flag = (index == exit_flags[0])
                else:
                    exit_flag = (index == exit_flags[block_index])  # Only backprop at the randomly selected timestep (consistent across all ranks)
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep
                
                # for I2V need have clean denoise result, for context teacher we need last 5s to be clean
                # print("current_num_frames:", current_start_frame, "start_gradient_frame_index", start_gradient_frame_index)
                # For context model, we need to fully denoise the context video part
                if context_teacher and current_start_frame <= start_gradient_frame_index and current_start_frame >= start_gradient_frame_index - self.context_window:
                    exit_flag = (index == num_denoising_steps - 1)
                # exit_flag = (index == num_denoising_steps - 1)
                # Hard code here, all of frame fully denoised
                if self.all_fully_denoised and current_start_frame <= start_gradient_frame_index:
                    exit_flag = (index == num_denoising_steps - 1)

                if not exit_flag:
                    with torch.no_grad():
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
                    # with torch.set_grad_enabled(current_start_frame >= start_gradient_frame_index):
                    if current_start_frame < start_gradient_frame_index:
                        with torch.no_grad():
                            _, denoised_pred, self.kv_cache1 = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length
                            )
                    else:
                        _, denoised_pred, self.kv_cache1 = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                    break

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update the cache
            context_timestep = torch.ones_like(timestep) * self.context_noise
            # add context noise
            denoised_pred = self.scheduler.add_noise(
                denoised_pred.detach().flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep * torch.ones(
                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
            ).unflatten(0, denoised_pred.shape[:2])

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
                    sink_tokens_count = self.sink_size * self.frame_seq_length 
                    
                    sink_k = self.kv_cache1["k"][:, :sink_tokens_count].clone()
                    sink_v = self.kv_cache1["v"][:, :sink_tokens_count].clone()
                    
                    sink_end_idx = torch.tensor([sink_tokens_count], dtype=torch.long, device=self.kv_cache1["local_end_index"].device)
                    
                    sink_cache = {
                        "k": sink_k,
                        "v": sink_v,
                        "global_end_index": sink_end_idx.clone(), 
                        "local_end_index": sink_end_idx.clone()   
                    }

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

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        # Step 3.5: Return the denoised timestep
        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1
        
        # for long rolling out
        # if output.shape[1] > 21:
        #     output = output[:, -21:, :, :, :]


        return output, denoised_timestep_from, denoised_timestep_to

    
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
        print("Creat KV Cache, size ", self.kv_cache_size)
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