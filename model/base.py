from typing import Tuple
from einops import rearrange
from torch import nn
import torch.distributed as dist
import torch
import os

from pipeline import SelfForcingTrainingPipeline
from utils.loss import get_denoising_loss
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from wan.modules.clip import CLIPModel
from safetensors.torch import load_file  
# from stable_video_infinity.diffsynth import ModelManager
import glob

class BaseModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self._initialize_models(args, device)

        self.device = device
        self.args = args
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        if hasattr(args, "denoising_step_list"):
            self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)
            if args.warp_denoising_step:
                timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
                self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

    def _initialize_models(self, args, device):
        self.real_model_name = getattr(args, "real_name", "Wan2.1-T2V-1.3B")
        self.fake_model_name = getattr(args, "fake_name", "Wan2.1-T2V-1.3B")
        self.fake_model_type = getattr(args, "fake_model_type", "self_attn")
        self.return_attn_map = getattr(args, "return_attn_map", False)
        self.real_score_length = getattr(args, "real_score_length", 21)
        self.context_window = getattr(args, "context_window", 21)
        self.decay_steps = getattr(args, "decay_steps", 1000)
        self.initial_bias_exponent = getattr(args, "initial_bias_exponent", 3)

        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True, return_attn_map=self.return_attn_map)
        self.generator.model.requires_grad_(True)

        if self.real_model_name.split("-")[0] == "Wan2.1":
            self.real_score = WanDiffusionWrapper(model_name=self.real_model_name, is_causal=False)
            self.real_score.model.requires_grad_(False)

        if self.fake_model_name.split("-")[0] == "Wan2.1":
            self.fake_score = WanDiffusionWrapper(model_name=self.fake_model_name, is_causal=False)
            self.fake_score.model.requires_grad_(True)

        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)

        self._context_teacher = getattr(args, "context_teacher", False)
        self._context_fake = False
        if self._context_teacher: # only use context fake when use context teacher
            self._context_fake = getattr(args, "context_fake", False)

        # load context teacher model
        context_teacher_path = getattr(args, "context_teacher_path", None)
        fake_context_teacher_path = getattr(args, "fake_context_teacher_path", None)
        if context_teacher_path is None:
            raise ValueError("The required argument 'context_teacher_path' was not provided. Please specify the path.")

        # svi model lora load
        elif hasattr(args, "svi_model") and getattr(args.svi_model, "lora", False):
            from stable_video_infinity.diffsynth import ModelManager
            model_manager = ModelManager(device="cpu", train_architecture='lora')
            if args.svi_model.model_type == "Wan2.1-T2V-1.3B":
                model_manager.load_models([
                [
                    "wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
                    ],
            ])
            elif args.svi_model.model_type == "Wan2.1-T2V-14B":
                model_manager.load_models([
                    [
                        "wan_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors",
                        "wan_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors",
                        "wan_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors",
                        "wan_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors",
                        "wan_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors",
                        "wan_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors",
                        ],
                    "wan_models/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth",
                    "wan_models/Wan2.1-T2V-14B/Wan2.1_VAE.pth",
                ])
            extra_module_root = args.svi_model.lora_path

            if extra_module_root.endswith('.safetensors'):
                safetensors_files = [extra_module_root]
            else:
                safetensors_files = glob.glob(os.path.join(extra_module_root, "*.safetensors"))
                safetensors_files.sort()

            model_manager.load_lora_v2(safetensors_files, lora_alpha=args.svi_model.lora_alpha)

            dit_model_name = 'wan_video_dit'
            if dit_model_name in model_manager.model_name:
                dit_index = model_manager.model_name.index(dit_model_name)
                dit_model = model_manager.model[dit_index]

                merged_dit_state_dict = dit_model.state_dict()
                
                self.real_score.model.load_state_dict(merged_dit_state_dict, strict=False, assign=True)
                self.real_score.model.requires_grad_(False)
                print("Loaded context teacher (with LoRA).")

                if self.context_fake:
                    self.fake_score.model.load_state_dict(merged_dit_state_dict, strict=False, assign=True)
                    self.fake_score.model.requires_grad_(True)
                    print("Loaded context fake score (with LoRA).")
                else:
                    print("Loaded context fake score (without LoRA).")

                print("Converting models to bfloat16...")
                self.real_score.model.to(dtype=torch.bfloat16)
                self.fake_score.model.to(dtype=torch.bfloat16)

                self.real_score.model.requires_grad_(False)
                self.fake_score.model.requires_grad_(True)

                print("Teacher Model Dtype:", next(self.real_score.model.parameters()).dtype)
                print("Critic Model Dtype:", next(self.fake_score.model.parameters()).dtype)
                del model_manager
                del merged_dit_state_dict
        else:
            if self.real_model_name.split("-")[0] == "Wan2.1":
                self.real_score_context_keys = None
                state_dict = load_file(context_teacher_path, device="cpu") 
                
                self.real_score.model.load_state_dict(state_dict, strict=False, assign=True)
                self.real_score.model.requires_grad_(False)
                print("Loaded context teacher (safetensors).")

            if self.fake_model_name.split("-")[0] == "Wan2.1":
                fake_state_dict = load_file(fake_context_teacher_path, device="cpu")
                
                self.fake_score.model.load_state_dict(fake_state_dict, strict=False, assign=True)
                self.fake_score.model.requires_grad_(True)
                print("Loaded context fake score (safetensors).")

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def _get_timestep(
            self,
            min_timestep: int,
            max_timestep: int,
            batch_size: int,
            num_frame: int,
            num_frame_per_block: int,
            uniform_timestep: bool = False
    ) -> torch.Tensor:
        """
        Randomly generate a timestep tensor based on the generator's task type. It uniformly samples a timestep
        from the range [min_timestep, max_timestep], and returns a tensor of shape [batch_size, num_frame].
        - If uniform_timestep, it will use the same timestep for all frames.
        - If not uniform_timestep, it will use a different timestep for each block.
        """
        if uniform_timestep:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, 1],
                device=self.device,
                dtype=torch.long
            ).repeat(1, num_frame)
            return timestep
        else:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, num_frame],
                device=self.device,
                dtype=torch.long
            )
            # make the noise level the same within every block
            if self.independent_first_frame:
                # the first frame is always kept the same
                timestep_from_second = timestep[:, 1:]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1, num_frame_per_block)
                timestep_from_second[:, :, 1:] = timestep_from_second[:, :, 0:1]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1)
                timestep = torch.cat([timestep[:, 0:1], timestep_from_second], dim=1)
            else:
                timestep = timestep.reshape(
                    timestep.shape[0], -1, num_frame_per_block)
                timestep[:, :, 1:] = timestep[:, :, 0:1]
                timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep

    def _switch_model_params(model, target_params):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in target_params:
                    param.data.copy_(target_params[name])


class SelfForcingModel(BaseModel):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.denoising_loss_func = get_denoising_loss(args.denoising_loss_type)()

    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        initial_latent: torch.tensor = None,
        step: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optionally simulate the generator's input from noise using backward simulation
        and then run the generator for one-step.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
            - initial_latent: a tensor containing the initial latents [B, F, C, H, W].
        Output:
            - pred_image: a tensor with shape [B, F, C, H, W].
            - denoised_timestep: an integer
        """
        # Step 1: Sample noise and backward simulate the generator's input
        assert getattr(self.args, "backward_simulation", True), "Backward simulation needs to be enabled"
        if initial_latent is not None:
            conditional_dict["initial_latent"] = initial_latent
        if self.args.i2v:
            noise_shape = [image_or_video_shape[0], image_or_video_shape[1] - 1, *image_or_video_shape[2:]]
        else:
            noise_shape = image_or_video_shape.copy()

        # During training, the number of generated frames should be uniformly sampled from
        # [21, self.num_training_frames], but still being a multiple of self.num_frame_per_block
        # min_num_frames = 20 if self.args.independent_first_frame else 21
        min_num_frames = self.num_min_training_frames - 1 if self.args.independent_first_frame else self.num_min_training_frames
        max_num_frames = self.num_training_frames - 1 if self.args.independent_first_frame else self.num_training_frames
        assert max_num_frames % self.num_frame_per_block == 0
        assert min_num_frames % self.num_frame_per_block == 0
        max_num_blocks = max_num_frames // self.num_frame_per_block
        min_num_blocks = min_num_frames // self.num_frame_per_block

        rollout_sample_rule = self.args.get("rollout_sample_rule", "default")
        if rollout_sample_rule == "linear":
            progress = min(1.0, (step - 0) / self.decay_steps)
            current_max_blocks_float = min_num_blocks + progress * (max_num_blocks - min_num_blocks)
            
            current_max_blocks = int(current_max_blocks_float)
            num_generated_blocks = torch.randint(min_num_blocks, current_max_blocks + 1, (1,), device=self.device)
        elif rollout_sample_rule == "exp":
            progress = min(1.0, step / self.decay_steps)
            exponent = 1.0 + (self.initial_bias_exponent - 1.0) * (1.0 - progress)
            weighted_rand = torch.rand((1,), device=self.device) ** exponent
            num_generated_blocks = min_num_blocks + (weighted_rand * (max_num_blocks - min_num_blocks + 1)).floor().long()
        else: 
            num_generated_blocks = torch.randint(min_num_blocks, max_num_blocks + 1, (1,), device=self.device)

        dist.broadcast(num_generated_blocks.detach(), src=0)
        num_generated_blocks = num_generated_blocks.item()
        num_generated_frames = num_generated_blocks * self.num_frame_per_block
        if self.args.independent_first_frame and initial_latent is None:
            num_generated_frames += 1
            min_num_frames += 1
        # Sync num_generated_frames across all processes
        noise_shape[1] = num_generated_frames
        print("self._context_teacher", self._context_teacher, "self.num_training_frames", self.num_training_frames, "num_generated_frames", num_generated_frames)

        pred_image_or_video, denoised_timestep_from, denoised_timestep_to = self._consistency_backward_simulation(
            noise=torch.randn(noise_shape,
                              device=self.device, dtype=self.dtype),
            **conditional_dict,
        )

        # Slice last 21 frames
        if pred_image_or_video.shape[1] > self.real_score_length:
            with torch.no_grad():
                # Reencode to get image latent
                if self._context_teacher:
                    latent_to_decode = pred_image_or_video[:, :-(self.real_score_length + self.context_window - 1), ...]
                else:
                    latent_to_decode = pred_image_or_video[:, :-(self.real_score_length - 1), ...]
                # Deccode to video
                pixels = self.vae.decode_to_pixel(latent_to_decode)
                # pred_image_or_video_debug = self.vae.decode_to_pixel(pred_image_or_video)
                # pixels_debug = rearrange(pixels, "b t c h w -> b c t h w")
                # pixels_debug = (pixels * 0.5 + 0.5).clamp(0, 1)
                # pred_image_or_video_debug = (pred_image_or_video_debug * 0.5 + 0.5).clamp(0, 1)
                # print("latent_to_decode", latent_to_decode.shape, "pixels_debug", pixels_debug.shape)

                # from torchvision.utils import save_image
                # os.makedirs("debug", exist_ok=True)
                # for i in range(pixels_debug.shape[1]):
                #     save_image(pixels_debug[:,i], f'debug/video_i2v_pixel{i}.png')
                
                # for i in range(pred_image_or_video_debug.shape[1]):
                #     save_image(pred_image_or_video_debug[:,i], f'debug/video_i2v_pred_image_or_video_debug{i}.png')

                frame = pixels[:, -1:, ...].to(self.dtype)
                frame = rearrange(frame, "b t c h w -> b c t h w")
                # Encode frame to get image latent
                image_latent = self.vae.encode_to_latent(frame).to(self.dtype)
            if self._context_teacher:
                pred_image_or_video_last_21 = torch.cat([image_latent, pred_image_or_video[:, -(self.real_score_length + self.context_window - 1):, ...]], dim=1)
            else:
                pred_image_or_video_last_21 = torch.cat([image_latent, pred_image_or_video[:, -(self.real_score_length - 1):, ...]], dim=1)
        else:
            pred_image_or_video_last_21 = pred_image_or_video
            frame = None

        if num_generated_frames != min_num_frames and not self._context_teacher:
            # Currently, we do not use gradient for the first chunk, since it contains image latents
            gradient_mask = torch.ones_like(pred_image_or_video_last_21, dtype=torch.bool)
            if self.args.independent_first_frame:
                gradient_mask[:, :1] = False
            else:
                gradient_mask[:, :self.num_frame_per_block] = False
        else:
            gradient_mask = None

        pred_image_or_video_last_21 = pred_image_or_video_last_21.to(self.dtype)
        return pred_image_or_video_last_21, gradient_mask, denoised_timestep_from, denoised_timestep_to, frame

    def _run_generator_long(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        context_latent: torch.tensor = None,
        initial_latent: torch.tensor = None,
        w_context = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optionally simulate the generator's input from noise using backward simulation
        and then run the generator for one-step.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
            - initial_latent: a tensor containing the initial latents [B, F, C, H, W].
        Output:
            - pred_image: a tensor with shape [B, F, C, H, W].
            - denoised_timestep: an integer
        """
        # Step 1: Sample noise and backward simulate the generator's input
        assert getattr(self.args, "backward_simulation", True), "Backward simulation needs to be enabled"
        if initial_latent is not None:
            conditional_dict["initial_latent"] = initial_latent
        if self.args.i2v:
            noise_shape = [image_or_video_shape[0], image_or_video_shape[1] - 1, *image_or_video_shape[2:]]
        else:
            noise_shape = image_or_video_shape.copy()

        # During training, the number of generated frames should be uniformly sampled from
        # [21, self.num_training_frames], but still being a multiple of self.num_frame_per_block
        min_num_frames = 20 if self.args.independent_first_frame else 21
        max_num_frames = self.num_training_frames - 1 if self.args.independent_first_frame else self.num_training_frames
        assert max_num_frames % self.num_frame_per_block == 0
        assert min_num_frames % self.num_frame_per_block == 0
        max_num_blocks = max_num_frames // self.num_frame_per_block
        min_num_blocks = min_num_frames // self.num_frame_per_block
        num_generated_blocks = torch.randint(min_num_blocks, max_num_blocks + 1, (1,), device=self.device)
        dist.broadcast(num_generated_blocks, src=0)
        num_generated_blocks = num_generated_blocks.item()
        num_generated_frames = num_generated_blocks * self.num_frame_per_block
        if self.args.independent_first_frame and initial_latent is None:
            num_generated_frames += 1
            min_num_frames += 1
        # Sync num_generated_frames across all processes

        noise_shape[1] = num_generated_frames
        # print(self.args, self.args.independent_first_frame)
        print(num_generated_frames)

        if w_context:
            pred_image_or_video, denoised_timestep_from, denoised_timestep_to = self._consistency_backward_simulation_with_context(
                context_latent=context_latent, num_frames_to_generate=num_generated_frames, teacher_forcing=self.args.teacher_forcing, **conditional_dict,)
        else:
            pred_image_or_video, denoised_timestep_from, denoised_timestep_to = self._consistency_backward_simulation(
                noise=torch.randn(noise_shape,
                                device=self.device, dtype=self.dtype),
                **conditional_dict,
            )

        # Slice last 21 frames
        if pred_image_or_video.shape[1] > 21:
            with torch.no_grad():
                # Reencode to get image latent
                latent_to_decode = pred_image_or_video[:, :-20, ...]
                # Deccode to video
                vae_dtype = next(self.vae.parameters()).dtype
                pixels = self.vae.decode_to_pixel(latent_to_decode.to(dtype=vae_dtype))
                frame = pixels[:, -1:, ...].to(self.dtype)
                frame = rearrange(frame, "b t c h w -> b c t h w")
                # Encode frame to get image latent
                image_latent = self.vae.encode_to_latent(frame).to(self.dtype)
            pred_image_or_video_last_21 = torch.cat([image_latent, pred_image_or_video[:, -20:, ...]], dim=1)
        else:
            pred_image_or_video_last_21 = pred_image_or_video

        if num_generated_frames != min_num_frames:
            # Currently, we do not use gradient for the first chunk, since it contains image latents
            gradient_mask = torch.ones_like(pred_image_or_video_last_21, dtype=torch.bool)
            if self.args.independent_first_frame:
                gradient_mask[:, :1] = False
            else:
                gradient_mask[:, :self.num_frame_per_block] = False
        else:
            gradient_mask = None

        pred_image_or_video_last_21 = pred_image_or_video_last_21.to(self.dtype)
        return pred_image_or_video_last_21, gradient_mask, denoised_timestep_from, denoised_timestep_to

    def _consistency_backward_simulation(
        self,
        noise: torch.Tensor,
        **conditional_dict: dict
    ) -> torch.Tensor:
        """
        Simulate the generator's input from noise to avoid training/inference mismatch.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Here we use the consistency sampler (https://arxiv.org/abs/2303.01469)
        Input:
            - noise: a tensor sampled from N(0, 1) with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - output: a tensor with shape [B, T, F, C, H, W].
            T is the total number of timesteps. output[0] is a pure noise and output[i] and i>0
            represents the x0 prediction at each timestep.
        """
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        return self.inference_pipeline.inference_with_trajectory(
            noise=noise, context_teacher=self._context_teacher, **conditional_dict
        )

    def _initialize_inference_pipeline(self):
        """
        Lazy initialize the inference pipeline during the first backward simulation run.
        Here we encapsulate the inference code with a model-dependent outside function.
        We pass our FSDP-wrapped modules into the pipeline to save memory.
        """
        self.inference_pipeline = SelfForcingTrainingPipeline(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.args.independent_first_frame,
            same_step_across_blocks=self.args.same_step_across_blocks,
            last_step_only=self.args.last_step_only,
            num_max_frames=self.num_training_frames,
            context_noise=self.args.context_noise,
            sink_size=self.args.model_kwargs.sink_size,
            local_attn_size=self.args.model_kwargs.local_attn_size,
            cache_size=self.args.model_kwargs.cache_size,
            all_fully_denoised=self.args.all_fully_denoised,
            context_window=self.context_window,
            real_score_length=self.real_score_length,
        )

    def _consistency_backward_simulation_with_context(
        self,
        context_latent: torch.Tensor,
        num_frames_to_generate,
        teacher_forcing=False,
        **conditional_dict: dict
    ) -> torch.Tensor:
        """
        Simulate the generator's input from noise to avoid training/inference mismatch.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Here we use the consistency sampler (https://arxiv.org/abs/2303.01469)
        Input:
            - noise: a tensor sampled from N(0, 1) with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - output: a tensor with shape [B, T, F, C, H, W].
            T is the total number of timesteps. output[0] is a pure noise and output[i] and i>0
            represents the x0 prediction at each timestep.
        """
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        return self.inference_pipeline.predict_from_context(
            context_latent=context_latent, num_frames_to_generate=num_frames_to_generate, teacher_forcing=teacher_forcing,
            **conditional_dict
        )
