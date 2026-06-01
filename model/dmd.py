from pipeline import SelfForcingTrainingPipeline
import torch.nn.functional as F
from typing import Optional, Tuple
import torch
import os
import gc
from einops import rearrange
from torchvision.io import write_video

from model.base import SelfForcingModel
from utils.scheduler import SchedulerInterface, FlowMatchScheduler

def save_latent_as_video(
    latent_tensor,
    vae, 
    base_output_path,
    step,
    video_name,
    device_id,
    fps=16,
):
    """
    Decodes a latent tensor using a VAE and saves the resulting video as an MP4 file.

    Args:
        latent_tensor (torch.Tensor): The latent tensor to be decoded and saved.
        vae (Any): The VAE model instance, which must have a `decode_to_pixel` method.
        base_output_path (str): The root directory to save videos in. A 'saved_video' subfolder
                                will be created here.
        step (int): The current training step, used in the filename.
        video_name (str): A descriptive name for the video (e.g., 'pred_real', 'pred_fake').
        device_id (int or str): The device ID, used in the filename.
        fps (int, optional): The frames per second for the output video. Defaults to 16.
    """
    with torch.no_grad():
        # 1. Decode the latent tensor to pixel space
        video_tensor = vae.decode_to_pixel(latent_tensor.to(torch.bfloat16))
        
        # 2. Normalize from [-1, 1] to [0, 1]
        video_tensor = (video_tensor * 0.5 + 0.5).clamp(0, 1)

    # 3. Clear VAE cache if the method exists (for stability)
    if hasattr(vae, 'model') and hasattr(vae.model, 'clear_cache') and callable(vae.model.clear_cache):
        vae.model.clear_cache()

    # 4. Reshape from (B, T, C, H, W) to (B, T, H, W, C) for video writing
    video_tensor_hwc = rearrange(video_tensor, 'b t c h w -> b t h w c').cpu()

    # 5. Select the first video in the batch
    video_to_save = video_tensor_hwc[0]
    
    # 6. Scale from [0, 1] float to [0, 255] uint8
    video_to_save = (255.0 * video_to_save).clamp(0, 255).to(torch.uint8)

    # 7. Create the output directory and save the file
    output_dir = os.path.join(base_output_path, "saved_video")
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"step_{step}_{video_name}_{device_id}.mp4")
    write_video(file_path, video_to_save, fps=fps)

class DMD(SelfForcingModel):
    def __init__(self, args, device):
        """
        Initialize the DMD (Distribution Matching Distillation) module.
        This class is self-contained and compute generator and fake score losses
        in the forward pass.
        """
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.same_step_across_blocks = getattr(args, "same_step_across_blocks", True)
        self.num_training_frames = getattr(args, "num_training_frames", 21)
        self.num_min_training_frames = getattr(args, "num_min_training_frames", 21)
        self.context_window = getattr(args, "context_window", 21)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

        # this will be init later with fsdp-wrapped modules
        self.inference_pipeline: SelfForcingTrainingPipeline = None

        # Step 2: Initialize all dmd hyperparameters
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep) # for context training use lower timestep
        if hasattr(args, "real_guidance_scale"):
            self.real_guidance_scale = args.real_guidance_scale
            self.fake_guidance_scale = args.fake_guidance_scale
        else:
            self.real_guidance_scale = args.guidance_scale
            self.fake_guidance_scale = 0.0
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.ts_schedule = getattr(args, "ts_schedule", True)
        self.ts_schedule_max = getattr(args, "ts_schedule_max", False)
        self.min_score_timestep = getattr(args, "min_score_timestep", 0)

        self.teacher_foring = getattr(args, "teacher_foring", False)
        self.vis_per_step = getattr(args, "vis_per_step", 5)
        self._context_teacher = getattr(args, "context_teacher", False)
        self.context_fake = False
        if self._context_teacher:
            self.context_fake = getattr(args, "context_fake", False)

        self.output_path = getattr(args, "logdir", None)

        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None

    def _update_kv_cache_dict(self, kv_cache_dict):
        self.kv_cache_dict = kv_cache_dict

    def _cache_clean_latents_real(self, cond_latents, model_max_length, offload_kv_cache, device, dtype):
        timestep = torch.zeros(cond_latents.shape[0], cond_latents.shape[1]).to(device=device, dtype=dtype)
        # make null prompt tensor(skip_crs_attn=True, so tensors below will not be actually used)
        empty_embeds = torch.zeros([cond_latents.shape[0], 1, model_max_length, 4096], device=device, dtype=dtype)
        _, kv_cache_dict = self.real_score(
            hidden_states=cond_latents.permute(0, 2, 1, 3, 4), 
            timestep=timestep, 
            encoder_hidden_states=empty_embeds,
            return_kv=True, 
            skip_crs_attn=True, 
            offload_kv_cache=offload_kv_cache
        )
        self._update_kv_cache_dict(kv_cache_dict)

    def _cache_clean_latents_fake(self, cond_latents, model_max_length, offload_kv_cache, device, dtype):
        timestep = torch.zeros(cond_latents.shape[0], cond_latents.shape[1]).to(device=device, dtype=dtype)
        # make null prompt tensor(skip_crs_attn=True, so tensors below will not be actually used)
        empty_embeds = torch.zeros([cond_latents.shape[0], 1, model_max_length, 4096], device=device, dtype=dtype)
        _, kv_cache_dict = self.fake_score(
            hidden_states=cond_latents.permute(0, 2, 1, 3, 4), 
            timestep=timestep, 
            encoder_hidden_states=empty_embeds,
            return_kv=True, 
            skip_crs_attn=True, 
            offload_kv_cache=offload_kv_cache
        )
        self._update_kv_cache_dict(kv_cache_dict)
    
    def _get_kv_cache_dict(self):
        return self.kv_cache_dict
    
    def _clear_cache(self):
        self.kv_cache_dict = None
        gc.collect()
        torch.cuda.empty_cache()
    
    def optimized_scale(self, positive_flat, negative_flat):
        """ from CFG-zero paper
        """
        # Calculate dot production
        dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
        # Squared norm of uncondition
        squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8
        # st_star = v_condˆT * v_uncond / ||v_uncond||ˆ2
        st_star = dot_product / squared_norm
        return st_star

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    def _compute_kl_grad(
        self, noisy_image_or_video: torch.Tensor,
        noise: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict, unconditional_dict: dict,
        normalization: bool = True,
        context_video: torch.Tensor=None,
        step: int = 0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the KL grad (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - noisy_image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - estimated_clean_image_or_video: a tensor with shape [B, F, C, H, W] representing the estimated clean image or video.
            - timestep: a tensor with shape [B, F] containing the randomly generated timestep.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - normalization: a boolean indicating whether to normalize the gradient.
        Output:
            - kl_grad: a tensor representing the KL grad.
            - kl_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 0: Computer real context input, noisy_coarse_real_image
        _timestep = torch.cat([timestep, timestep], dim=1)

        # Step 1: Compute the fake score
        if self._context_teacher and self.context_fake and context_video is not None:
            _, pred_fake_image_cond = self.fake_score(
                noisy_image_or_video=torch.cat([context_video, noisy_image_or_video], dim=1),
                conditional_dict={'prompt_embeds': conditional_dict['prompt_embeds']},
                timestep=_timestep,
                context_model=True,
                mode="base",
            )

            if self.fake_guidance_scale != 0.0:
                _, pred_fake_image_uncond = self.fake_score(
                    noisy_image_or_video=torch.cat([context_video, noisy_image_or_video], dim=1),
                    conditional_dict=unconditional_dict,
                    timestep=_timestep,
                    context_model=True,
                    mode="base",
                )
                pred_fake_image = pred_fake_image_cond + (
                    pred_fake_image_cond - pred_fake_image_uncond
                ) * self.fake_guidance_scale
            else:
                pred_fake_image = pred_fake_image_cond
        else:
            _, pred_fake_image_cond = self.fake_score(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict={'prompt_embeds': conditional_dict['prompt_embeds']},
                timestep=timestep,
                mode="base"
            )

            if self.fake_guidance_scale != 0.0:
                _, pred_fake_image_uncond = self.fake_score(
                    noisy_image_or_video=noisy_image_or_video,
                    conditional_dict=unconditional_dict,
                    timestep=timestep,
                    mode="base"
                )
                pred_fake_image = pred_fake_image_cond + (
                    pred_fake_image_cond - pred_fake_image_uncond
                ) * self.fake_guidance_scale
            else:
                pred_fake_image = pred_fake_image_cond

        # Step 2: Compute the real score
        # We compute the conditional and unconditional prediction
        # If is i2v we send the last frame to the real model
        # and add them together to achieve cfg (https://arxiv.org/abs/2207.12598)
        if self._context_teacher and context_video is not None:
            _, pred_real_image_cond = self.real_score(
                noisy_image_or_video=torch.cat([context_video, noisy_image_or_video], dim=1),
                conditional_dict={'prompt_embeds': conditional_dict['prompt_embeds']},
                timestep=_timestep,
                context_model=True,
                mode="base",
            )

            _, pred_real_image_uncond = self.real_score(
                noisy_image_or_video=torch.cat([context_video, noisy_image_or_video], dim=1),
                conditional_dict=unconditional_dict,
                timestep=_timestep,
                context_model=True,
                mode="base",
            )

        else:
            _, pred_real_image_cond = self.real_score(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict={'prompt_embeds': conditional_dict['prompt_embeds']},
                timestep=timestep,
                mode="base"
            )

            _, pred_real_image_uncond = self.real_score(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=unconditional_dict,
                timestep=timestep,
                mode="base"
            )

        pred_real_image = pred_real_image_cond + (
            pred_real_image_cond - pred_real_image_uncond
        ) * self.real_guidance_scale

        if step % self.vis_per_step == 0:
            save_latent_as_video(
                    latent_tensor=torch.cat([context_video, estimated_clean_image_or_video], dim=1) if self._context_teacher else estimated_clean_image_or_video,
                    vae=self.vae,
                    base_output_path=self.output_path,
                    step=step,
                    video_name="student_generated_video",
                    device_id=self.device
                )

            save_latent_as_video(
                    latent_tensor=torch.cat([context_video, noisy_image_or_video], dim=1) if self._context_teacher else noisy_image_or_video,
                    vae=self.vae,
                    base_output_path=self.output_path,
                    step=step,
                    video_name="pred_real_image_input",
                    device_id=self.device
                )

            save_latent_as_video(
                latent_tensor=torch.cat([context_video, pred_real_image], dim=1).to(dtype=self.dtype) if self._context_teacher else pred_real_image,
                vae=self.vae,
                base_output_path=self.output_path,
                step=step,
                video_name="pred_real_image",
                device_id=self.device
            )
            
            save_latent_as_video(
                latent_tensor=pred_fake_image,
                vae=self.vae,
                base_output_path=self.output_path,
                step=step,
                video_name="pred_fake_image",
                device_id=self.device
            )

            save_latent_as_video(
                latent_tensor=(pred_fake_image - pred_real_image),
                vae=self.vae,
                base_output_path=self.output_path,
                step=step,
                video_name="grad",
                device_id=self.device
            )

        # Step 3: Compute the DMD gradient (DMD paper eq. 7).
        if self._context_teacher and context_video is not None:
            if self.context_fake:
                grad = (pred_fake_image[:, noisy_image_or_video.shape[1]:] - pred_real_image[:, noisy_image_or_video.shape[1]:])
            else:
                grad = (pred_fake_image - pred_real_image[:, noisy_image_or_video.shape[1]:])
        else:
            grad = (pred_fake_image - pred_real_image)

        if normalization:
            # Step 4: Gradient normalization (DMD paper eq. 8).
            if self._context_teacher and context_video is not None:
                p_real = (estimated_clean_image_or_video - pred_real_image[:, noisy_image_or_video.shape[1]:])
            else:
                p_real = (estimated_clean_image_or_video - pred_real_image)
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / normalizer
        grad = torch.nan_to_num(grad)
        grad = grad.detach()

        return grad, {
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep.detach()
        }

    def compute_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0,
        step: int = 0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the DMD loss (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - gradient_mask: a boolean tensor with the same shape as image_or_video indicating which pixels to compute loss .
        Output:
            - dmd_loss: a scalar tensor representing the DMD loss.
            - dmd_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        batch_size, num_frame = image_or_video.shape[:2]

        if self._context_teacher:
            # num_frame = image_or_video.shape[1] // 2
            context_video = image_or_video[:, :self.context_window, ...]
            original_latent = image_or_video[:, self.context_window:, ...]
            num_frame = original_latent.shape[1]
            print("tgt_latent_len", num_frame, "context_video shape", context_video.shape)
        else:
            original_latent = image_or_video
            context_video = None

        with torch.no_grad():
            # Step 1: Randomly sample timestep based on the given schedule and corresponding noise
            min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
            max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
            timestep = self._get_timestep(
                min_timestep,
                max_timestep,
                batch_size,
                num_frame,
                self.num_frame_per_block,
                uniform_timestep=True
            )

            # TODO:should we change it to `timestep = self.scheduler.timesteps[timestep]`?
            if self.timestep_shift > 1:
                timestep = self.timestep_shift * \
                    (timestep / 1000) / \
                    (1 + (self.timestep_shift - 1) * (timestep / 1000)) * 1000
            timestep = timestep.clamp(self.min_step, self.max_step)

            noise = torch.randn_like(original_latent)
            noisy_latent = self.scheduler.add_noise(
                original_latent.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1)
            ).detach().unflatten(0, (batch_size, num_frame))

            # Step 2: Compute the KL grad
            grad, dmd_log_dict = self._compute_kl_grad(
                noisy_image_or_video=noisy_latent,
                noise=noise,
                estimated_clean_image_or_video=original_latent,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                context_video=context_video,
                step=step,
            )

            print("grad shape", grad.shape)

        #     original_latent = torch.cat([context_video, original_latent], dim=1)
        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            )[gradient_mask], (original_latent.double() - grad.double()).detach()[gradient_mask], reduction="mean")
        else:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            ), (original_latent.double() - grad.double()).detach(), reduction="mean")
        return dmd_loss, dmd_log_dict

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        context_latent: torch.Tensor = None,
        target_latent: torch.Tensor = None,
        initial_latent: torch.Tensor = None,
        w_context = False,
        teacher_forcing = False,
        step = 0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        print("Using DMD loss.")
        # Step 1: Unroll generator to obtain fake videos
        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to, frame = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent,
            step=step
        )

        # Step 2: Compute the DMD loss
        # print("self._context_teacher....", self._context_teacher)
        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
            step=step,
        )

        return dmd_loss, dmd_log_dict

    def critic_loss(
            self,
            image_or_video_shape,
            conditional_dict: dict,
            unconditional_dict: dict,
            clean_latent: torch.Tensor,
            initial_latent: torch.Tensor = None,
            step: int = 0,
        ) -> Tuple[torch.Tensor, dict]:
            """
            Generate image/videos from noise and train the critic with generated samples.
            The noisy input to the generator is backward simulated.
            This removes the need of any datasets during distillation.
            See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
            Input:
                - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
                - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
                - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
                - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
            Output:
                - loss: a scalar tensor representing the generator loss.
                - critic_log_dict: a dictionary containing the intermediate tensors for logging.
            """
            # Step 1: Run generator on backward simulated noisy input
            with torch.no_grad():
                generated_image, _, denoised_timestep_from, denoised_timestep_to, _ = self._run_generator(
                    image_or_video_shape=image_or_video_shape,
                    conditional_dict=conditional_dict,
                    initial_latent=initial_latent,
                    step=step,
                )
            # print("_generated_image shape, image_or_video_shape shape", _generated_image.shape, image_or_video_shape)

            if (self._context_teacher and self.context_fake):
                image_or_video_shape[1] *= 2 

            if self._context_teacher and not self.context_fake:
                context_video = generated_image[:, :self.context_window, ...]
                generated_image = generated_image[:, self.context_window:, ...]
            else:
                context_video = None
            # else:
            #     generated_image = _generated_image

            # Step 2: Compute the fake prediction
            min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
            max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
            critic_timestep = self._get_timestep(
                min_timestep,
                max_timestep,
                image_or_video_shape[0],
                image_or_video_shape[1],
                self.num_frame_per_block,
                uniform_timestep=True
            )

            if self.timestep_shift > 1:
                critic_timestep = self.timestep_shift * \
                    (critic_timestep / 1000) / (1 + (self.timestep_shift - 1) * (critic_timestep / 1000)) * 1000

            critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

            critic_noise = torch.randn_like(generated_image)
            noisy_generated_image = self.scheduler.add_noise(
                generated_image.flatten(0, 1),
                critic_noise.flatten(0, 1),
                critic_timestep.flatten(0, 1)
            ).unflatten(0, image_or_video_shape[:2])

            if (self._context_teacher and self.context_fake):
                noisy_generated_image[:, :image_or_video_shape[1]//2, ...] = generated_image[:, :image_or_video_shape[1]//2, ...]


            pred_fake_flow, pred_fake_image = self.fake_score(
                noisy_image_or_video=noisy_generated_image,
                conditional_dict=conditional_dict,
                timestep=critic_timestep,
                context_model=self.context_fake,
                mode="base",
            )

            # Step 3: Compute the denoising loss for the fake critic
            if self.args.denoising_loss_type == "flow":
                from utils.wan_wrapper import WanDiffusionWrapper
                # flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                #     scheduler=self.scheduler,
                #     x0_pred=pred_fake_image.flatten(0, 1),
                #     xt=noisy_generated_image.flatten(0, 1),
                #     timestep=critic_timestep.flatten(0, 1)
                # )
                flow_pred = pred_fake_flow
                pred_fake_noise = None
            else:
                flow_pred = None
                # pred_fake_noise = self.scheduler.convert_x0_to_noise(
                #     x0=pred_fake_image.flatten(0, 1),
                #     xt=noisy_generated_image.flatten(0, 1),
                #     timestep=critic_timestep.flatten(0, 1)
                # ).unflatten(0, image_or_video_shape[:2])

            
            # print("save flow pred, shape:", flow_pred.shape)
            # save_latent_as_video(
            #     latent_tensor=flow_pred.unsqueeze(0),
            #     vae=self.vae,
            #     base_output_path=self.output_path,
            #     step=0,
            #     video_name="flow_pred",
            #     device_id=self.device
            # )

            # print("save critic_noise - generated_image, shape:", (critic_noise - generated_image).shape)
            # save_latent_as_video(
            #     latent_tensor=(critic_noise - generated_image),
            #     vae=self.vae,
            #     base_output_path=self.output_path,
            #     step=0,
            #     video_name="generated_image",
            #     device_id=self.device
            # )

            # print("pred_fake_image, shape:", pred_fake_image.shape)
            # print("timestep:", critic_timestep)
            if step % self.vis_per_step == 0:
                with torch.no_grad():
                    save_latent_as_video(
                        latent_tensor=noisy_generated_image,
                        vae=self.vae,
                        base_output_path=self.output_path,
                        step=step,
                        video_name="critic_noisy_generated_image",
                        device_id=self.device
                    )

                    save_latent_as_video(
                        latent_tensor=pred_fake_image,
                        vae=self.vae,
                        base_output_path=self.output_path,
                        step=step,
                        video_name="critic_pred_fake_image",
                        device_id=self.device
                    )
            
            if self._context_teacher:
                if self.context_fake:
                    print("use context loss.")
                    tgt_latent_len = image_or_video_shape[1] // 2
                    training_target = self.scheduler.training_target(generated_image[:, tgt_latent_len:, ...], critic_noise[:, tgt_latent_len:, ...], critic_timestep[:, tgt_latent_len:, ...])
                    denoising_loss = torch.nn.functional.mse_loss(pred_fake_flow[:, tgt_latent_len:, ...].float(), training_target.float())
                    # Add wight; 
                    denoising_loss = denoising_loss * self.scheduler.training_weight(critic_timestep).mean()
                else:
                    print("use default loss.")
                    training_target = self.scheduler.training_target(generated_image, critic_noise, critic_timestep)
                    denoising_loss = torch.nn.functional.mse_loss(pred_fake_flow.float(), training_target.float()).mean()
                print("denoising_loss", denoising_loss, "timestep", critic_timestep)
                # denoising_loss = self.denoising_loss_func(
                #     x=generated_image[:, image_or_video_shape[1]//2:, ...].flatten(0, 1),
                #     x_pred=pred_fake_image.flatten(0, 1),
                #     noise=critic_noise[:, image_or_video_shape[1]//2:, ...].flatten(0, 1),
                #     noise_pred=pred_fake_noise,
                #     alphas_cumprod=self.scheduler.alphas_cumprod,
                #     timestep=critic_timestep.flatten(0, 1),
                #     flow_pred=flow_pred[:, image_or_video_shape[1]//2:, ...],
                #     scheduler=self.scheduler,
                # ) 
            else:
                denoising_loss = self.denoising_loss_func(
                    x=generated_image.flatten(0, 1),
                    x_pred=pred_fake_image.flatten(0, 1),
                    noise=critic_noise.flatten(0, 1),
                    noise_pred=pred_fake_noise,
                    alphas_cumprod=self.scheduler.alphas_cumprod,
                    timestep=critic_timestep.flatten(0, 1),
                    flow_pred=flow_pred,
                    scheduler=self.scheduler,
                )

            # Step 5: Debugging Log
            critic_log_dict = {
                "critic_timestep": critic_timestep.detach()
            }

            return denoising_loss, critic_log_dict