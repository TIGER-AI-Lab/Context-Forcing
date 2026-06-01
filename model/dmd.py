from pipeline import SelfForcingTrainingPipeline
import torch.nn.functional as F
from typing import Optional, Tuple
import torch
import os
import gc
from einops import rearrange
from torchvision.io import write_video

from model.base import SelfForcingModel


def save_latent_as_video(latent_tensor, vae, base_output_path, step, video_name, device_id, fps=16):
    """Decode a latent tensor with the VAE and save as MP4."""
    with torch.no_grad():
        video_tensor = vae.decode_to_pixel(latent_tensor.to(torch.bfloat16))
        video_tensor = (video_tensor * 0.5 + 0.5).clamp(0, 1)

    if hasattr(vae, 'model') and hasattr(vae.model, 'clear_cache') and callable(vae.model.clear_cache):
        vae.model.clear_cache()

    video_tensor_hwc = rearrange(video_tensor, 'b t c h w -> b t h w c').cpu()
    video_to_save = (255.0 * video_tensor_hwc[0]).clamp(0, 255).to(torch.uint8)

    output_dir = os.path.join(base_output_path, "saved_video")
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"step_{step}_{video_name}_{device_id}.mp4")
    write_video(file_path, video_to_save, fps=fps)


class DMD(SelfForcingModel):
    def __init__(self, args, device):
        """
        Initialize the DMD (Distribution Matching Distillation) module.
        This class is self-contained and computes generator and fake score losses
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

        # initialized later with the FSDP-wrapped pipeline
        self.inference_pipeline: SelfForcingTrainingPipeline = None

        # DMD hyperparameters
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
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

    def _clear_cache(self):
        self.kv_cache_dict = None
        gc.collect()
        torch.cuda.empty_cache()

    def optimized_scale(self, positive_flat, negative_flat):
        """CFG-zero* optimized scale (https://arxiv.org/abs/2503.18886)."""
        dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
        squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8
        return dot_product / squared_norm

    def _convert_flow_pred_to_x0(
        self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert flow matching prediction to x0.
        x_t = (1-sigma_t)*x0 + sigma_t*noise  =>  x0 = x_t - sigma_t * flow_pred
        """
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device),
            [flow_pred, xt, self.scheduler.sigmas, self.scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    def _compute_kl_grad(
        self,
        noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        normalization: bool = True,
        context_video: torch.Tensor = None,
        step: int = 0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the DMD KL gradient (eq 7 in https://arxiv.org/abs/2311.18828).

        Context Forcing: when context_teacher is active, both real and fake scores
        receive the full generation history prepended as context, eliminating the
        student-teacher supervision mismatch.
        """
        # Doubled timestep for the concatenated [context | target] sequence
        _timestep = torch.cat([timestep, timestep], dim=1)

        # --- Fake score forward ---
        ctx_len = context_video.shape[1] if context_video is not None else None
        if self._context_teacher and self.context_fake and context_video is not None:
            _, pred_fake_image_cond = self.fake_score(
                noisy_image_or_video=torch.cat([context_video, noisy_image_or_video], dim=1),
                conditional_dict={'prompt_embeds': conditional_dict['prompt_embeds']},
                timestep=_timestep,
                context_model=True,
                context_length=ctx_len,
                mode="base",
            )
            if self.fake_guidance_scale != 0.0:
                _, pred_fake_image_uncond = self.fake_score(
                    noisy_image_or_video=torch.cat([context_video, noisy_image_or_video], dim=1),
                    conditional_dict=unconditional_dict,
                    timestep=_timestep,
                    context_model=True,
                    context_length=ctx_len,
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

        # --- Real score (long-context teacher) forward ---
        if self._context_teacher and context_video is not None:
            _, pred_real_image_cond = self.real_score(
                noisy_image_or_video=torch.cat([context_video, noisy_image_or_video], dim=1),
                conditional_dict={'prompt_embeds': conditional_dict['prompt_embeds']},
                timestep=_timestep,
                context_model=True,
                context_length=ctx_len,
                mode="base",
            )
            _, pred_real_image_uncond = self.real_score(
                noisy_image_or_video=torch.cat([context_video, noisy_image_or_video], dim=1),
                conditional_dict=unconditional_dict,
                timestep=_timestep,
                context_model=True,
                context_length=ctx_len,
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

        # --- Periodic training visualization ---
        if step % self.vis_per_step == 0:
            save_latent_as_video(
                latent_tensor=torch.cat([context_video, estimated_clean_image_or_video], dim=1) if self._context_teacher else estimated_clean_image_or_video,
                vae=self.vae, base_output_path=self.output_path,
                step=step, video_name="student_generated_video", device_id=self.device
            )
            save_latent_as_video(
                latent_tensor=torch.cat([context_video, noisy_image_or_video], dim=1) if self._context_teacher else noisy_image_or_video,
                vae=self.vae, base_output_path=self.output_path,
                step=step, video_name="pred_real_image_input", device_id=self.device
            )
            save_latent_as_video(
                latent_tensor=torch.cat([context_video, pred_real_image], dim=1).to(dtype=self.dtype) if self._context_teacher else pred_real_image,
                vae=self.vae, base_output_path=self.output_path,
                step=step, video_name="pred_real_image", device_id=self.device
            )
            save_latent_as_video(
                latent_tensor=pred_fake_image,
                vae=self.vae, base_output_path=self.output_path,
                step=step, video_name="pred_fake_image", device_id=self.device
            )
            save_latent_as_video(
                latent_tensor=(pred_fake_image - pred_real_image),
                vae=self.vae, base_output_path=self.output_path,
                step=step, video_name="grad", device_id=self.device
            )

        # --- DMD gradient (eq. 7) ---
        if self._context_teacher and context_video is not None:
            if self.context_fake:
                grad = (pred_fake_image[:, noisy_image_or_video.shape[1]:] - pred_real_image[:, noisy_image_or_video.shape[1]:])
            else:
                grad = (pred_fake_image - pred_real_image[:, noisy_image_or_video.shape[1]:])
        else:
            grad = (pred_fake_image - pred_real_image)

        # --- Gradient normalization (eq. 8) ---
        if normalization:
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
        Compute the contextual DMD loss (eq 7 in https://arxiv.org/abs/2311.18828).

        When context_teacher is active, image_or_video contains [context | target] frames.
        The loss is computed only on the target portion; context frames provide the
        long-range supervision signal to the teacher.
        """
        batch_size, num_frame = image_or_video.shape[:2]

        if self._context_teacher:
            context_video = image_or_video[:, :self.context_window, ...]
            original_latent = image_or_video[:, self.context_window:, ...]
            num_frame = original_latent.shape[1]
        else:
            original_latent = image_or_video
            context_video = None

        with torch.no_grad():
            min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
            max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
            timestep = self._get_timestep(
                min_timestep, max_timestep,
                batch_size, num_frame,
                self.num_frame_per_block,
                uniform_timestep=True
            )

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

            grad, dmd_log_dict = self._compute_kl_grad(
                noisy_image_or_video=noisy_latent,
                estimated_clean_image_or_video=original_latent,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                context_video=context_video,
                step=step,
            )

        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(
                original_latent.double()[gradient_mask],
                (original_latent.double() - grad.double()).detach()[gradient_mask],
                reduction="mean"
            )
        else:
            dmd_loss = 0.5 * F.mse_loss(
                original_latent.double(),
                (original_latent.double() - grad.double()).detach(),
                reduction="mean"
            )
        return dmd_loss, dmd_log_dict

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        initial_latent: torch.Tensor = None,
        step: int = 0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Unroll the generator from noise and compute the contextual DMD loss.
        Uses backward simulation (no real data needed); see Sec 4.5 of DMD2
        (https://arxiv.org/abs/2405.14867).
        """
        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to, _ = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent,
            step=step
        )

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
        initial_latent: torch.Tensor = None,
        step: int = 0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate samples from the current generator and train the fake score on them.
        Uses backward simulation (no real data needed); see Sec 4.5 of DMD2
        (https://arxiv.org/abs/2405.14867).
        """
        # Step 1: Generate samples (no grad — critic only updates fake score)
        with torch.no_grad():
            generated_image, _, denoised_timestep_from, denoised_timestep_to, _ = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                initial_latent=initial_latent,
                step=step,
            )

        if self._context_teacher and self.context_fake:
            image_or_video_shape[1] *= 2

        if self._context_teacher and not self.context_fake:
            context_video = generated_image[:, :self.context_window, ...]
            generated_image = generated_image[:, self.context_window:, ...]
        else:
            context_video = None

        # Step 2: Sample critic timestep and add noise
        min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
        max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
        critic_timestep = self._get_timestep(
            min_timestep, max_timestep,
            image_or_video_shape[0], image_or_video_shape[1],
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

        if self._context_teacher and self.context_fake:
            # Keep context frames clean (zero-noise timestep)
            noisy_generated_image[:, :image_or_video_shape[1] // 2, ...] = generated_image[:, :image_or_video_shape[1] // 2, ...]

        # Step 3: Fake score forward
        # When context_fake=True the sequence is [context(N) | target(N)], so context_length = N.
        critic_context_length = image_or_video_shape[1] // 2 if self.context_fake else None
        pred_fake_flow, pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_generated_image,
            conditional_dict=conditional_dict,
            timestep=critic_timestep,
            context_model=self.context_fake,
            context_length=critic_context_length,
            mode="base",
        )

        # Step 4: Periodic visualization
        if step % self.vis_per_step == 0:
            with torch.no_grad():
                save_latent_as_video(
                    latent_tensor=noisy_generated_image, vae=self.vae,
                    base_output_path=self.output_path, step=step,
                    video_name="critic_noisy_generated_image", device_id=self.device
                )
                save_latent_as_video(
                    latent_tensor=pred_fake_image, vae=self.vae,
                    base_output_path=self.output_path, step=step,
                    video_name="critic_pred_fake_image", device_id=self.device
                )

        # Step 5: Denoising loss for the fake score
        if self._context_teacher:
            if self.context_fake:
                tgt_latent_len = image_or_video_shape[1] // 2
                training_target = self.scheduler.training_target(
                    generated_image[:, tgt_latent_len:, ...],
                    critic_noise[:, tgt_latent_len:, ...],
                    critic_timestep[:, tgt_latent_len:, ...]
                )
                denoising_loss = torch.nn.functional.mse_loss(
                    pred_fake_flow[:, tgt_latent_len:, ...].float(),
                    training_target.float()
                )
                denoising_loss = denoising_loss * self.scheduler.training_weight(critic_timestep).mean()
            else:
                training_target = self.scheduler.training_target(generated_image, critic_noise, critic_timestep)
                denoising_loss = torch.nn.functional.mse_loss(
                    pred_fake_flow.float(), training_target.float()
                ).mean()
        else:
            flow_pred = pred_fake_flow if self.args.denoising_loss_type == "flow" else None
            denoising_loss = self.denoising_loss_func(
                x=generated_image.flatten(0, 1),
                x_pred=pred_fake_image.flatten(0, 1),
                noise=critic_noise.flatten(0, 1),
                noise_pred=None,
                alphas_cumprod=self.scheduler.alphas_cumprod,
                timestep=critic_timestep.flatten(0, 1),
                flow_pred=flow_pred,
                scheduler=self.scheduler,
            )

        critic_log_dict = {
            "critic_timestep": critic_timestep.detach()
        }

        return denoising_loss, critic_log_dict
