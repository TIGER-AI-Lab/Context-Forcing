import gc
import logging

from utils.dataset import ShardingLMDBDataset, cycle
from utils.dataset import TextDataset
from utils.dataset import UltraVidDataset
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.cuda.amp import GradScaler, autocast
from utils.misc import (
    set_seed,
    merge_dict_list
)
import torch.distributed as dist
from omegaconf import OmegaConf
from model import CausVid, DMD, SiD
import torch
import bitsandbytes.optim as bnb_optim
import wandb
from tqdm import tqdm 
import time
import os
import math
import random
from torch.optim.lr_scheduler import LambdaLR


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0
        self.max_step = config.max_step

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb
        self.use_amp = self.config.training.get('use_amp', False)
        self.scaler = GradScaler(enabled=self.use_amp)

        if dist.get_rank() == 0:
            print(f"Automatic Mixed Precision (AMP) enabled: {self.use_amp}")

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        print(f"set seed to {config.seed + global_rank}")
        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer
        if config.distribution_loss == "causvid":
            self.model = CausVid(config, device=self.device)
        elif config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        elif config.distribution_loss == "sid":
            self.model = SiD(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")

        # Save pretrained model state_dicts to CPU
        self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "generator_score_cpu_offload", False)
        )

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "real_score_cpu_offload", False)
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "fake_score_cpu_offload", False)
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        # Previous generator_optimizer (for reference)
        self.generator_optimizer = torch.optim.AdamW(
                    [param for param in self.model.generator.parameters()
                     if param.requires_grad],
                    lr=config.lr,
                    betas=(config.beta1, config.beta2),
                    weight_decay=config.weight_decay
                )

        # NEW 8-bit generator_optimizer
        # self.generator_optimizer = bnb_optim.AdamW8bit(
        #     [param for param in self.model.generator.parameters()
        #     if param.requires_grad],
        #     lr=config.lr,
        #     betas=(config.beta1, config.beta2),
        #     weight_decay=config.weight_decay
        # )

        # Previous critic_optimizer (for reference)
        self.critic_optimizer = torch.optim.AdamW(
                    [param for param in self.model.fake_score.parameters()
                     if param.requires_grad],
                    lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
                    betas=(config.beta1_critic, config.beta2_critic),
                    weight_decay=config.weight_decay
                )

        # NEW 8-bit critic_optimizer
        # self.critic_optimizer = bnb_optim.AdamW8bit(
        #     [param for param in self.model.fake_score.parameters()
        #     if param.requires_grad],
        #     lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
        #     betas=(config.beta1_critic, config.beta2_critic),
        #     weight_decay=config.weight_decay
        # )

        # Step 3: Initialize the dataloader
        if self.config.long:
            dataset = UltraVidDataset(
                        csv_path=config.long_csv_file_path,
                        video_folder=config.long_videos_directory,
                        prompt_column='Summarized Description', # or 'Brief Description', 'Detailed Description', etc.
                        num_frames=config.num_frames
                    )
        elif self.config.i2v:
            dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8))
        else:
            dataset = TextDataset(config.data_path, seed=config.seed)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=0)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        # self.dataloader = cycle(dataloader)
        self.sampler = sampler
        self.dataloader = dataloader
        self.dataloader_iterator = None

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        global_batch_size = config.batch_size * world_size
        self.steps_per_epoch = len(dataset) // global_batch_size
        self.steps_per_epoch = max(1, self.steps_per_epoch)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            elif "generator_ema" in state_dict:
                state_dict = state_dict["generator_ema"]
            self.model.generator.load_state_dict(
                state_dict, strict=True
            )

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.context_mix_step = getattr(config, "context_mix_step", 10)
        self.previous_time = None
        self.last_eff_g = 1.0
        self.last_theta_g = 1.0

        self._ema_R = None          
        self._ema_beta = 0.9        
        self.R_rel_target = 1.5     
        self.R_scale_floor = 0.5    
        self.R_scale_ceiling = 1.0 
        self.R_rel_margin = 0.0 

    def _p_ctx_schedule(self, step: int) -> float:
        stage2_start = int(self.config.get("stage2_start_step", 150)) 
        stage2_full  = int(self.config.get("stage2_full_step", 260)) 

        p_min = float(self.config.get("p_ctx_min", 0.3))  
        p_max = float(self.config.get("p_ctx_max", 0.7))  

        if step < stage2_start:
            return 0.0 
        if step >= stage2_full:
            return p_max 

        alpha = (step - stage2_start) / max(1, (stage2_full - stage2_start))
        return p_min + (p_max - p_min) * alpha

    def save(self):
        """
        Saves the current state of the trainer to a checkpoint file,
        compatible with FSDP. This includes model weights, optimizer states,
        EMA state, and the current step for complete resumption.
        """
        # These gathering operations are collective and MUST be called on all ranks.
        print("Gathering FSDP model states for checkpoint...")
        logging.info("Gathering FSDP model states for checkpoint...")
        generator_state_dict = fsdp_state_dict(self.model.generator)
        generator_optimizer_state_dict= None
        if self.config.save_optimizer:
            generator_optimizer_state_dict = FSDP.optim_state_dict(self.model.generator, self.generator_optimizer)
    
        critic_state_dict = None
        if not self.config.teacher_forcing:
            critic_state_dict = fsdp_state_dict(self.model.fake_score)

        # Optimizer and EMA states are not FSDP-managed in the same way,
        # but we still gather them on all ranks before saving on main.
        # generator_optimizer_state_dict = self.generator_optimizer.state_dict()
        
        critic_optimizer_state_dict = None
        if not self.config.teacher_forcing and self.config.save_optimizer:
            critic_optimizer_state_dict = FSDP.optim_state_dict(self.model.fake_score, self.critic_optimizer)

        ema_state_dict = None
        if self.config.ema_start_step < self.step and self.generator_ema is not None:
            ema_state_dict = self.generator_ema.state_dict()
        
        # Create the comprehensive checkpoint dictionary for resuming.
        checkpoint = {
            'step': self.step,
            'generator': generator_state_dict,
            'critic': critic_state_dict,
            'generator_optimizer': generator_optimizer_state_dict,
            'critic_optimizer': critic_optimizer_state_dict,
            'generator_ema': ema_state_dict,
        }
        print("Creat ckpt done.")

        # Only the main process handles writing the file to disk.
        if self.is_main_process:
            checkpoint_dir = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            model_path = os.path.join(checkpoint_dir, "model.pt")

            # Save the single, comprehensive checkpoint file.
            print(f"Saving checkpoint to {model_path}")
            torch.save(checkpoint, model_path)
            print(f"Saved complete checkpoint to {model_path}")
            logging.info(f"Saved complete checkpoint to {model_path}")

    @torch.no_grad()
    def l2_param_norm(self, params) -> float:
        """Return ||θ||_2 over trainable params."""
        s = 0.0
        for p in params:
            if p.requires_grad:
                s += (p.detach().float() ** 2).sum().item()
        return math.sqrt(s)

    @torch.no_grad()
    def l2_grad_norm(self, params) -> float:
        """Return ||g||_2 over params that have grads."""
        s = 0.0
        for p in params:
            if p.requires_grad and p.grad is not None:
                s += (p.grad.detach().float() ** 2).sum().item()
        return math.sqrt(s)
    
    def _cosine_with_warmup(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_scale: float = 0.2,
        warmup_start_scale: float = 0.3,
        last_epoch: int = -1, 
    ):
        def lr_lambda(step: int):
            if step < warmup_steps:
                w = max(1, warmup_steps)
                return warmup_start_scale + (1.0 - warmup_start_scale) * (step / w)
            prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return min_lr_scale + 0.5 * (1.0 - min_lr_scale) * (1 + math.cos(math.pi * prog))

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        if last_epoch is not None and last_epoch >= 0:
            scheduler.last_epoch = last_epoch

        return scheduler

    def _constant_lr(self, optimizer):
        return LambdaLR(optimizer, lr_lambda=lambda _: 1.0)


    # -------------------- Strength ratios (size-agnostic main: R_rel) --------------------

    @torch.no_grad()
    def compute_strength_ratios(
        self,
        G_params,
        D_params,
        lr_g: float,
        lr_d: float,
        clip_g: float = None,
        clip_d: float = None,
        lora_scale: float = 1.0,   # e.g., alpha/r for LoRA (256/128 = 2.0)
        freq_g: int = 1,
        freq_d: int = 1,
        theta_cache: dict | None = None,
        refresh_theta: bool = False,
    ):
        """
        Compute:
        - R_rel: size-agnostic relative-step ratio (primary indicator)
        - R_raw: quick indicator without size normalization
        Also returns raw/eff grad-norms and ||θ|| for logging.

        Call after backward() and before optimizer.step().
        """
        raw_g = self.l2_grad_norm(G_params)
        raw_d = self.l2_grad_norm(D_params)

        eff_g = raw_g if (clip_g is None or clip_g <= 0) else min(raw_g, clip_g)
        eff_d = raw_d if (clip_d is None or clip_d <= 0) else min(raw_d, clip_d)

        if theta_cache is None or refresh_theta or ("theta_g" not in theta_cache):
            theta_g = self.l2_param_norm(G_params)
            theta_d = self.l2_param_norm(D_params)
            theta_cache = {"theta_g": theta_g, "theta_d": theta_d}
        else:
            theta_g, theta_d = theta_cache["theta_g"], theta_cache["theta_d"]

        # Relative step magnitudes (size-agnostic)
        rel_g = (eff_g * lr_g * freq_g) / max(1e-12, theta_g)
        rel_d = (eff_d * lr_d * freq_d * lora_scale) / max(1e-12, theta_d)
        R_rel = rel_d / max(1e-12, rel_g)

        # Quick pointer (no size normalization)
        R_raw = (eff_d * lr_d * freq_d * lora_scale) / max(1e-12, eff_g * lr_g * freq_g)

        return {
            "R_rel": R_rel,
            "R_raw": R_raw,
            "raw_g": raw_g,
            "raw_d": raw_d,
            "eff_g": eff_g,
            "eff_d": eff_d,
            "theta_g": theta_g,
            "theta_d": theta_d,
            "theta_cache": theta_cache,
        }

    def fwdbwd_one_step(self, batch, train_generator, accum_steps=1, is_last_micro=True):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        # if self.step % 20 == 0:
        #     torch.cuda.empty_cache()

        # Step 1: prompts
        if self.config.camera_prompt:
            text_prompts = [self.config.camera_prompt + p for p in batch["prompts"]]
            print(text_prompts)
        else:
            text_prompts = batch["prompts"]
        if self.config.i2v:
            clean_latent = None
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            clean_latent = None
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: cond
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts)
            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach() for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        R_eval_interval = int(self.config.get("R_eval_interval", 1))

        # Step 3: G
        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None,
                step=self.step,
            )

            if self.use_amp:
                if accum_steps > 1:
                    generator_loss = generator_loss / accum_steps
                    
                self.scaler.scale(generator_loss).backward()
            else:
                if accum_steps > 1:
                    generator_loss = generator_loss / accum_steps
                # torch.cuda.empty_cache()
                generator_loss.backward()
            gen_loss_cpu = generator_loss.detach().item()

            generator_grad_norm = float("nan")
            raw_g = eff_g = theta_g = float("nan")

            if is_last_micro:
                generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm_generator)
                if self.use_amp:
                    self.scaler.unscale_(self.generator_optimizer)
                
                if R_eval_interval > 0 and self.step % R_eval_interval == 0:
                    try:
                        G_params = [p for p in self.model.generator.parameters() if p.requires_grad]
                        raw_g = self.l2_grad_norm(G_params)
                        eff_g = min(raw_g, self.max_grad_norm_generator)
                        theta_g = self.l2_param_norm(G_params)
                    except Exception:
                        raw_g = eff_g = theta_g = float("nan")

                    self.last_raw_g = raw_g if (isinstance(raw_g, (int, float)) and math.isfinite(raw_g)) else 0.0
                    self.last_eff_g = eff_g
                    self.last_theta_g = theta_g

            device = self.device
            generator_log_dict.update({
                "generator_loss": torch.tensor(gen_loss_cpu, device='cpu'), 
                "generator_grad_norm": torch.tensor(generator_grad_norm, device='cpu') 
                                    if not torch.is_tensor(generator_grad_norm) else generator_grad_norm.cpu(),
                "raw_g":  torch.tensor(raw_g, device='cpu'), 
                "eff_g":  torch.tensor(eff_g, device='cpu'), 
                "theta_g": torch.tensor(theta_g, device='cpu'), 
            })
            del generator_loss

            if self.step % 2 == 0 and self.is_main_process:
                print("step:", self.step, "accum_steps:", accum_steps, generator_log_dict)
            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: D
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None,
            step=self.step,
        )
        
        if self.use_amp:
            if accum_steps > 1:
                critic_loss = critic_loss / accum_steps
            
            self.scaler.scale(critic_loss).backward()
        else:
            if accum_steps > 1:
                critic_loss = critic_loss / accum_steps
            critic_loss.backward()

        cri_loss_cpu = critic_loss.detach().item()

        critic_grad_norm = float("nan")
        raw_d = eff_d = theta_d = R_raw = R_rel = float("nan")

        if is_last_micro:
            critic_grad_norm = self.model.fake_score.clip_grad_norm_(self.max_grad_norm_critic)
            if self.use_amp:
                self.scaler.unscale_(self.critic_optimizer)

            if R_eval_interval > 0 and self.step % R_eval_interval == 0:
                with torch.no_grad():
                    try:
                        D_params = [p for p in self.model.fake_score.parameters() if p.requires_grad]
                        raw_d = self.l2_grad_norm(D_params)
                        eff_d = min(raw_d, self.max_grad_norm_critic)
                        theta_d = self.l2_param_norm(D_params)

                        lr_g = self.generator_optimizer.param_groups[0]["lr"]
                        lr_d = self.critic_optimizer.param_groups[0]["lr"]
                        fake_lora_cfg = self.config.get("fake_lora", {})
                        rank = float(fake_lora_cfg.get("rank", 1.0))
                        alpha = float(fake_lora_cfg.get("lora_alpha", rank))
                        lora_scale = alpha / max(rank, 1e-6)

                        ratio = float(self.config.get("dfake_gen_update_ratio", 1.0))
                        if ratio >= 1.0:
                            freq_g = 1.0
                            freq_d = min(ratio, 8.0)
                        else:
                            freq_g = min(1.0 / max(ratio, 1e-6), 8.0)
                            freq_d = 1.0

                        R_raw = (eff_d * lr_d * freq_d * lora_scale) / max(
                            1e-12,
                            (self.last_eff_g if hasattr(self, "last_eff_g") else 1.0) * lr_g * freq_g,
                        )
                        theta_g = getattr(self, "last_theta_g", None)
                        if theta_g is None or not math.isfinite(theta_g):
                            R_rel = float("nan")
                        else:
                            rel_g = ((self.last_eff_g if hasattr(self, "last_eff_g") else 0.0)
                                    * lr_g * freq_g) / max(1e-12, theta_g)
                            rel_d = (eff_d * lr_d * freq_d * lora_scale) / max(1e-12, theta_d)
                            R_rel = rel_d / max(1e-12, rel_g)

                        # ==== Debug ====
                        print("---- R computation debug ----")
                        print(f"lr_g={lr_g:.3e}, lr_d={lr_d:.3e}, lora_scale={lora_scale}")
                        print(f"freq_g={freq_g}, freq_d={freq_d}")
                        print(f"raw_g={getattr(self, 'last_raw_g', float('nan')):.6f}, "
                            f"eff_g={getattr(self, 'last_eff_g', float('nan')):.6f}, "
                            f"theta_g={getattr(self, 'last_theta_g', float('nan')):.6f}")
                        print(f"raw_d={raw_d:.6f}, eff_d={eff_d:.6f}, theta_d={theta_d:.6f}")
                        print(f"R_raw={R_raw:.3f}, R_rel={R_rel:.3f}")
                        print("-----------------------------")
                        # =============================

                    except Exception as e:
                        print("R debug exception:", e)
                        raw_d = eff_d = theta_d = R_raw = R_rel = float("nan")

                    # === Gradient scaling controller for D (fake_score) ===
                    try:
                        warm = int(self.config.get("warmup_steps", 0))
                        if self.step >= warm:
                            cur_R = float(R_rel) if math.isfinite(float(R_rel)) else float("nan")

                            if math.isfinite(cur_R):
                                if self._ema_R is None or not math.isfinite(self._ema_R):
                                    self._ema_R = cur_R
                                else:
                                    self._ema_R = self._ema_beta * self._ema_R + (1.0 - self._ema_beta) * cur_R

                                trigger = self._ema_R > (self.R_rel_target * (1.0 + self.R_rel_margin))

                                if trigger:
                                    scale = max(self.R_scale_floor,
                                                min(self.R_scale_ceiling,
                                                    self.R_rel_target / max(1e-12, self._ema_R)))
                                    for p in self.model.fake_score.parameters():
                                        if p.requires_grad and (p.grad is not None):
                                            p.grad.mul_(scale)

                                    critic_log_dict["critic_grad_scale"] = float(scale)

                                critic_log_dict["R_rel_ema"] = float(self._ema_R)
                    except Exception as _e:
                        pass
                    # === end controller ===

        device = self.device
        critic_log_dict.update({
            "critic_loss": torch.tensor(cri_loss_cpu, device='cpu'), 
            "critic_grad_norm": torch.tensor(critic_grad_norm, device='cpu')
                                if not torch.is_tensor(critic_grad_norm) else critic_grad_norm.cpu(),
            "raw_d":  torch.tensor(raw_d, device='cpu'),
            "eff_d":  torch.tensor(eff_d, device='cpu'),
            "theta_d": torch.tensor(theta_d, device='cpu'),
            "R_raw":  torch.tensor(R_raw, device='cpu'),
            "R_rel":  torch.tensor(R_rel, device='cpu'),
        })
        del critic_loss

        if self.step % 2 == 0 and self.is_main_process:
            print("step:", self.step, "accum_steps:", accum_steps, critic_log_dict)

        if "raw_g" in generator_log_dict:
            self.last_eff_g = generator_log_dict["eff_g"].item()
            self.last_theta_g = generator_log_dict["theta_g"].item()

        return critic_log_dict

    def fwdbwd_one_step_long(self, batch, train_generator):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        video_tensor = batch["video_tensor"] # for vae latent
        with torch.no_grad():
            self.model.vae.eval()
            video_latent = self.model.vae.encode_to_latent(video_tensor.to(device=self.device, dtype=torch.bfloat16))
        
        batch_size = video_tensor.shape[0]
        del video_tensor
        torch.cuda.empty_cache()
        # video_latent = self.model.vae.encode_to_latent(video_tensor.to(device=self.device, dtype=torch.bfloat16))
        print(video_latent.shape)

        midpoint_frame = video_latent.shape[1] // 2
        context_latent = video_latent[:, :midpoint_frame]
        target_latent = video_latent[:, midpoint_frame:]

        clean_latent = None
        image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator or self.config.teacher_forcing:
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                context_latent=context_latent,
                target_latent=target_latent,
                initial_latent=image_latent if self.config.i2v else None,
                w_context=True if self.config.long else False,
                teacher_forcing=True if self.config.teacher_forcing else False,
            )

            generator_loss.backward()
            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm_generator)

            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": generator_grad_norm})

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        critic_loss, critic_log_dict = self.model.critic_loss_long(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            context_latent=context_latent,
            initial_latent=image_latent if self.config.i2v else None,
            w_context=True if self.config.long else False,
        )
        print("loss:", critic_loss)
        critic_loss.backward()
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(
            self.max_grad_norm_critic)

        critic_log_dict.update({"critic_loss": critic_loss,
                                "critic_grad_norm": critic_grad_norm})

        return critic_log_dict

    def generate_video(self, pipeline, prompts, image=None):
        batch_size = len(prompts)
        if image is not None:
            image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device="cuda", dtype=torch.bfloat16)

            # Encode the input image as the first latent
            initial_latent = pipeline.vae.encode_to_latent(image).to(device="cuda", dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames - 1, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )
        else:
            initial_latent = None
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )

        video, _ = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent
        )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        return current_video

    def load(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            logging.warning(f"Checkpoint file not found at {checkpoint_path}. Starting from scratch.")
            return 0

        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        self.step = checkpoint.get('step', 0)

        self.model.generator.load_state_dict(checkpoint['generator'])
        if (not self.config.teacher_forcing) and checkpoint.get('critic', None) is not None:
            self.model.fake_score.load_state_dict(checkpoint['critic'])
            print(f"Resuming training critic: {checkpoint_path}")

        ema_state = checkpoint.get('generator_ema', None)
        if ema_state is not None:
            if self.generator_ema is None:
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)
            self.generator_ema.load_state_dict(ema_state)

        # ---- Generator optimizer ----
        
        gen_osd = checkpoint.get('generator_optimizer', None)
        if self.config.save_optimizer and gen_osd is not None:
            try:
                sharded_osd = FSDP.optim_state_dict_to_load(
                    optim_state_dict=gen_osd,
                    model=self.model.generator,
                    optim=self.generator_optimizer,
                )
                self.generator_optimizer.load_state_dict(sharded_osd)
                print("Generator optimizer state loaded.")
            except Exception as e:
                print(f"Failed to load generator optimizer state, using fresh optimizer. Error: {e}")

        # ---- Critic optimizer ----
        crit_osd = checkpoint.get('critic_optimizer', None)
        if (not self.config.teacher_forcing) and self.config.save_optimizer and crit_osd is not None:
            try:
                sharded_osd = FSDP.optim_state_dict_to_load(
                    optim_state_dict=crit_osd,
                    model=self.model.fake_score,
                    optim=self.critic_optimizer,
                )
                self.critic_optimizer.load_state_dict(sharded_osd)
                print("Critic optimizer state loaded.")
            except Exception as e:
                print(f"Failed to load critic optimizer state, using fresh optimizer. Error: {e}")
        
        return self.step

    def train(self):
        start_step = self.step

        resume_path = self.config.get('resume_checkpoint_path')
        
        if resume_path:
            try:
                logging.info(f"Resuming training from checkpoint: {resume_path}")
                print(f"Resuming training from checkpoint: {resume_path}")
                start_step = self.load(resume_path)
                print("start step", start_step)
            except Exception as e:
                logging.error(f"Failed to load checkpoint '{resume_path}'. Starting from scratch. Error: {e}")
                start_step = 0
        else:
            logging.info("No resume checkpoint provided. Starting training from scratch.")

        if self.is_main_process:
            progress_bar = tqdm(initial=start_step, total=self.max_step, desc="Training Steps")

        fake_pretrain_steps = int(self.config.get("fake_pretrain_steps", 50))
        base_lr_d = self.critic_optimizer.param_groups[0]["lr"]

        start_step = self.step 
        fake_pretrain_steps = int(self.config.get("fake_pretrain_steps", 50))
        warmup_steps_cfg = int(self.config.get("warmup_steps", 0))
        warmup_steps = max(warmup_steps_cfg, fake_pretrain_steps)
        total_steps = self.max_step

        self.schG = self._cosine_with_warmup(
            self.generator_optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_scale=0.2,
            warmup_start_scale=0.3,
            last_epoch=start_step - 1, 
        )
        self.schD = None

        while True:
            if self.step >= self.max_step:
                print(f"Training finished after reaching max_step: {self.max_step}")
                break

            if self.dataloader_iterator is None or self.step % self.steps_per_epoch == 0:
                current_epoch = self.step // self.steps_per_epoch
                print(f"Step {self.step}: Reshuffling data and creating new iterator for epoch {current_epoch}...")
                self.sampler.set_epoch(current_epoch)
                self.dataloader_iterator = iter(self.dataloader)

            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0
            accum_steps = self.config.get('gradient_accumulate', 1)

            fake_only_phase = (self.step < fake_pretrain_steps)

            if self.step < 0:
                TRAIN_GENERATOR = False

            # context_switch_step = self.config.get('context_switch_step', 0)
            # if context_switch_step > 0:
            #     chunk_index = self.step // context_switch_step
            #     if chunk_index % 2 == 1:
            #         self.model._context_teacher = False
            #     else:
            #         self.model._context_teacher = True
            #     print(f"Current step:{self.step}, context teacher:{self.model._context_teacher}")

            p_ctx = self._p_ctx_schedule(self.step)

            use_context = None
            if dist.is_available() and dist.is_initialized():
                flag = torch.zeros(1, dtype=torch.int64, device=self.device)

                if self.is_main_process:
                    flag[0] = 1 if random.random() < p_ctx else 0

                dist.broadcast(flag, src=0)
                use_context = bool(flag.item())
            else:
                use_context = (random.random() < p_ctx)

            # self.model._context_teacher = use_context
            # self.model._context_teacher = True

            if self.is_main_process:
                print(
                    f"Current step: {self.step}, "
                    f"p_ctx={p_ctx:.3f}, context_teacher={self.model._context_teacher}"
                )
            # ================================================================



            # =================== Train the generator ===================
            if (TRAIN_GENERATOR or self.config.teacher_forcing) and (not fake_only_phase):
                self.generator_optimizer.zero_grad(set_to_none=True)
                extras_list = []

                for i in range(accum_steps):
                    try:
                        batch = next(self.dataloader_iterator)
                    except StopIteration:
                        print("Iterator exhausted. Will create a new one in the next step.")
                        self.dataloader_iterator = None
                        break

                    is_last = (i == accum_steps - 1)
                    if self.config.long:
                        extra = self.fwdbwd_one_step_long(batch, True)
                    else:
                        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=self.use_amp):
                            extra = self.fwdbwd_one_step(
                                batch, True,
                                accum_steps=accum_steps,
                                is_last_micro=is_last
                            )
                    extras_list.append(extra)

                generator_log_dict = merge_dict_list(extras_list)
                if self.use_amp:
                    self.scaler.step(self.generator_optimizer)
                    self.scaler.update()
                else:
                    self.generator_optimizer.step()
                if self.schG is not None:
                    self.schG.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)

            # =================== Train the critic (fake score) ===================
            if not self.config.teacher_forcing:
                if fake_only_phase:
                    for g in self.critic_optimizer.param_groups:
                        g["lr"] = base_lr_d * 0.2
                else:
                    for g in self.critic_optimizer.param_groups:
                        g["lr"] = base_lr_d
                    if self.schD is None:
                        self.schD = self._constant_lr(self.critic_optimizer)

                self.critic_optimizer.zero_grad(set_to_none=True)
                extras_list = []

                for i in range(accum_steps):
                    try:
                        batch = next(self.dataloader_iterator)
                    except StopIteration:
                        print("Iterator exhausted. Will create a new one in the next step.")
                        self.dataloader_iterator = None
                        continue 

                    is_last = (i == accum_steps - 1)
                    if self.config.long:
                        extra = self.fwdbwd_one_step_long(batch, False)
                    else:
                        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=self.use_amp):
                            extra = self.fwdbwd_one_step(
                                batch, False,
                                accum_steps=accum_steps,
                                is_last_micro=is_last
                            )
                    extras_list.append(extra)

                critic_log_dict = merge_dict_list(extras_list)
                if self.use_amp:
                    self.scaler.step(self.critic_optimizer)
                    self.scaler.update()
                else:
                    self.critic_optimizer.step()
                if self.schD is not None:
                    self.schD.step()

            # =================== Step counter ===================
            self.step += 1

            if self.is_main_process:
                progress_bar.update(1)

            if self.step >= self.max_step:
                print(f"Training finished after reaching max_step: {self.max_step}")
                break

            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            # Save the model
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # =================== Logging ===================
            if self.is_main_process:
                wandb_loss_dict = {}

                if self.config.teacher_forcing:
                    wandb_loss_dict.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                        }
                    )
                else:
                    if (TRAIN_GENERATOR and (not fake_only_phase)):
                        wandb_loss_dict.update(
                            {
                                "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                                "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                                "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].mean().item()
                            }
                        )

                    wandb_loss_dict.update(
                        {
                            "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                            "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()
                        }
                    )

                if not self.disable_wandb and len(wandb_loss_dict) > 0:
                    wandb.log(wandb_loss_dict, step=self.step)

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time

        if self.is_main_process:
            progress_bar.close()
