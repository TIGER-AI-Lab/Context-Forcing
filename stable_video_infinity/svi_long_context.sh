export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 #,6,7 #0 #,1,2,3, #4,5,6,7
export NCCL_P2P_DISABLE=1
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u train_svi_long_context.py \
    --learning_rate 1e-4 \
    --lora_rank 128 \
    --lora_alpha 128 \
    --dataset_path "/path/to/video_clips" \
    --metadata_file_path "/path/to/metadata.csv" \
    --dit_path "../wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
    --vae_path "../wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
    --text_encoder_path "../wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" \
    --max_epochs 10 \
    --train_architecture lora \
    --use_gradient_checkpointing \
    --use_gradient_checkpointing_offload \
    --training_strategy "deepspeed_stage_2" \
    --output_path "experiments/train/svi-long-context" \
    --use_error_recycling \
    --error_buffer_k 500 \
    --y_error_num 3 \
    --num_motion_frames 1 \
    --buffer_warmup_iter 50 \
    --buffer_replacement_strategy l2_batch \
    --y_error_sample_from_all_grids \
    --num_grids 50 \
    --ref_pad_num -1 \
    --noise_prob 0.01 \
    --y_prob 0.9 \
    --latent_prob 0.9 \
    --context_prob 0.9 \
    --clean_prob 0.5 \
    --clean_buffer_update_prob 0.1 \
    --batch_size 1 \
    --dataloader_num_workers 2 \
    --exp_prefix train-svi-shot-context