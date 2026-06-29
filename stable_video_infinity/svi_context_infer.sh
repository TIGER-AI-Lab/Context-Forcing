export CUDA_VISIBLE_DEVICES=0

python inference_recammaster_lora.py  \
    --dataset_path /path/to/test_videos  \
    --metadata_file_path /path/to/test_videos/test.csv \
    --output_dir ./outputs/svi_long_context_infer  \
    --model_type Wan2.1-T2V-1.3B \
    --num_chunks 6 \
    --lora_path ckpts/lora_weights.safetensors \
    # --from_scratch \
