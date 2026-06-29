export CUDA_VISIBLE_DEVICES=0

python inference.py \
    --config_path configs/context_dmd_inference.yaml \
    --output_folder outputs/test_$(date +%m%d) \
    --checkpoint_path checkpoints/model.pt \
    --num_output_frames 252 \
    --data_path prompts/demo_test.txt \
    --seed 7 \
    --use_ema