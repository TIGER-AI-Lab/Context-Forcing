export CUDA_VISIBLE_DEVICES=4,5
export MASTER_ADDR=$(hostname)
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_IFNAME=^docker0,lo
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=2 --rdzv_id=5245 \
    --rdzv_backend=c10d \
    --rdzv_endpoint $MASTER_ADDR \
    inference.py \
    --config_path configs/self_forcing_dmd_laz_12_rope_continue_fix_long_Wan_new_lr_rollout_sink3_5s_sim_sample.yaml \
    --output_folder videos/ll_context_sim_95_1min \
    --checkpoint_path /map-vepfs/shuochen/LongLive/longlive_models/generator_fused_lora.ckpt \
    --num_output_frames 252 \
    --data_path prompts/MovieGenVideoBench_extended.txt \
    --seed 0 \