export CUDA_VISIBLE_DEVICES=0,1
export MASTER_ADDR=$(hostname)
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_IFNAME=^docker0,lo
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=2 --rdzv_id=5245 \
    --rdzv_backend=c10d \
    --rdzv_endpoint $MASTER_ADDR \
    inference.py \
    --config_path configs/context_dmd_inference.yaml \
    --output_folder outputs/test_dist \
    --checkpoint_path checkpoints/model.pt \
    --num_output_frames 252 \
    --data_path prompts/demo_test.txt \
    --seed 0 \
    --use_ema
