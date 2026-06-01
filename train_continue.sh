export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=$(hostname)
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_IFNAME=^docker0,lo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=5246 \
  --rdzv_backend=c10d \
  --rdzv_endpoint $MASTER_ADDR \
  train.py \
  --config_path configs/context_dmd_stage_1.yaml \
  --logdir "logs/context_dmd_stage_1_$(date +%m%d)" \
  --disable-wandb > ./logs/context_dmd_stage_1_$(date +%m%d).log 2>&1
