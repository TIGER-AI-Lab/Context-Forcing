export CUDA_VISIBLE_DEVICES=5

python inference.py \
    --config_path configs/self_forcing_dmd_laz_12_rope_continue_fix_long_Wan_new_lr_rollout_sink3_5s_sim_sample.yaml \
    --output_folder videos/sf_rope_continue_sim_sample_1min_0126_sink_1 \
    --checkpoint_path /map-vepfs/shuochen/Long_video/checkpoints/self_forcing_dmd.pt \
    --num_output_frames 252 \
    --data_path prompts/long_test.txt \
    --seed 1 \
    --use_ema \
    # semantic
    # long_test
    # MovieGenVideoBench_extended.txt
    # /data/shuochen/Long_video/checkpoints/self_forcing_dmd.pt