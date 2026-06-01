export CUDA_VISIBLE_DEVICES=5 #,1,2,3 #,4,5,6,7 #0 #,1,2,3, #4,5,6,7
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# "/data/shuochen/Long_video/wan_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors,/data/shuochen/Long_video/wan_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors,/data/shuochen/Long_video/wan_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors,/data/shuochen/Long_video/wan_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors,/data/shuochen/Long_video/wan_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors,/data/shuochen/Long_video/wan_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors"  \

python inference_recammaster_lora.py  \
    --dataset_path /home/chenshuo/data/long_video_test  \
    --metadata_file_path /home/chenshuo/data/long_video_test/test.csv \
    --output_dir ./outputs/text_long_0115_3500_6_chunks_lr  \
    --model_type Wan2.1-T2V-1.3B \
    --num_chunks 6 \
    --lora_path /home/chenshuo/Long_video/stable_video_infinity/ckpts/lora_weights.safetensors \
    # --from_scratch \
