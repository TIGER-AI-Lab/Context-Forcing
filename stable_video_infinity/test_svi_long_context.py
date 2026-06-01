
import sys
import torch
import torch.nn as nn
from diffsynth import ModelManager, SVIVideoContextPipeline, save_video, VideoData
import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import pandas as pd
import torchvision
from PIL import Image
import numpy as np
import json
import glob
import random
import torchvision.transforms.functional as F
import torchvision.transforms as transforms

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_path, max_samples=None):
        try:
            self.metadata = pd.read_csv(metadata_path)
            if 'videoFile' not in self.metadata.columns and 'clip_id' in self.metadata.columns:
                 self.metadata['videoFile'] = self.metadata['clip_id']
            if 'caption' not in self.metadata.columns and 'Summarized Description' in self.metadata.columns:
                self.metadata['caption'] = self.metadata['Summarized Description']
        except:
            self.metadata = pd.read_csv(metadata_path, sep='\t')
        
        # Filter for valid entries if necessary, here we just take what's there
        if max_samples:
            self.metadata = self.metadata.sample(n=min(len(self.metadata), max_samples), random_state=42)
            self.metadata = self.metadata.reset_index(drop=True)
            
        self.samples = []
        for _, row in self.metadata.iterrows():
            self.samples.append({
                "path": str(row['videoFile']),
                "text": str(row['caption']) if not pd.isna(row['caption']) else "A video"
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def parse_args():
    parser = argparse.ArgumentParser(description="SVI Long Context Inference")
    parser.add_argument(
        "--metadata_file_path",
        type=str,
        default="metadata.csv",
        help="Path to the metadata CSV file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Path to save the results.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="Wan2.1-T2V-1.3B",
        choices=["Wan2.1-T2V-1.3B", "Wan2.1-T2V-14B",],
        help="Model type",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to the lora model.",
    )
    parser.add_argument(
        "--lora_alpha",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        '--num_chunks',
        type=int,
        default=4,
        help="Generate chunk number."
    )
    parser.add_argument(
        '--num_frames',
        type=int,
        default=84,
        help="Frames per chunk. Default 84 to match training target length."
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        '--tiled', 
        action='store_true', 
        default=True, 
        help="Enable tiled decoding."
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10,
        help="Max number of samples to generate."
    )
    
    args = parser.parse_args()
    return args

def get_context_input(video_tensor):
    # Logic from train_svi_long_context.py :: split_video_tensor
    # Adapting for inference where we feed the PREVIOUS generated frames to generate the NEXT chunk.
    # The 'video_tensor' here is the accumulated history of generated frames (C, T_history, H, W).
    
    # Constants from training script
    SINK_LEN = 9             
    MOTION_LEN = 24 
    MAX_CONTEXT_WINDOW = 18 * 16 # 18s video ~ 288 frames
    CTX_CHUNK_STRIDE = 24    
    CTX_CHUNK_KEEP = 4       

    C, T, H, W = video_tensor.shape
    
    # 1. Sink Frames: [0 : 9]
    if T < SINK_LEN:
         sink_frames = video_tensor
    else:
        sink_frames = video_tensor[:, :SINK_LEN, :, :]

    # 2. Motion Frames: [Last 24]
    # In training: motion is [T-108 : T-84], target is [T-84 : T]
    # In inference: We have history up to T_history. We want to predict next TARGET_LEN frames.
    # So our "Motion" comes from the end of current history.
    if T < MOTION_LEN:
        # Should not happen if first chunk is 84
        motion_frames = video_tensor 
    else:
        motion_frames = video_tensor[:, -MOTION_LEN:, :, :]

    # 3. Context Frames Logic
    # Region between sink and motion
    ctx_region_end = T - MOTION_LEN 
    
    ctx_region_start_ideal = ctx_region_end - MAX_CONTEXT_WINDOW
    ctx_region_start = max(SINK_LEN, ctx_region_start_ideal)
    
    context_chunks = []
    
    if ctx_region_start < ctx_region_end:
        ctx_raw = video_tensor[:, ctx_region_start - 1:ctx_region_end, :, :] # (C, T_ctx, H, W)
        
        if ctx_raw.shape[1] > 0:
            ctx_len = ctx_raw.shape[1]
            for i in range(0, ctx_len, CTX_CHUNK_STRIDE):
                start = i
                end = min(i + CTX_CHUNK_KEEP, ctx_len) + 1 
                
                if start < ctx_len: 
                     safe_end = min(end, ctx_len)
                     if start < safe_end:
                        chunk_tensor = ctx_raw[:, start:safe_end, :, :]
                        if chunk_tensor.shape[1] > 0:
                            context_chunks.append(chunk_tensor)
    
    print(f"  Context Construction: History T={T} -> Sink {sink_frames.shape}, Motion {motion_frames.shape}, Ctx {len(context_chunks)} chunks")
    
    return {
        "sink": sink_frames,       
        "context": context_chunks, 
        "motion": motion_frames,   
    }


if __name__ == '__main__':
    args = parse_args()

    # 1. Load Model
    model_manager = ModelManager(device="cpu", train_architecture='lora') # Assuming lora architecture
    
    if args.model_type == "Wan2.1-T2V-1.3B":
        model_manager.load_models([
            "/home/chenshuo/diffusion_pytorch_model.safetensors", # core model
            "../wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
            "../wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
        ])
    elif args.model_type == "Wan2.1-T2V-14B":
        # Load logic for 14B if needed, keeping simple for now
        pass 

    if args.lora_path:
        extra_module_root = args.lora_path
        if extra_module_root.endswith('.safetensors'):
            safetensors_files = [extra_module_root]
        else:
            safetensors_files = glob.glob(os.path.join(extra_module_root, "*.safetensors"))
            safetensors_files.sort()
        model_manager.load_lora_v2(safetensors_files, lora_alpha=args.lora_alpha)

    pipe = SVIVideoContextPipeline.from_model_manager(model_manager, device="cuda", torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Dataset
    dataset = VideoDataset(args.metadata_file_path, max_samples=args.max_samples)
    print(f"Loaded {len(dataset)} samples.")

    # 3. Inference Loop
    for i, sample in enumerate(dataset):
        prompt = sample['text']
        video_path = sample['path']
        print(f"Processing ({i}/{len(dataset)}): {prompt}")

        full_video_frames = [] # Store PIL images/arrays for saving
        
        # We need to maintain the history as a tensor (C, T, H, W)
        history_tensor = None 

        for chunk_id in range(args.num_chunks):
            print(f"  Generating Chunk {chunk_id}...")
            
            if chunk_id == 0:
                # T2V Generation for first chunk
                source_input = None 
            else:
                # Prepare input: Sink, Context, Motion
                source_input = get_context_input(history_tensor)

            with torch.no_grad():
                output_video = pipe(
                    prompt=prompt,
                    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                    source_video=source_input,
                    cfg_scale=args.cfg_scale,
                    num_inference_steps=50,
                    seed=42, 
                    tiled=args.tiled,
                    num_frames=args.num_frames
                )
            
            new_chunk_tensor = output_video[0].cpu() # (C, T, H, W)
            
            if history_tensor is None:
                history_tensor = new_chunk_tensor
            else:
                history_tensor = torch.cat([history_tensor, new_chunk_tensor], dim=1)
                
            # Convert to frames for saving
            new_frames = pipe.tensor2video(new_chunk_tensor.unsqueeze(0)) # Returns list of PIL images
            full_video_frames.extend(new_frames)

        # Save concatenated video
        save_name = f"sample_{i}_{random.randint(0, 10000)}.mp4"
        save_path = os.path.join(args.output_dir, save_name)
        save_video(full_video_frames, save_path, fps=16)
        print(f"Saved to {save_path}")