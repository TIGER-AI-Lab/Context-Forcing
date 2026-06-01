from utils.lmdb import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from torch.utils.data import Dataset
import numpy as np
import torch
import lmdb
import json
from pathlib import Path
from PIL import Image
import pandas as pd
from decord import VideoReader, cpu
import torchvision.transforms.v2 as v2
import os
import random


class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None, seed=42, shuffle=True):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

        if shuffle:
            rng = random.Random(seed)
            
            indices = list(range(len(self.prompt_list)))
            rng.shuffle(indices)

            self.prompt_list = [self.prompt_list[i] for i in indices]
            if self.extended_prompt_list is not None:
                self.extended_prompt_list = [self.extended_prompt_list[i] for i in indices]

            print(f"--- Dataset initialized and shuffled once with seed: {seed}. Total samples: {len(self.prompt_list)} ---")

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch

class TextCSVDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        # with open(prompt_path, encoding="utf-8") as f:
        #     self.prompt_list = [line.rstrip() for line in f]

        self.annotations = pd.read_csv(prompt_path)
        self.name_list = self.annotations["clip_id"].to_list()
        self.prompt_list = self.annotations["Summarized Description"].to_list()

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "names": self.name_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class ODERegressionLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.max_pair = max_pair

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env,
            "prompts", str, idx
        )
        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }


class ShardingLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.envs = []
        self.index = []

        for fname in sorted(os.listdir(data_path)):
            path = os.path.join(data_path, fname)
            env = lmdb.open(path,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
            self.envs.append(env)

        self.latents_shape = [None] * len(self.envs)
        for shard_id, env in enumerate(self.envs):
            self.latents_shape[shard_id] = get_array_shape_from_lmdb(env, 'latents')
            for local_i in range(self.latents_shape[shard_id][0]):
                self.index.append((shard_id, local_i))

            # print("shard_id ", shard_id, " local_i ", local_i)

        self.max_pair = max_pair

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
            Outputs:
                - prompts: List of Strings
                - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        shard_id, local_idx = self.index[idx]

        latents = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "latents", np.float16, local_idx,
            shape=self.latents_shape[shard_id][1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "prompts", str, local_idx
        )

        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }


class TextImagePairDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        eval_first_n=-1,
        pad_to_multiple_of=None
    ):
        """
        Args:
            data_dir (str): Path to the directory containing:
                - target_crop_info_*.json (metadata file)
                - */ (subdirectory containing images with matching aspect ratio)
            transform (callable, optional): Optional transform to be applied on the image
        """
        self.transform = transform
        data_dir = Path(data_dir)

        # Find the metadata JSON file
        metadata_files = list(data_dir.glob('target_crop_info_*.json'))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata file found in {data_dir}")
        if len(metadata_files) > 1:
            raise ValueError(f"Multiple metadata files found in {data_dir}")

        metadata_path = metadata_files[0]
        # Extract aspect ratio from metadata filename (e.g. target_crop_info_26-15.json -> 26-15)
        aspect_ratio = metadata_path.stem.split('_')[-1]

        # Use aspect ratio subfolder for images
        self.image_dir = data_dir / aspect_ratio
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        eval_first_n = eval_first_n if eval_first_n != -1 else len(self.metadata)
        self.metadata = self.metadata[:eval_first_n]

        # Verify all images exist
        for item in self.metadata:
            image_path = self.image_dir / item['file_name']
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

        self.dummy_prompt = "DUMMY PROMPT"
        self.pre_pad_len = len(self.metadata)
        if pad_to_multiple_of is not None and len(self.metadata) % pad_to_multiple_of != 0:
            # Duplicate the last entry
            self.metadata += [self.metadata[-1]] * (
                pad_to_multiple_of - len(self.metadata) % pad_to_multiple_of
            )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Returns:
            dict: A dictionary containing:
                - image: PIL Image
                - caption: str
                - target_bbox: list of int [x1, y1, x2, y2]
                - target_ratio: str
                - type: str
                - origin_size: tuple of int (width, height)
        """
        item = self.metadata[idx]

        # Load image
        image_path = self.image_dir / item['file_name']
        image = Image.open(image_path).convert('RGB')

        # Apply transform if specified
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'prompts': item['caption'],
            'target_bbox': item['target_crop']['target_bbox'],
            'target_ratio': item['target_crop']['target_ratio'],
            'type': item['type'],
            'origin_size': (item['origin_width'], item['origin_height']),
            'idx': idx
        }


def cycle(dl):
    while True:
        for data in dl:
            yield data


class UltraVidDataset(Dataset):
    def __init__(self, csv_path, video_folder, 
                 prompt_column='Summarized Description', 
                 num_frames=16, 
                 height=480, 
                 width=832,
                 clip_duration_sec=10.0): # The desired clip duration is now a key parameter
        """
        Args:
            csv_path (str): Path to the CSV file with annotations.
            video_folder (str): Path to the folder containing the full video files.
            prompt_column (str): The name of the column in the CSV that contains the text prompts.
            num_frames (int): The number of frames to sample from the target clip duration.
            height (int): The target height for the video frames.
            width (int): The target width for the video frames.
            clip_duration_sec (float): The duration in seconds of the clip to sample from (e.g., 10.0 for 10s).
        """
        self.annotations = pd.read_csv(csv_path)
        self.video_folder = video_folder
        self.prompt_column = prompt_column
        self.num_frames = num_frames
        self.clip_duration_sec = clip_duration_sec
        
        self.frame_process = v2.Compose([
            v2.ToTensor(),
            v2.Resize(size=(height, width), antialias=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Retrieves a sample, ensuring it's robust against metadata/file mismatches.
        """
        data_row = self.annotations.iloc[idx]
        video_filename = data_row['clip_id']
        prompt = data_row[self.prompt_column]
        
        file_total_frames_from_csv = data_row['total_frames']
        file_duration_from_csv = data_row['duration']

        video_path = os.path.join(self.video_folder, video_filename)

        try:
            # 1. Open the video file FIRST to get its TRUE length.
            vr = VideoReader(video_path, ctx=cpu(0))
            actual_total_frames = len(vr)

            # 2. Calculate the desired end frame based on CSV metadata.
            sampling_end_frame = file_total_frames_from_csv
            if file_duration_from_csv > 0:
                fps = file_total_frames_from_csv / file_duration_from_csv
                clip_frame_count = int(fps * self.clip_duration_sec)
                sampling_end_frame = min(clip_frame_count, file_total_frames_from_csv)
            else:
                print(f"Warning: Video {video_filename} has invalid duration. Using total frames from CSV.")

            # 3. THE FIX: Cap the sampling range using the ACTUAL frame count from the video file.
            #    This prevents the "Out of bound indices" error.
            final_end_frame = min(sampling_end_frame, actual_total_frames)

            # Ensure there are enough frames to sample from.
            if final_end_frame < self.num_frames:
                print(f"Warning: The valid sampling range ({final_end_frame} frames) is smaller than "
                    f"num_frames ({self.num_frames}). Some frames will be duplicated.")
                # If the entire video is shorter than num_frames, sample from the whole video.
                if actual_total_frames < self.num_frames:
                    final_end_frame = actual_total_frames

            # 4. Generate indices within the SAFE, corrected range.
            #    We use max(0, ...) to prevent a negative range if final_end_frame is 0.
            indices = np.linspace(0, max(0, final_end_frame - 1), self.num_frames, dtype=int)
            
            # Get frames using the safe indices.
            video_frames = vr.get_batch(indices).asnumpy()
            
            # Process the frames (using the workaround loop for older torchvision).
            processed_frames = [self.frame_process(frame) for frame in video_frames]
            video_tensor = torch.stack(processed_frames, dim=0)
            video_tensor = video_tensor.permute(1, 0, 2, 3)

            return {"video_tensor": video_tensor, "prompts": prompt}

        except Exception as e:
            print(f"Error loading or processing video at index {idx} with path {video_path}: {e}")
            return {"video_tensor": None, "prompts": None}


if __name__ == "__main__":
    import torchvision

    class Config:
        long_csv_file_path = "/home/c58wei/scratch/worldmodel/data/UltraVideo/long.csv"
        long_videos_directory = "/home/c58wei/scratch/worldmodel/data/UltraVideo/clips_long_960/clips_long_960"

    config = Config()
    dataset = UltraVidDataset(
                        csv_path=config.long_csv_file_path,
                        video_folder=config.long_videos_directory,
                        prompt_column='Summarized Description', # 或 'Brief Description', 'Detailed Description', etc.
                        num_frames=161
                    )


    print(f"✅ dataset length: {len(dataset)}")

    
    random_index = random.randint(0, len(dataset) - 1)

    batch = dataset.__getitem__(random_index) 

    video_tensor, prompt = batch["video_tensor"], batch["prompts"]
    import ipdb; ipdb.set_trace()

    output_filename = "test_sample.mp4"
    denormalized_tensor = video_tensor * 0.5 + 0.5
    
    # 2. Scale: Multiply by 255 to get the range [0, 255]
    scaled_tensor = denormalized_tensor * 255
    permuted_tensor = scaled_tensor.permute(1, 2, 3, 0)
    
    # 4. Convert to uint8: This is the correct data type for video frames
    tensor_to_save = permuted_tensor.to(torch.uint8)
    torchvision.io.write_video(
            filename=output_filename,
            video_array=tensor_to_save,
            fps=16,
            video_codec='h264' # Use a common codec
        )

    print(video_tensor, prompt)
