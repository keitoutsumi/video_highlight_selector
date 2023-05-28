import torch
from torch.utils.data import Dataset
import os
import random
from typing import Tuple
import cv2
import numpy as np

class ValorantClipDataset(Dataset):
    def __init__(self, root_dir: str, split: str, ratio: float = 0.8, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.clips_0 = sorted(os.listdir(os.path.join(root_dir, "0")))
        self.clips_1 = sorted(os.listdir(os.path.join(root_dir, "1")))

        random.shuffle(self.clips_0)
        random.shuffle(self.clips_1)

        min_length = min(len(self.clips_0), len(self.clips_1))-10
        self.clips_0 = self.clips_0[:min_length]
        self.clips_1 = self.clips_1[:min_length]

        split_idx = int(min_length * ratio)

        if self.split == 'train':
            self.clips_0 = self.clips_0[:split_idx]
            self.clips_1 = self.clips_1[:split_idx]
        elif self.split == 'eval':
            self.clips_0 = self.clips_0[split_idx:]
            self.clips_1 = self.clips_1[split_idx:]

    def __len__(self):
        return min(len(self.clips_0), len(self.clips_1))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        clip_path_0 = os.path.join(self.root_dir, "0", self.clips_0[idx])
        clip_path_1 = os.path.join(self.root_dir, "1", self.clips_1[idx])

        video_tensor_0 = self._load_video_tensor(clip_path_0)
        video_tensor_1 = self._load_video_tensor(clip_path_1)
        label = 1

        if self.transform:
            video_tensor_0 = self.transform(video_tensor_0)
            video_tensor_1 = self.transform(video_tensor_1)
            
        switch = random.choice([True,False])
        
        if switch:
            video_tensor_0,video_tensor_1=video_tensor_1,video_tensor_0
            label = 1-label
            
        return video_tensor_0, video_tensor_1, label

    def _load_video_tensor(self, clip_path: str) -> torch.Tensor:
        cap = cv2.VideoCapture(clip_path)
        frames = []
        frame_count=0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count%4==0:
                frame = cv2.resize(frame, (320, 180))
                frames.append(frame)
        cap.release()
        frames_np = np.stack(frames, axis=0)
        return torch.tensor(frames_np, dtype=torch.float32).permute(3, 0, 1, 2) / 255.0
    
def custom_collate_fn(batch):
    max_frames = max([max(video_tensor_0.shape[1], video_tensor_1.shape[1]) for video_tensor_0, video_tensor_1, label in batch])

    padded_batch = []
    for video_tensor_0, video_tensor_1, label in batch:
        padding_0 = torch.zeros(video_tensor_0.shape[0], max_frames - video_tensor_0.shape[1], *video_tensor_0.shape[2:])
        padding_1 = torch.zeros(video_tensor_1.shape[0], max_frames - video_tensor_1.shape[1], *video_tensor_1.shape[2:])

        padded_video_tensor_0 = torch.cat([video_tensor_0, padding_0], dim=1)
        padded_video_tensor_1 = torch.cat([video_tensor_1, padding_1], dim=1)

        padded_batch.append((padded_video_tensor_0, padded_video_tensor_1, label))

    video_tensors_0, video_tensors_1, labels = zip(*padded_batch)

    return torch.stack(video_tensors_0, dim=0), torch.stack(video_tensors_1, dim=0), torch.tensor(labels)

