import os
import random
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor
import cv2
from PIL import Image

class VideoDataset(Dataset):
    def __init__(self, base_dir, transform=None):
        self.base_dir = base_dir
        self.transform = transform
        self.folder_0 = os.path.join(self.base_dir, "0")
        self.folder_1 = os.path.join(self.base_dir, "1")
        self.files_0 = os.listdir(self.folder_0)
        self.files_1 = os.listdir(self.folder_1)

    def __len__(self):
        return min(len(self.files_0), len(self.files_1))

    def __getitem__(self, index):
        video_0_path = os.path.join(self.folder_0, self.files_0[index])
        video_1_path = os.path.join(self.folder_1, self.files_1[index])

        video_0 = self.read_video(video_0_path)
        video_1 = self.read_video(video_1_path)

        video_0 = [Image.fromarray(frame) for frame in video_0]
        video_1 = [Image.fromarray(frame) for frame in video_1]
        
        if self.transform:
            video_0 = [self.transform(frame) for frame in video_0]
            video_1 = [self.transform(frame) for frame in video_1]

        return video_0, video_1

    def read_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

def get_transforms():
    return Compose([Resize((256, 256)), ToTensor()])

def create_data_loaders(base_dir, batch_size):
    transform = get_transforms()
    dataset = VideoDataset(base_dir, transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader
