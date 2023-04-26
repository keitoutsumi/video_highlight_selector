import os
import random
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor
import cv2
from PIL import Image

class VideoDataset(Dataset):
    def __init__(self, base_dir, transform=None, num_frames=90, split_ratio=0.8):
        self.base_dir = base_dir
        self.transform = transform
        self.folder_0 = os.path.join(self.base_dir, "0")
        self.folder_1 = os.path.join(self.base_dir, "1")
        self.files_0 = os.listdir(self.folder_0)
        self.files_1 = os.listdir(self.folder_1)
        self.num_frames = num_frames
        (self.train_files_0, self.train_files_1), (self.test_files_0, self.test_files_1) = self.split_data(split_ratio)

    def __len__(self):#returns minimum size of videos in folder
        return min(len(self.files_0), len(self.files_1))

    def __getitem__(self, index, mode='train'):
        if mode == 'train':
            video_0_path = os.path.join(self.folder_0, self.train_files_0[index])
            video_1_path = os.path.join(self.folder_1, self.train_files_1[index])
        elif mode == 'test':
            video_0_path = os.path.join(self.folder_0, self.test_files_0[index])
            video_1_path = os.path.join(self.folder_1, self.test_files_1[index])
        else:
            raise ValueError("Invalid mode. Choose either 'train' or 'test'.")

        video_0 = self.read_video(video_0_path)
        video_1 = self.read_video(video_1_path)
        
        #numpy -> PIL
        video_0 = [Image.fromarray(frame) for frame in video_0]
        video_1 = [Image.fromarray(frame) for frame in video_1]
        
        if self.transform:
            video_0 = [self.transform(frame) for frame in video_0]
            video_1 = [self.transform(frame) for frame in video_1]

        return video_0, video_1


    def split_data(self,split_ratio):
        random.shuffle(self.files_0)
        random.shuffle(self.files_1)
        
        train_len=int(self.__len__*split_ratio)
        
        train_files_0=self.files_0[:train_len]
        train_files_1=self.files_1[:train_len]
        test_files_0=self.files_0[train_len:self.__len__]
        test_files_1=self.files_1[train_len:self.__len__]
        
        return (train_files_0,train_files_1),(test_files_0,test_files_1)

    def read_video(self, video_path):#converts a list of frames given a video (resamples frame size to 90)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        
        if total_frames>self.num_frames:
            step = total_frames//self.num_frames
            for i in range(self.num_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES,i*step)
                ret,frame=cap.read()
                if ret:
                    frames.append(frame)
        else:
            for i in range(self.num_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES,i%total_frames)
                ret,frame=cap.read()
                if ret:
                    frames.append(frame)
        
        cap.release()
        return frames

def get_transforms():
    return Compose([Resize((256, 256)), ToTensor()])

def create_data_loaders(base_dir, batch_size):
    transform = get_transforms()
    dataset= VideoDataset(base_dir,transform)
    train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
    test_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,collate_fn=collate_fn)
    return data_loader


def collate_fn(batch):#custom collate_fn
    video_0_batch, video_1_batch = [], []
    
    for video_0, video_1 in batch:
        video_0_batch.append(torch.stack(video_0))
        video_1_batch.append(torch.stack(video_1))

    video_0_batch = torch.stack(video_0_batch)
    video_1_batch = torch.stack(video_1_batch)

    return video_0_batch, video_1_batch

