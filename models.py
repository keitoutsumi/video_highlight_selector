import torch
import torch.nn as nn
from torchvision.models.video import r3d_18

class VideoUnderstandingModel(nn.Module):
    def __init__(self, num_classes: int):
        super(VideoUnderstandingModel, self).__init__()
        self.model = r3d_18(weights=True)
        self.model.fc = nn.Identity()  # Remove the last fully connected layer to get features

    def forward(self, x):
        return self.model(x)

class ClipSelectionModel(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(ClipSelectionModel, self).__init__()
        # self.fc1 = nn.Linear(input_size, input_size // 2)
        # self.fc2 = nn.Linear(input_size // 2, output_size)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 2)
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=-1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x