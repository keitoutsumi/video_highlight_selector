import torch
from torchvision.models.video import r3d_18

def load_slowfast_model():
    model = r3d_18(pretrained=True)
    model = model.cuda()
    model.eval()
    return model