import torch
from torchvision.models.video import r3d_18
#video understanding model to extract video features
def load_slowfast_model():
    model = r3d_18(pretrained=True)
    model = model.cuda()
    model.eval()
    return model
