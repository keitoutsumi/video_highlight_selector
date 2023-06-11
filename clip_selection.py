import torch
import torch.nn as nn
#clip selection model to select clips from feature vectors
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

def create_mlp_model(input_size, hidden_size, output_size):
    model = MLP(input_size, hidden_size, output_size)
    model = model.cuda()
    return model
