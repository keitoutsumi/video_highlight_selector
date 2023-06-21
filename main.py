import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import ValorantClipDataset, custom_collate_fn
from models import VideoUnderstandingModel, ClipSelectionModel
from utils import visualize_training
import numpy as np
from tqdm import tqdm

def train(device, dataloader, understanding_model, selection_model, optimizer, criterion):
    understanding_model.eval()
    selection_model.train()

    running_loss = 0
    correct = 0 #number of correct predictions
    total = 0 #number of pairs processed
    for i, (video_tensor_0, video_tensor_1, label) in enumerate(tqdm(dataloader,desc="Training")):
    # for i, (video_tensor_0, video_tensor_1, label) in enumerate(dataloader):
        video_tensor_0, video_tensor_1, label = video_tensor_0.to(device), video_tensor_1.to(device), label.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            feature_0 = understanding_model(video_tensor_0)
            feature_1 = understanding_model(video_tensor_1)

        output = selection_model(feature_0, feature_1)
        _, predicted = torch.max(output, 1)#predictedd label
        loss = criterion(output, label)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total += label.size(0)
        correct += (predicted == label).sum().item()

    return running_loss / (i + 1), correct / total

def evaluate(device, dataloader, understanding_model, selection_model):
    understanding_model.eval()
    selection_model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for video_tensor_0, video_tensor_1, label in tqdm(dataloader, desc="Evaluating"):
            video_tensor_0, video_tensor_1, label = video_tensor_0.to(device), video_tensor_1.to(device), label.to(device)

            feature_0 = understanding_model(video_tensor_0)
            feature_1 = understanding_model(video_tensor_1)
            output = selection_model(feature_0, feature_1)
            _, predicted = torch.max(output, 1)

            total += label.size(0)
            correct += (predicted == label).sum().item()

    return correct / total


def main():
    device = torch.device("cuda")
    root_dir = "F:\\valorant_video_annotation"
    epochs = 100

    train_dataset = ValorantClipDataset(root_dir, split='train')
    eval_dataset = ValorantClipDataset(root_dir, split='eval')

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True,collate_fn=custom_collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False,collate_fn=custom_collate_fn)

    understanding_model = VideoUnderstandingModel(num_classes=2).to(device)
    selection_model = ClipSelectionModel(input_size=1024, output_size=2).to(device)

    optimizer = optim.Adam(selection_model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    loss_history = []
    accuracy_history = []
    # print("everything before training is ok")

    for epoch in range(epochs):
        train_loss, train_accuracy = train(device, train_dataloader, understanding_model, selection_model, optimizer, criterion)
        torch.cuda.empty_cache()
        print(f"Epoch {epoch + 1}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")

        loss_history.append(train_loss)
        accuracy_history.append(train_accuracy)

    visualize_training(loss_history, accuracy_history)

    eval_accuracy = evaluate(device, eval_dataloader, understanding_model, selection_model)
    torch.cuda.empty_cache()
    print(f"Evaluation Accuracy: {eval_accuracy:.4f}")

if __name__ == "__main__":
    main()
