import torch
import torch.optim as optim
import torch.nn.functional as F
import dataset
import slowfast
import mlp
import matplotlib.pyplot as plt

def train_epoch(dataloader, slowfast_model, mlp_model, optimizer):
    slowfast_model.eval()
    mlp_model.train()
    total_loss = 0
    correct = 0

    for video_0, video_1 in dataloader:
        video_0, video_1 = video_0.cuda(), video_1.cuda()

        with torch.no_grad():
            features_0 = slowfast_model(video_0)
            features_1 = slowfast_model(video_1)

        optimizer.zero_grad()
        output = mlp_model(torch.cat((features_0, features_1), dim=1))
        target = torch.ones(output.size(0), dtype=torch.long).cuda()
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    return total_loss / len(dataloader), correct / len(dataloader)

def evaluate(dataloader, slowfast_model, mlp_model):
    slowfast_model.eval()
    mlp_model.eval()
    correct = 0

    with torch.no_grad():
        for video_0, video_1 in dataloader:
            video_0, video_1 = video_0.cuda(), video_1.cuda()
            features_0 = slowfast_model(video_0)
            features_1 = slowfast_model(video_1)
            output = mlp_model(torch.cat((features_0, features_1), dim=1))
            target = torch.ones(output.size(0), dtype=torch.long).cuda()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    return correct / len(dataloader)

def main():
    base_dir = "D:\\valorant_video_annotation"
    batch_size = 8
    epochs = 20
    lr = 0.001

    train_dataloader, test_dataloader = dataset.create_data_loaders(base_dir, batch_size)
    slowfast_model = slowfast.load_slowfast_model()
    mlp_model = mlp.create_mlp_model(1000, 500, 2)
    optimizer = optim.Adam(mlp_model.parameters(), lr=lr)

    train_losses, test_accuracies = [], []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(train_dataloader, slowfast_model, mlp_model, optimizer)
        test_acc = evaluate(test_dataloader, slowfast_model, mlp_model)
        train_losses.append(train_loss)
        test_accuracies.append(test_acc)
        print(f"Epoch: {epoch}, Train Loss: {train_loss}, Train Acc: {train_acc}, Test Acc: {test_acc}")

    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
