# train.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

class CAMCNN(nn.Module):
    def __init__(self):
        super(CAMCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)
        self.feature_map1 = None
        self.feature_map2 = None

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        self.feature_map1 = x.detach().cpu()
        x = self.pool(F.relu(self.conv2(x)))
        self.feature_map2 = x.detach().cpu()
        gap = self.gap(x).view(x.size(0), -1)
        out = self.fc(gap)
        return out


if __name__ == '__main__':
    # 数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_set = datasets.MNIST('.', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

    # 训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CAMCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # 计算当前 batch 正确数
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
            total += target.size(0)

            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/5], Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {running_loss / (batch_idx + 1):.4f}, "
                      f"Accuracy: {100.0 * correct / total:.2f}%")

        print(f"Epoch {epoch+1} finished. Avg Loss: {running_loss / len(train_loader):.4f}, "
              f"Train Accuracy: {100.0 * correct / total:.2f}%")

    # 保存模型
    torch.save(model.state_dict(), "mnist_cnn_cam.pth")
    print("Model saved as mnist_cnn_cam.pth")
