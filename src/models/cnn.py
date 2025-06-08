import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
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