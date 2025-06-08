# Import essential modules. Feel free to add whatever you need.
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, transforms

import torch.optim as optim
from tqdm import tqdm

def forward(x_0, t, alphas_cumprod):
    sqrt_alpha_bar_t = torch.sqrt(alphas_cumprod[t])
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alphas_cumprod[t])
    epsilon = torch.randn(*x_0.shape)
    x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * epsilon
    print(1,sqrt_one_minus_alpha_bar_t)
    return torch.clamp(x_t, min=-1.0, max=1.0)


class Conv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layer = nn.Sequential(
          nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), 
          nn.BatchNorm2d(out_channels),            
          nn.GELU()          
)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class DownConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),           
            nn.GELU()
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),    
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=7)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avg_pool(x)


class Unflatten(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.unflatten = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=7, stride=7, padding=0),
            nn.BatchNorm2d(in_channels),     
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unflatten(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            Conv(in_channels, out_channels),
            Conv(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            DownConv(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            UpConv(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UnconditionalUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_hiddens: int,
    ):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, num_hiddens)
        self.down1 = DownBlock(num_hiddens, num_hiddens)
        self.down2 = DownBlock(num_hiddens, num_hiddens * 2)

        self.flatten = Flatten()
        self.unflatten = Unflatten(num_hiddens * 2)

        self.up2 = UpBlock(num_hiddens * 4, num_hiddens * 1)
        self.up1 = UpBlock(num_hiddens * 2, num_hiddens)

        self.dec1 = ConvBlock(num_hiddens * 2, num_hiddens)
        self.final = nn.Conv2d(num_hiddens, in_channels, kernel_size=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-2:] == (28, 28), "Expect input shape to be (28, 28)."
        enc1 = self.enc1(x)
        enc2 = self.down1(enc1)
        enc3 = self.down2(enc2)

        bottleneck = self.flatten(enc3)
        bottleneck = self.unflatten(bottleneck)

        dec2 = self.up2(torch.cat([bottleneck, enc3], dim=1))
        dec1 = self.up1(torch.cat([dec2, enc2], dim=1))
        out = self.dec1(torch.cat([dec1, enc1], dim=1))
        return self.final(out)


class FCBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=out_channels),
            nn.GELU(), 
            nn.Linear(in_features=out_channels, out_features=out_channels)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TimeConditionalUNetModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_hiddens: int,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.enc1 = ConvBlock(in_channels, num_hiddens)
        self.down1 = DownBlock(num_hiddens, num_hiddens)
        self.down2 = DownBlock(num_hiddens, num_hiddens * 2)

        self.flatten = Flatten()
        self.unflatten = Unflatten(num_hiddens * 2)

        self.up2 = UpBlock(num_hiddens * 4, num_hiddens * 1)
        self.up1 = UpBlock(num_hiddens * 2, num_hiddens)

        self.dec1 = ConvBlock(num_hiddens * 2, num_hiddens)
        self.final = nn.Conv2d(num_hiddens, in_channels, kernel_size=1)

        self.fc1_t = FCBlock(1, num_hiddens * 2)
        self.fc2_t = FCBlock(1, num_hiddens)


    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (N, C, H, W) input tensor.
            t: (N,) normalized time tensor.

        Returns:
            (N, C, H, W) output tensor.
        """
        assert x.shape[-2:] == (28, 28), "Expect input shape to be (28, 28)."
        t = t.unsqueeze(1) / self.num_classes
        t1 = self.fc1_t(t).unsqueeze(2).unsqueeze(3)
        t2 = self.fc2_t(t).unsqueeze(2).unsqueeze(3)

        enc1 = self.enc1(x)
        enc2 = self.down1(enc1)
        enc3 = self.down2(enc2)

        bottleneck = self.flatten(enc3)
        bottleneck = self.unflatten(bottleneck) + t1

        dec2 = self.up2(torch.cat([bottleneck, enc3], dim=1)) + t2
        dec1 = self.up1(torch.cat([dec2, enc2], dim=1))
        out = self.dec1(torch.cat([dec1, enc1], dim=1))
        return self.final(out)