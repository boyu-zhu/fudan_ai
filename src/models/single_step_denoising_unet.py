# Import essential modules. Feel free to add whatever you need.
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, transforms

import torch.optim as optim
from tqdm import tqdm
transform = transforms.Compose([
    transforms.ToTensor(), 

])

train_dataset = MNIST(
    root='./data',
    train=True, 
    transform=transform, 
    download=True 
)

test_dataset = MNIST(
    root='./data',
    train=False, 
    transform=transform
)


image, label = train_dataset[0] 



def add_noise(img, sigma):
    noise = torch.randn_like(img) * sigma
    noisy_img = img + noise
    return noisy_img

sigmas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

def normalize(img):
    img = (img - img.min()) / (img.max() - img.min())
    return img


fig, axes = plt.subplots(1, len(sigmas), figsize=(15, 3))
fig.suptitle('A visualization of the noising process using different σ')


for i, sigma in enumerate(sigmas):
    noisy_image = add_noise(image, sigma)
    axes[i].imshow(normalize(noisy_image).permute(1,2,0), cmap='gray')
    axes[i].set_title(f'σ = {sigma}')
    axes[i].axis('off')

plt.show()
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

def forward(x_0, t, alphas_cumprod):
    sqrt_alpha_bar_t = torch.sqrt(alphas_cumprod[t])
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alphas_cumprod[t])
    epsilon = torch.randn(*x_0.shape)
    x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * epsilon
    print(1,sqrt_one_minus_alpha_bar_t)
    return torch.clamp(x_t, min=-1.0, max=1.0)
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
def draw_losses_plot(losses):
        plt.figure(figsize=(10, 5))
        plt.plot(losses, label="Batch Loss")
        plt.xlabel("Batch Number")
        plt.ylabel("Loss")
        plt.title("Training Loss Per Batch")
        plt.legend()
        plt.grid()
        plt.show()
def show(model):
    test_dataset = MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.eval()
    with torch.no_grad():
        for clean_images, _ in tqdm(test_loader):
            clean_images = clean_images.to(device)
            noisy_images = add_noise(clean_images, sigma = 0.5)
            denoised_images = model(noisy_images)
            noisy_images = noisy_images.cpu().numpy()
            denoised_images = denoised_images.cpu().numpy()
            clean_images = clean_images.cpu().numpy()
            for i in range(3):
                plt.subplot(3, 3, i + 1)
                plt.imshow(clean_images[i][0], cmap='gray')
                plt.title("Clean Image")
                plt.axis('off')

                plt.subplot(3, 3, i + 4)
                plt.imshow(noisy_images[i][0], cmap='gray')
                plt.title("Noisy Image")
                plt.axis('off')

                plt.subplot(3, 3, i + 7)
                plt.imshow(denoised_images[i][0], cmap='gray')
                plt.title("Denoised Image")
                plt.axis('off')

            plt.show()
            break        
def train(train_dataset):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        model = UnconditionalUNet(in_channels=1, num_hiddens=128)
        model = model.to('cuda:0' if torch.cuda.is_available() else 'cpu')
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        num_epochs = 5
        # num_epochs = 1 # set to 1 to test model performance at 1 epoch

        losses = []
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for clean_images, _ in tqdm(train_loader):
                clean_images = clean_images.to(device)
                noisy_images = add_noise(clean_images, sigma = 0.5)
                outputs = model(noisy_images)
                loss = criterion(outputs, clean_images)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(torch.log(loss).item())
                total_loss += loss.item()
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

        draw_losses_plot(losses)
        show(model)
        
