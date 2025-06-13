# Import essential modules. Feel free to add whatever you need.
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, transforms

import torch.optim as optim
from tqdm import tqdm
from single_step_denoising_unet import * 
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


class TimeConditionalUNet(nn.Module):
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
def ddpm_schedule(beta1: float, beta2: float, num_ts: int) -> dict:
    """Constants for DDPM training and sampling.

    Arguments:
        beta1: float, starting beta value.
        beta2: float, ending beta value.
        num_ts: int, number of timesteps.

    Returns:
        dict with keys:
            betas: linear schedule of betas from beta1 to beta2.
            alphas: 1 - betas.
            alpha_bars: cumulative product of alphas.
    """
    assert beta1 < beta2 < 1.0, "Expect beta1 < beta2 < 1.0."
    betas = torch.linspace(beta1, beta2, num_ts)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    return {
        'betas': betas,
        'alphas': alphas,
        'alpha_bars': alpha_bars,
    }
def add_noise(x0, t, ddpm_schedule, noise):
    """
    Args:
        x0: (N, C, H, W) clean images
        t: (N,) time step tensor
        noise: (N, C, H, W) random noise

    Returns:
        xt: Noisy images at time t
    """
    # betas = ddpm_schedule['betas']
    # alphas = ddpm_schedule['alphas']
    alpha_bars = ddpm_schedule['alpha_bars'].to(x0.device)
    sqrt_alpha_t_bar = torch.sqrt(alpha_bars)
    sqrt_1_minus_alpha_t_bar = torch.sqrt(1.0 - alpha_bars)
    # noise = torch.randn_like(x0).to(device)
    batch_size = x0.size(0)
    # print(sqrt_alpha_t_bar.device)
    # print(t.device)

    sqrt_alpha_t_bar_t = sqrt_alpha_t_bar[t].view(batch_size, 1, 1, 1)  # 维度对齐
    sqrt_1_minus_alpha_t_bar_t = sqrt_1_minus_alpha_t_bar[t].view(batch_size, 1, 1, 1)
    return sqrt_alpha_t_bar_t * x0 + sqrt_1_minus_alpha_t_bar_t * noise


def ddpm_forward(
    unet: TimeConditionalUNet,
    ddpm_schedule: dict,
    x_0: torch.Tensor,
    num_ts: int,
) -> torch.Tensor:
    """Algorithm 1 of the DDPM paper.

    Args:
        unet: TimeConditionalUNet
        ddpm_schedule: dict
        x_0: (N, C, H, W) input tensor.
        num_ts: int, number of timesteps.
    Returns:
        (,) diffusion loss.
    """
    unet.train()
    # YOUR CODE HERE.
    N, C, H, W = x_0.shape

    t = torch.randint(0, num_ts, (N,), device=x_0.device)
    clean_images = x_0.clone()
    noise = torch.randn_like(x_0)
    
    xt = add_noise(x0=clean_images, ddpm_schedule=ddpm_schedule,t=t, noise =noise)
    predicted_noise = unet(xt, t)

    # loss = torch.mean((predicted_noise - noise) ** 2)
    loss = torch.nn.functional.mse_loss(predicted_noise, noise)

    return loss
@torch.inference_mode()
def ddpm_sample(
    unet: TimeConditionalUNet,
    ddpm_schedule: dict,
    img_wh: tuple[int, int],
    num_ts: int,
    seed: int = 0,
) -> torch.Tensor:
    """Algorithm 2 of the DDPM paper with classifier-free guidance.

    Args:
        unet: TimeConditionalUNet
        ddpm_schedule: dict
        img_wh: (H, W) output image width and height.
        num_ts: int, number of timesteps.
        seed: int, random seed.

    Returns:
        (N, C, H, W) final sample.
    """
    unet.eval()
    # YOUR CODE HERE.
    x_t = torch.randn((1, 1, *img_wh), device=device) 

    betas = ddpm_schedule['betas']
    alphas = ddpm_schedule['alphas']
    alpha_bars = ddpm_schedule['alpha_bars']

    for t in range(num_ts - 1, -1, -1):
        # Calculate noise prediction
        noise_pred = unet(x_t, torch.tensor([t],device=device))  # UNet prediction for this time step
        noise = torch.randn_like(x_t)  # Random noise
        
        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_bar_t = alpha_bars[t]
        alpha_bar_prev_t = alpha_bars[t-1]

        if t > 0:
            z = torch.randn_like(x_t)
        else:
            z = torch.zeros_like(x_t)
        x0 = (1 / torch.sqrt(alpha_bar_t)) * (x_t - torch.sqrt(1 - alpha_bar_t) * noise_pred)
        term1 = torch.sqrt(alpha_bar_prev_t) * beta_t / (1 - alpha_bar_t) * x0
        term2 = torch.sqrt(alpha_t) * (1 - alpha_bar_prev_t) / (1 - alpha_bar_t) * x_t
        term3 = torch.sqrt(beta_t) * noise
        x_t = term1 + term2 + term3


    return x_t

class DDPM(nn.Module):
    def __init__(
        self,
        unet: TimeConditionalUNet,
        betas: tuple[float, float] = (1e-4, 0.02),
        num_ts: int = 300,
        p_uncond: float = 0.1,
    ):
        super().__init__()
        self.unet = unet
        self.betas = betas
        self.num_ts = num_ts
        self.p_uncond = p_uncond
        self.ddpm_schedule = ddpm_schedule(betas[0], betas[1], num_ts)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, H, W) input tensor.

        Returns:
            (,) diffusion loss.
        """
        return ddpm_forward(
            self.unet, self.ddpm_schedule, x, self.num_ts
        )

    @torch.inference_mode()
    def sample(
        self,
        img_wh: tuple[int, int],
        seed: int = 0,
        
    ):
        return ddpm_sample(
            self.unet, self.ddpm_schedule, img_wh, self.num_ts, seed
        )
        
def train(train_loader,model):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    num_timesteps = 300
    num_epochs = 20
    learning_rate = 1e-3
    gamma = 0.1 ** (1.0 / num_epochs)
    losses = []

    sample_epochs = {5, 20}  

    sampled_images_dict = {}
    unet = TimeConditionalUNet(        
            in_channels = 1,
            num_classes = num_timesteps,
            num_hiddens = 64,)
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    for epoch in range(num_epochs):
        total_loss = 0
        for x, c in tqdm(train_loader):
            x, c = x.to(device), c.to(device)

            loss = model(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            # print(loss.item())
            losses.append(torch.log(loss).item())
            total_loss += loss.item()
        scheduler.step()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

        if epoch + 1 in sample_epochs:
            model.eval()
            with torch.no_grad():
                num_samples = 16
                img_wh = (28, 28)
                sampled_images = []

                for _ in range(num_samples):
                    sampled_image = model.sample(img_wh=img_wh, seed=epoch + 1)
                    sampled_image = (
                        sampled_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                    )
                    sampled_images.append(sampled_image)
                
                sampled_images_dict[epoch + 1] = sampled_images

            model.train()

    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Batch Loss")
    plt.xlabel("Batch Number")
    plt.ylabel("Loss")
    plt.title("Training Loss Per Batch")
    plt.legend()
    plt.grid()
    plt.show()

    for epoch, sampled_images in sampled_images_dict.items():
        grid_size = (4, 4)
        fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(8, 8))

        for i, image in enumerate(sampled_images):
            row = i // grid_size[1]
            col = i % grid_size[1]
            axes[row, col].imshow(image, cmap="gray")
            axes[row, col].axis("off")

        plt.suptitle(f"Sampled Images at Epoch {epoch}")
        plt.tight_layout()
        plt.show()