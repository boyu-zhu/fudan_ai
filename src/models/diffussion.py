import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, transforms

import torch.optim as optim
from tqdm import tqdm



def ddpm_forward(
    unet: ClassConditionalUNet,
    ddpm_schedule: dict,
    x_0: torch.Tensor,
    c: torch.Tensor,
    p_uncond: float,
    num_ts: int,
) -> torch.Tensor:
    """Algorithm 1 of the DDPM paper.

    Args:
        unet: ClassConditionalUNet
        ddpm_schedule: dict
        x_0: (N, C, H, W) input tensor.
        c: (N,) int64 condition tensor.
        p_uncond: float, probability of unconditioning the condition.
        num_ts: int, number of timesteps.

    Returns:
        (,) diffusion loss.
    """
    unet.train()
    device = c.device
    N = x_0.size(0)
    num_classes = unet.num_classes
    one_hot_c = torch.nn.functional.one_hot(c, num_classes=num_classes).float()
    mask = torch.bernoulli(torch.full((N,), 1 - p_uncond)).to(x_0.device)
    one_hot_c = one_hot_c * mask.unsqueeze(1)

    t = torch.randint(0, num_ts, (N,), device=device)
    epsilon = torch.randn_like(x_0)

    x_t = add_noise(x_0, t, ddpm_schedule, epsilon)
    epsilon_pred = unet(x=x_t, t=t, c=one_hot_c)

    loss = torch.nn.functional.mse_loss(epsilon_pred, epsilon)

    return loss



@torch.inference_mode()
def ddpm_sample(
    unet: ClassConditionalUNet,
    ddpm_schedule: dict,
    c: torch.Tensor,
    img_wh: tuple[int, int],
    num_ts: int,
    guidance_scale: float = 5.0,
    seed: int = 0,
) -> torch.Tensor:
    """Algorithm 2 of the DDPM paper with classifier-free guidance.

    Args:
        unet: ClassConditionalUNet
        ddpm_schedule: dict
        c: (N,) int64 condition tensor. Only for class-conditional
        img_wh: (H, W) output image width and height.
        num_ts: int, number of timesteps.
        guidance_scale: float, CFG scale.
        seed: int, random seed.

    Returns:
        (N, C, H, W) final sample.
        (N, T_animation, C, H, W) caches.
    """
    unet.eval()
    # YOUR CODE HERE.
    torch.manual_seed(seed)

    N = c.size(0)
    H, W = img_wh
    device = c.device

    x_t = torch.randn((N, 1, H, W), device=device)
    num_classes = unet.num_classes
    one_hot_c = torch.nn.functional.one_hot(c, num_classes=num_classes).float().to(device)
    zero_matrix = torch.zeros_like(one_hot_c)

    animation_cache = []

    betas = ddpm_schedule['betas']
    alphas = ddpm_schedule['alphas']
    alpha_bars = ddpm_schedule['alpha_bars']

    for t in range(num_ts, 0, -1): 
        t_tensor = torch.full((N,), t, device=device, dtype=torch.long)
        epsilon_pred_cond = unet(x_t, t=t_tensor, c=one_hot_c)
        epsilon_pred_uncond = unet(x_t, t=t_tensor, c=zero_matrix)

        epsilon_pred = (
            epsilon_pred_uncond + guidance_scale * (epsilon_pred_cond - epsilon_pred_uncond)
        )

        if t > 1:
            noise = torch.randn_like(x_t)
        else:
            noise = torch.zeros_like(x_t)

        beta_t = betas[t - 1]
        alpha_t = alphas[t - 1]
        alpha_bar_t = alpha_bars[t - 1]
        alpha_bar_prev_t = alpha_bars[t-2] if t > 1 else torch.tensor(1.0,device=device)


        x0 = (1 / torch.sqrt(alpha_bar_t)) * (x_t - torch.sqrt(1 - alpha_bar_t) * epsilon_pred)
        term1 = torch.sqrt(alpha_bar_prev_t) * beta_t / (1 - alpha_bar_t) * x0
        term2 = torch.sqrt(alpha_t) * (1 - alpha_bar_prev_t) / (1 - alpha_bar_t) * x_t
        term3 = torch.sqrt(beta_t) * noise
        x_t = term1 + term2 + term3
        animation_cache.append(x_t)


    return x_t, animation_cache


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


class DDPM(nn.Module):
    def __init__(
        self,
        unet: ClassConditionalUNet,
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

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, H, W) input tensor.
            c: (N,) int64 condition tensor.

        Returns:
            (,) diffusion loss.
        """
        return ddpm_forward(
            self.unet, self.ddpm_schedule, x, c, self.p_uncond, self.num_ts
        )

    @torch.inference_mode()
    def sample(
        self,
        c: torch.Tensor,
        img_wh: tuple[int, int],
        guidance_scale: float = 5.0,
        seed: int = 0,
    ):
        return ddpm_sample(
            self.unet, self.ddpm_schedule, c, img_wh, self.num_ts, guidance_scale, seed
        )