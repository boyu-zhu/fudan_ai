import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
from torchvision import transforms
from torchvision.datasets import MNIST
from model import CAMCNN

os.makedirs("vis", exist_ok=True)

from math import ceil, sqrt
from PIL import Image

def plot_feature_map(feature_map, layer_name, save_path=None):
    fmap = feature_map[0].cpu().numpy()  # [C, H, W]
    num_filters = fmap.shape[0]

    # 自动计算行列数，尽可能接近正方形排列
    num_cols = ceil(sqrt(num_filters))
    num_rows = ceil(num_filters / num_cols)

    # 创建子图：大小固定为 280x280 像素（2.8英寸 * 100dpi）
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(2.8, 2.8), dpi=100)
    axes = np.array(axes).reshape(-1)  # 扁平化，方便索引
    for ax in axes:
        ax.axis("off")

    for i in range(num_filters):
        axes[i].imshow(fmap[i], cmap='viridis')
        axes[i].axis("off")

    plt.tight_layout(pad=0.1)

    if save_path is None:
        save_path = f"vis/{layer_name}_feature_maps.png"
    plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0.05)
    plt.close()

    # 读取为 RGB 图像
    img = imageio.v3.imread(save_path)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    # 强制 resize 为 280x280，防止 bbox_inches 裁切影响尺寸
    img = np.array(Image.fromarray(img).resize((280, 280)))

    return save_path, img

def pad_to_same_size(img, target_shape):
    """Pad img (H, W, 3) to match target shape (H, W)"""
    h, w = img.shape[:2]
    H, W = target_shape
    padded = np.ones((H, W, 3), dtype=np.uint8) * 255  # white background
    padded[:h, :w, :] = img
    return padded

def save_feature_map_gif(frame1, frame2, gif_path="vis/feature_transition.gif"):
    H = max(frame1.shape[0], frame2.shape[0])
    W = max(frame1.shape[1], frame2.shape[1])
    frame1 = pad_to_same_size(frame1, (H, W))
    frame2 = pad_to_same_size(frame2, (H, W))
    imageio.mimsave(gif_path, [frame1, frame2], duration=2000.0) 

    return gif_path

def plot_output_logits(output_tensor, save_path="vis/output_logits.png"):
    logits = output_tensor[0].cpu().numpy()  # shape: [10]
    plt.figure(figsize=(8, 4))
    plt.imshow(logits[np.newaxis, :], cmap='gray', aspect='auto')
    plt.yticks([])
    plt.xticks(ticks=range(len(logits)), labels=[str(i) for i in range(len(logits))])
    plt.title("Model Output (Logits)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path
import os
import math
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn

def plot_avg_kernels_to_image(weight_tensor):
    # weight_tensor shape: (out_channels, in_channels, kH, kW)
    avg_kernels = weight_tensor.mean(dim=1).cpu().numpy()  # (out_channels, kH, kW)
    out_channels, kH, kW = avg_kernels.shape

    fig, axes = plt.subplots(1, out_channels, figsize=(out_channels*1.5, 1.5))
    if out_channels == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(avg_kernels[i], cmap='viridis')  # 彩色映射
        ax.axis('off')
    plt.tight_layout()

    # 转成PIL图像
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    return img


def plot_avg_kernels_to_image(weight_tensor):
    # weight_tensor shape: (out_channels, in_channels, kH, kW)
    avg_kernels = weight_tensor.mean(dim=1).cpu().numpy()  # (out_channels, kH, kW)
    out_channels, kH, kW = avg_kernels.shape

    num_cols = 8
    num_rows = (out_channels + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 1.5, num_rows * 1.5))

    for i in range(num_rows * num_cols):
        ax = axes.flat[i]
        if i < out_channels:
            ax.imshow(avg_kernels[i], cmap='viridis')  # 彩色映射
        ax.axis('off')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    return img

def plot_linear_weights_to_image(weight_tensor):
    w = weight_tensor.cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(w, cmap='plasma', aspect='auto')  # 彩色映射
    ax.axis('off')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    return img

def visualize_model_weights(model, save_path='vis/training_progress.png'):
    """
    绘制训练进度图，整合：
    - 卷积层平均卷积核图
    - 线性层权重图
    - Loss曲线
    自适应排列，大小固定 280x280
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    imgs = []

    # 遍历模型各层，生成卷积核和线性权重图
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            w = module.weight.data.clone()
            print(f"绘制卷积核图: {name}")
            imgs.append(plot_avg_kernels_to_image(w))
        elif isinstance(module, nn.Linear):
            w = module.weight.data.clone()
            print(f"绘制线性权重图: {name}")
            imgs.append(plot_linear_weights_to_image(w))

    # 添加 Loss 曲线图

    n = len(imgs)
    if n == 0:
        print("无图像内容可绘制")
        return None

    # 计算行列数，尽量接近正方形
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    # 子图大小
    sub_w = 280 // cols
    sub_h = 280 // rows

    # 新建白色背景大图
    combined_img = Image.new('RGB', (280, 280), color=(255, 255, 255))

    for idx, img in enumerate(imgs):
        # 缩放子图
        img = img.resize((sub_w, sub_h), Image.LANCZOS)
        r = idx // cols
        c = idx % cols
        combined_img.paste(img, (c * sub_w, r * sub_h))

    combined_img.save(save_path, dpi=(100,100))
    print(f"✅ 保存训练进度图到 {save_path}")
    return save_path


def visualize_training_progress(model, loss_list, save_dir="vis"):
    os.makedirs(save_dir, exist_ok=True)

    # 绘制 loss 曲线图
    plt.figure(figsize=(3.5, 3.5))  # 280x280像素左右大小（dpi=80，3.5英寸*80=280）
    plt.plot(range(1, len(loss_list)+1), loss_list, marker='o', color='red')
    plt.title("Training Loss")
    plt.xlabel("step")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "training_loss.png")
    plt.savefig(save_path, dpi=80)  # dpi=80*3.5=280像素大小
    plt.close()

    return save_path

if __name__ == '__main__':
    # 初始化模型
    model = CAMCNN()
    model.load_state_dict(torch.load("mnist_cnn_cam.pth", map_location="cpu"))
    model.eval()

    # 准备输入图像
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist = MNIST(".", train=False, download=True, transform=transform)
    img, label = mnist[0]
    input_tensor = img.unsqueeze(0)  # [1, 1, 28, 28]

    # 推理并捕获特征图与输出
    with torch.no_grad():
        output = model(input_tensor)

    # 保存特征图图像并返回 numpy 图像对象
    savepath1,frame1 = plot_feature_map(model.feature_map1, "conv1")
    savepath2,frame2 = plot_feature_map(model.feature_map2, "conv2")

    # 保存动态 GIF
    save_feature_map_gif(frame1, frame2)

    # 可视化输出 logits
    plot_output_logits(output)