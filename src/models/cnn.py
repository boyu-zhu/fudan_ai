import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import io
import gradio as gr
# from model import CAMCNN
# from models.cnn import CNNModel, model_inference, train 
# from feature_visualize import *

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import time
os.makedirs("log/vis", exist_ok=True)

from math import ceil, sqrt

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

def visualize_training_progress(model, loss_list, save_dir="log/vis"):
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
        save_path = f"log/vis/{layer_name}_feature_maps.png"
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

def save_feature_map_gif(frame1, frame2, gif_path="log/vis/feature_transition.gif"):
    H = max(frame1.shape[0], frame2.shape[0])
    W = max(frame1.shape[1], frame2.shape[1])
    frame1 = pad_to_same_size(frame1, (H, W))
    frame2 = pad_to_same_size(frame2, (H, W))
    imageio.mimsave(gif_path, [frame1, frame2], duration=2000.0) 

    return gif_path

def plot_output_logits(output_tensor, save_path="log/vis/output_logits.png"):
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

def visualize_model_weights(model, save_path='log/vis/training_progress.png'):
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



def train(epochs=5, batch_size=64, lr=1e-3):
    model = CNNModel()
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = MNIST(root='data/', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    step_loss_list = []           # 所有step的loss（不用于画图）
    loss_plot_points = []         # 每100 step画一个点
    log_text = ""

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for step, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val * images.size(0)
            step_loss_list.append(loss_val)

            # 每 100 step 记录一次可视化点
            if step % 100 == 0 or step == len(train_loader):
                loss_plot_points.append(loss_val)

                elapsed = time.time() - start_time
                log_text += f"Epoch {epoch+1}/{epochs} - Step {step}/{len(train_loader)} - Loss: {loss_val:.4f} - Time: {elapsed:.1f}s\n"
                
                training_curve_path = visualize_training_progress(model, loss_plot_points)
                weights_vis_path = visualize_model_weights(model)
                
                yield training_curve_path, weights_vis_path, log_text

        epoch_loss /= len(train_loader.dataset)

    torch.save(model.state_dict(), "ckp/mnist_cnn_cam.pth")
    print("Model saved as ckp/mnist_cnn_cam.pth")

# --- 预处理手绘图像 ---
def preprocess_sketch(sketch_img):
    img_array = sketch_img['composite']
    pil_img = Image.fromarray(img_array.astype(np.uint8)).convert("L")
    pil_img = pil_img.resize((28, 28))
    img_array = np.array(pil_img).astype(np.float32)
    img_array = 255 - img_array  # 反色
    img_array = img_array / 255.0
    img_tensor = (img_array - 0.5) / 0.5
    img_tensor = torch.tensor(img_tensor).unsqueeze(0).unsqueeze(0)
    return img_tensor


# --- 推理函数 ---
def model_inference(sketch_img):
    model = CNNModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("ckp/mnist_cnn_cam.pth", map_location=device))

    input_tensor = preprocess_sketch(sketch_img).to(device)  # <- 转设备
    model.to(device)  # 确保模型也在设备上
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    pred_label = torch.argmax(output, dim=1).item()

    savepath1, frame1 = plot_feature_map(model.feature_map1, "conv1")
    savepath2, frame2 = plot_feature_map(model.feature_map2, "conv2")
    gif_path = save_feature_map_gif(frame1, frame2)
    logits_path = plot_output_logits(output)

    return f"预测数字是：{pred_label}", gif_path, logits_path