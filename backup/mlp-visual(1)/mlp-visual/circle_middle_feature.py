import io
import os
import re
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from draw_structure import generate_activation_frames
from mlp_task import MLPModel
from PIL import Image
def generate_2d_grid(n_points=1000):
    x = np.linspace(-1, 1, n_points)
    y = np.linspace(-1, 1, n_points)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    return torch.tensor(grid_points, dtype=torch.float32), xx, yy

def get_positive_activation_masks(model, layer_sizes, layer_idx, neuron_idx):
    # 构造输入点
    
    inputs, xx, yy = generate_2d_grid()

    activations = []
    if layer_idx==0:
        act_values = inputs[:, neuron_idx].detach().cpu().numpy().reshape(xx.shape)
        mask = act_values > 0
        return mask, xx, yy

    else:
        layer_idx=layer_idx-1
        def hook(module, input, output):
            # 只记录指定神经元的激活值
            act = output[:, neuron_idx].detach().cpu().numpy()
            activations.append(act)

        # 注册 hook
        layer_count = 0
        handle = None
        for layer in model.net:
            if isinstance(layer, nn.Linear):
                if layer_count == layer_idx:
                    handle = layer.register_forward_hook(hook)
                    break
                layer_count += 1

        with torch.no_grad():
            model.eval()
            device = next(model.parameters()).device  # 获取模型所在设备
            inputs = inputs.to(device)               # 把 inputs 移到同一设备
            model(inputs)

        handle.remove()

        act_values = activations[0].reshape(xx.shape)
        mask = act_values > 0
        return mask, xx, yy

def plot_activation_region(mask, xx, yy, layer_idx, neuron_idx):
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, mask, levels=[0.5, 1], colors=['#1f77b4'], alpha=0.5)
    ax.set_title(f"Layer {layer_idx}, Neuron {neuron_idx} Positive Activation Region")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.tight_layout()
    return fig

def show_neuron_activation_region(layer_structure, layer_idx, neuron_idx):
    layer_sizes = [2] + [int(n) for n in layer_structure.split(',')] + [2]
    model = MLPModel(layer_sizes)
    model.load_state_dict(torch.load("Circle_saved_model.pth"))
    mask, xx, yy = get_positive_activation_masks(model, layer_sizes, layer_idx, neuron_idx)
    fig = plot_activation_region(mask, xx, yy, layer_idx, neuron_idx)
    fig_path = f"circle_middle/activation_region_layer{layer_idx}_neuron{neuron_idx}.jpg"
    fig.savefig(fig_path)
    return fig_path

def process_circle_input(x1, x2, layers):
    print("process_circle_input")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_tensor = torch.tensor([[x1, x2]], dtype=torch.float32).to(device)

    # 构建网络结构
    layer_sizes = [2] + [int(n) for n in layers.split(',')] + [2]
    model = MLPModel(layer_sizes).to(device)

    model_path = "CIRCLE_saved_model.pth"
    if not os.path.exists(model_path):
        print("模型文件不存在")
        return "", {}, None

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 前向传播
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        pred_class = torch.argmax(probabilities).item()
        probs_np = probabilities.squeeze().cpu().numpy()

    # 激活可视化帧生成
    frames = generate_activation_frames(model.cpu(), layer_sizes, input_tensor.cpu())

    for j, frame in enumerate(frames):
        frame.save(f"epoch_frames/circle_layer{j}.jpg", quality=80)

    # 生成 GIF
    duration = int(1000 / 10)
    folder_path = "epoch_frames"
    pattern = re.compile(r'circle_layer(\d+)\.jpg')

    matched_files = []
    for f in os.listdir(folder_path):
        match = pattern.match(f)
        if match:
            matched_files.append((int(match.group(1)), f))
    
    matched_files.sort()
    images = [Image.open(os.path.join(folder_path, f[1])) for f in matched_files]

    gif_path = os.path.join(folder_path, "circle_animation.gif")
    if images:
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )
        print(f"Saved GIF: {gif_path}")
    else:
        print("No matching images found.")

    # 概率分布图
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(range(2), probs_np, color='skyblue')
    bars[pred_class].set_color('orange')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Class 0', 'Class 1'])
    ax.set_ylim(0, 1)
    ax.set_title("Circle Prediction Probabilities")
    ax.set_xlabel("Class")
    ax.set_ylabel("Probability")
    plt.tight_layout()

    return gif_path, {str(i): float(probs_np[i]) for i in range(2)}, fig

def generate_all_neuron_activation_grid(layers_str):
    # 构造完整结构（输入+隐藏+输出）
    layer_sizes = [2] + [int(n) for n in layers_str.split(",")] + [2]
    model = MLPModel(layer_sizes)
    model.load_state_dict(torch.load("Circle_saved_model.pth"))
    model.eval()

    num_layers = len(layer_sizes)
    max_neurons = max(layer_sizes)

    fig, axes = plt.subplots(max_neurons, num_layers, figsize=(num_layers * 2.5, max_neurons * 2.5))
    if num_layers == 1:
        axes = np.expand_dims(axes, axis=1)
    elif max_neurons == 1:
        axes = np.expand_dims(axes, axis=0)

    centers = {}
    xx_global, yy_global = None, None

    for l in range(num_layers):
        n_neurons = layer_sizes[l]
        for n in range(n_neurons):
            row = n
            col = l
            ax = axes[row, col]

            mask, xx, yy = get_positive_activation_masks(model, layer_sizes, l, n)
            if xx_global is None:
                xx_global, yy_global = xx, yy
            ax.contourf(xx, yy, mask, levels=[0.5, 1], colors=['#1f77b4'], alpha=0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"L{l}-N{n}", fontsize=8)
            centers[(l, n)] = ((col + 0.5), max_neurons - row - 0.5)

        for r in range(n_neurons, max_neurons):
            axes[r, col].axis("off")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img