
import json
import os
import re
from matplotlib.collections import PatchCollection
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.patches import Circle
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from mlp_task import MLPModel,load_circle_dataset

# 可视化一个样本在模型中的逐层激活（不包含输入层）
def generate_activation_frames(model, layer_sizes, sample_tensor):
    model.eval()
    with torch.no_grad():
        current = sample_tensor
        activations = []
        
        input_act = sample_tensor.squeeze().numpy()
        input_act_norm = (input_act - input_act.min()) / (input_act.max() - input_act.min() + 1e-5)
        activations.append(input_act_norm.tolist())

        for idx, layer in enumerate(model.net):
            current = layer(current)
            is_last_layer = (idx == len(model.net) - 1)
            if isinstance(layer, nn.ReLU) or is_last_layer:
                act = current.squeeze().numpy()
                act_norm = (act - act.min()) / (act.max() - act.min() + 1e-5)
                activations.append(act_norm.tolist())

    # 绘制每一层的激活图
    frames = []
    for step in range(len(activations)):
        img = draw_activations_one_layer(layer_sizes, activations, step, model)
        frames.append(img)

    return frames

# 动态计算可视化参数
def calculate_visual_params(layer_sizes):
    # n_layers = len(layer_sizes)
    fig_width=70
    fig_height=70
    layer_spacing=1.5
    neuron_radius=0.3
    
    return fig_width, fig_height, layer_spacing, neuron_radius



def get_layer_weights(model, layer_idx):
    """获取特定层的权重矩阵"""
    weight_idx = 0
    for i, layer in enumerate(model.net):
        if isinstance(layer, nn.Linear):
            if weight_idx == layer_idx:
                return layer.weight.data.cpu().numpy()
            weight_idx += 1
    return None

def draw_weighted_connection(ax, x1, y1, x2, y2, weight, max_weight, neuron_radius):
    """绘制带权重的插值曲线连接"""
    # 标准化权重到[0,1]范围
    norm_weight = abs(weight) / (max_weight + 1e-5)
    
    # 控制点生成曲线
    control_x = (x1 + x2) / 2
    control_y1 = y1 + (y2 - y1) * 0.3 + np.random.normal(0, 0.1) * neuron_radius
    control_y2 = y1 + (y2 - y1) * 0.7 + np.random.normal(0, 0.1) * neuron_radius
    
    # 贝塞尔曲线插值
    t = np.linspace(0, 1, 10)
    x = (1-t)**3 * x1 + 3*(1-t)**2*t * control_x + 3*(1-t)*t**2 * control_x + t**3 * x2
    y = (1-t)**3 * y1 + 3*(1-t)**2*t * control_y1 + 3*(1-t)*t**2 * control_y2 + t**3 * y2
    
    # 根据权重设置线宽和透明度
    linewidth = 8 + norm_weight *15  # 线宽范围0.1-3.1
    alpha =np.sqrt(norm_weight*0.8)    # 透明度范围0.1-1.0
    color = (0.2, 0.4, 1.0, alpha)    # 深色半透明
    
    ax.plot(x, y, color=color, linewidth=linewidth, solid_capstyle='round')

# 绘制某一层的神经元激活
def draw_activations_one_layer(layer_sizes, activations, step, model):
    # 计算动态参数
    fig_width, fig_height, layer_spacing, neuron_radius = calculate_visual_params(layer_sizes)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_aspect('equal')
    ax.axis('off')

    patches = []
    colors = []

    positions = []
    layer_infos = []
    x_offsets = [0] * len(layer_sizes)  # 每层的额外横向偏移

    # 预计算第一层横向宽度（如果平铺）
    first_layer_width = 0
    if layer_sizes[0] > 20:
        n = int(np.ceil(np.sqrt(layer_sizes[0])))
        rows = cols = n
        x_spacing = neuron_radius * 1.5
        first_layer_width = cols * x_spacing + rows * neuron_radius * 0.6  # 平行四边形总宽度

        # 给后续层加上偏移
        for i in range(1, len(layer_sizes)):
            x_offsets[i] = first_layer_width + neuron_radius * 2  # 留出一点 margin

    # 第二步：计算每层位置
    for i, n_neurons in enumerate(layer_sizes):
        layer_pos = []
        x_base = i * layer_spacing + x_offsets[i] + 2

        if n_neurons > 20 and i == 0:  # 仅第一层使用平铺
            n = int(np.ceil(np.sqrt(n_neurons)))
            x_spacing = neuron_radius * 1.5
            y_spacing = neuron_radius * 1.8
            for idx in range(n_neurons):
                row = idx // n
                col = idx % n
                x = x_base + col * x_spacing + (row * neuron_radius * 0.6)
                y = -row * y_spacing
                layer_pos.append((x, y))
        else:
            # 居中竖直排列
            y_start = -(n_neurons - 1) * neuron_radius * 2.5 / 2
            for j in range(n_neurons):
                y = y_start + j * neuron_radius * 2.5
                x = x_base
                layer_pos.append((x, y))

        positions.append(layer_pos)


    # 第二次遍历：绘制神经元和设置颜色
    for i, layer_pos in enumerate(positions):
        for j, (x, y) in enumerate(layer_pos):
            circ = Circle((x, y), neuron_radius)
            patches.append(circ)
            if i == step:
                act = activations[i][j]
                color = (1.0, 1.0 - act, 1.0 - act)
            else:
                color = (0.8, 0.8, 0.8)
            colors.append(color)

    # 绘制连接线
        # 绘制连接线（只取最大权重前 top_k 条）
    # 绘制连接线（根据连接数量判断是否裁剪）
    for i in range(len(layer_sizes) - 1):
        weights = get_layer_weights(model, i)  # shape: [out_dim, in_dim]
        print(weights)
        max_weight = np.max(np.abs(weights))
        layer1_pos = positions[i]
        layer2_pos = positions[i + 1]

        total_connections = len(layer1_pos) * len(layer2_pos)
        threshold = 400  # 超过这个连接数就开始裁剪
        top_k = 30      # 裁剪时保留的最大连接数

        if total_connections <= threshold:
            # 连接数较少，全部绘制
            for a, (x1, y1) in enumerate(layer1_pos):
                for b, (x2, y2) in enumerate(layer2_pos):
                    draw_weighted_connection(ax, x1, y1, x2, y2, weights[b][a], max_weight, neuron_radius)
        else:
            # 连接数过多，裁剪，仅绘制 top_k 最大连接
            abs_weights = np.abs(weights)
            flat_indices = np.argsort(abs_weights.flatten())[::-1]  # 从大到小排序
            selected_indices = flat_indices[:top_k]
            
            for idx in selected_indices:
                b, a = np.unravel_index(idx, weights.shape)
                x1, y1 = layer1_pos[a]
                x2, y2 = layer2_pos[b]
                draw_weighted_connection(ax, x1, y1, x2, y2, weights[b][a], max_weight, neuron_radius)


    # 绘制神经元圆圈
    collection = PatchCollection(patches, facecolors=colors, edgecolors='black', 
                                 linewidths=1)
    ax.add_collection(collection)
    
    all_x = [x for layer in positions for (x, _) in layer]
    all_y = [y for layer in positions for (_, y) in layer]
    margin = neuron_radius * 3
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    # 渲染图像
    fig.canvas.draw()
    img_data = np.array(fig.canvas.renderer.buffer_rgba())
    img = Image.fromarray(img_data[..., :3])
    plt.close(fig)
    return img

def saved_frames_as_gif(dataset_name, sample_index=0, fps=10):
    """
    将所有形如 {dataset_name}_sample{sample_index}_epochX_layerY.jpg 的图像
    按 epoch 和 layer 顺序合成 GIF 并返回路径。
    """
    folder_path = 'epoch_frames'
    duration = int(1000 / fps)  # 每帧持续时间（毫秒）

    # 匹配所有 epoch/layer 图像
    pattern = re.compile(rf'{dataset_name}_sample{sample_index}_epoch(\d+)_layer(\d+)\.jpg')

    matched_files = []
    for f in os.listdir(folder_path):
        match = pattern.match(f)
        if match:
            epoch = int(match.group(1))
            layer = int(match.group(2))
            matched_files.append((epoch, layer, f))

    # 按 epoch 然后 layer 排序
    matched_files.sort(key=lambda x: (x[0], x[1]))

    # 加载图像
    images = [Image.open(os.path.join(folder_path, f[2])) for f in matched_files]

    # 输出 gif 路径
    gif_path = os.path.join(folder_path, f"{dataset_name}_sample{sample_index}_animation.gif")

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

    return gif_path
def generate_all_epoch_frames(layer_config, dataset_name, num_samples=1):
    frame_dir = "epoch_frames"
    if os.path.exists(frame_dir):
        for file in os.listdir(frame_dir):
            file_path = os.path.join(frame_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(frame_dir)
    layer_sizes = [28 * 28] + [int(n) for n in layer_config.split(',')] + [10] if dataset_name == "MNIST" else \
                  [2] + [int(n) for n in layer_config.split(',')] + [2]
    
    transform = transforms.ToTensor()
    if dataset_name == "MNIST":
        dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        samples = [dataset[i][0].view(1, -1) for i in range(num_samples)]
    else:
        X, y = load_circle_dataset()
        samples = [X[i].view(1, -1) for i in range(num_samples)]

    frame_dir = "epoch_frames"
    for file in sorted(os.listdir("weights")):
        if not file.endswith(".json"):
            continue
        epoch = file.split('_')[1].split('.')[0]
        model = MLPModel(layer_sizes)
        with open(os.path.join("weights", file), 'r') as f:
            weights = json.load(f)
        idx = 0
        for layer in model.net:
            if isinstance(layer, nn.Linear):
                layer.weight.data = torch.tensor(weights[f'layer_{idx}']['weights'])
                layer.bias.data = torch.tensor(weights[f'layer_{idx}']['biases'])
                idx += 1
        for i, sample in enumerate(samples):
            frames = generate_activation_frames(model, layer_sizes, sample)
            for j, frame in enumerate(frames):
                frame.save(f"{frame_dir}/{dataset_name}_sample{i}_epoch{epoch}_layer{j}.jpg", quality=20)
    
    for i, sample in enumerate(samples):
        saved_frames_as_gif(dataset_name,i,fps=5)

    return f"已保存 {num_samples} 个样本的所有 epoch 激活帧。"

