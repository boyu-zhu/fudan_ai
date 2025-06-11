# 基础库
import json
import os
import re
import io

# 数据处理和可视化
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from PIL import Image
from tqdm import tqdm

# PyTorch相关
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 机器学习数据集
from sklearn.datasets import make_circles
trained_model=None


class MLPModel(nn.Module):
    def __init__(self, layers):
        super(MLPModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = layers
        module = []
        input_dim = layers[0]
        for n in layers[1:-1]:
            module.append(nn.Linear(input_dim, n))
            module.append(nn.ReLU())
            input_dim = n
        module.append(nn.Linear(input_dim, layers[-1]))
        self.net = nn.Sequential(*module).to(self.device)
        print(f"Using device: {self.device}")

    def forward(self, x):
        return self.net(x)

    def get_device(self):
        return self.device
    
def load_circle_dataset():
    X, y = make_circles(n_samples=1000, noise=0.1, factor=0.5, random_state=42)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    return X, y


def save_weights_to_json(model, epoch, layer_sizes):
    """将模型权重保存到JSON文件"""
    weights_dict = {}
    weight_idx = 0
    
    for i, layer in enumerate(model.net):
        if isinstance(layer, nn.Linear):
            weights = layer.weight.data.cpu().numpy().tolist()
            biases = layer.bias.data.cpu().numpy().tolist()
            weights_dict[f'layer_{weight_idx}'] = {
                'weights': weights,
                'biases': biases,
                'shape': (layer_sizes[weight_idx], layer_sizes[weight_idx+1])
            }
            weight_idx += 1
    
    os.makedirs('weights', exist_ok=True)
    filename = f'weights/epoch_{epoch+1}.json'
    with open(filename, 'w') as f:
        json.dump(weights_dict, f, indent=2)

def train_with_activation_visual(dataset_name, layer_config, epochs, lr):
    weight_dir = "weights"
    if os.path.exists(weight_dir):
        for file in os.listdir(weight_dir):
            file_path = os.path.join(weight_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(weight_dir)
    transform = transforms.ToTensor()
    layer_config_=None
    layer_sizes=None
    if dataset_name == "MNIST":
        dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        layer_config_= [28*28]+[int(n) for n in layer_config.split(',')]+[10]
        layer_sizes = [28*28]+[int(n) for n in layer_config.split(',')] + [10]
    elif dataset_name == "Circle":
        X, y = load_circle_dataset()
        dataset = torch.utils.data.TensorDataset(X, y)
        layer_config_=[2]+[int(n) for n in layer_config.split(',')]+[2]
        layer_sizes = layer_config_
    else:
        return "Unsupported dataset", None, []

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = MLPModel(layer_config_)
    device = next(model.parameters()).device
    
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    losses = []
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        epoch_loss = 0
        model.train()
        
        for i, (data, target) in enumerate(dataloader):
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        save_weights_to_json(model, epoch, layer_sizes)
    global trained_model
    trained_model=model
    torch.save(model.state_dict(), f"{dataset_name}_saved_model.pth")
    # 绘制损失曲线
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), losses, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Curve', fontsize=14, pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=120, bbox_inches='tight')
    plt.close()
    loss_img = Image.open("loss_curve.png")

    return loss_img


def get_prediction(model, sketch_tensor):
    """获取手写输入的预测结果"""
    model.eval()
    with torch.no_grad():
        sketch_flat = sketch_tensor.view(1, -1)
        output = model(sketch_flat)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        pred_class = torch.argmax(probabilities).item()
        return pred_class, probabilities.squeeze().cpu().numpy()

def create_probability_plot(probabilities, pred_class):
    """创建预测概率分布图"""
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(10), probabilities, color='skyblue')
    bars[pred_class].set_color('orange')  # 高亮预测结果
    ax.set_xticks(range(10))
    ax.set_ylim(0, 1)
    ax.set_title('Prediction Probabilities')
    ax.set_xlabel('Digit Class')
    ax.set_ylabel('Probability')
    plt.tight_layout()
    return fig


def preprocess_sketch(editor_data):
    """处理ImageEditor输出"""
    if editor_data is None:
        return None
    # 提取合成图像（包含所有编辑痕迹）
    composite = editor_data["composite"]
    # composite = composite.convert("L")
    # 调整尺寸并归一化
    img = composite.copy()
    
    img_resized = img.resize((28, 28), Image.LANCZOS)
    img_resized = img_resized.convert('L')
    img_array = np.array(img_resized)
    img_array = img_array / 255.0
    tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)
    return 1 - tensor  # 颜色反转

def visualize_sketch_activation(model, layer_sizes, sketch_tensor):
    """可视化手写输入的激活"""
    model.eval()
    with torch.no_grad():
        # 展平输入
        sketch_flat = sketch_tensor.view(1, -1)
        # 生成激活帧
        frames = generate_activation_frames(model.cpu(), layer_sizes, sketch_flat.cpu())
    return frames

def process_sketch_input(sketch_fps,editor_data, layers):
    print("process_sketch_input")
    layer_sizes= [28*28]+[int(n) for n in layers.split(',')] + [10]
    model = MLPModel(layer_sizes)
    model.load_state_dict(torch.load("MNIST_saved_model.pth"))
    if editor_data and model is not None:
        sketch_tensor = preprocess_sketch(editor_data)
        print("preprocess_sketch finished")
        frames = visualize_sketch_activation(model, layer_sizes, sketch_tensor)
        print("visualize_sketch_activation finished")
        pred_class, probs = get_prediction(model, sketch_tensor)
        pred_result = {str(i): float(probs[i]) for i in range(10)}
        plot = create_probability_plot(probs, pred_class)
        print("create_probability_plot finished")
        for j, frame in enumerate(frames):
            frame.save(f"epoch_frames/sketch_layer{j}.jpg", quality=20)
        
        duration = int(1000 / sketch_fps)  # 每帧持续时间（毫秒）
        folder_path="epoch_frames"
        # 匹配所有 epoch/layer 图像
        pattern = re.compile(rf'sketch_layer(\d+)\.jpg')

        matched_files = []
        for f in os.listdir(folder_path):
            match = pattern.match(f)
            if match:
                layer = int(match.group(1))
                matched_files.append((layer, f))

        # 按 epoch 然后 layer 排序
        matched_files.sort(key=lambda x: (x[0]))

        # 加载图像
        images = [Image.open(os.path.join(folder_path, f[1])) for f in matched_files]

        # 输出 gif 路径
        gif_path = os.path.join(folder_path, f"sketch_animation.gif")

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

        return gif_path, pred_result, plot
    return "", {}, None


def play_saved_frames_as_gif(dataset_name, sample_index=0, fps=10):
    # 输出 gif 路径
    folder_path="epoch_frames"
    gif_path = os.path.join(folder_path, f"{dataset_name}_sample{sample_index}_animation.gif")
    return gif_path


def load_dataset_example(dataset_name, sample_index=0):
    sample_index = int(sample_index)

    if dataset_name == "MNIST":
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        image, _ = dataset[sample_index]
        image = transforms.ToPILImage()(image)

    elif dataset_name == "Circle":
        # 生成数据
        X, y = make_circles(n_samples=1000, noise=0.1, factor=0.5, random_state=42)

        # 创建图像
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        ax.set_facecolor("white")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.axhline(0, color='black', linewidth=1)
        ax.axvline(0, color='black', linewidth=1)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")

        # 所有点
        scatter = ax.scatter(X[:, 0], X[:, 1], c=['red' if label == 1 else 'blue' for label in y], s=10, alpha=0.5)

        # 高亮指定点
        xi, yi = X[sample_index]
        ax.scatter([xi], [yi], c='white', edgecolors='black', s=100, linewidths=2)
        ax.text(xi + 0.05, yi + 0.05, f"({xi:.2f}, {yi:.2f})", color="black", fontsize=8)

        # 转换为 PIL Image
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        plt.close(fig)
    else:
        image = Image.new("L", (28, 28), 255)

    return image


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