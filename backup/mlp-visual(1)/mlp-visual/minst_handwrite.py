import os
import re
from matplotlib import pyplot as plt
import numpy as np
import torch
from draw_structure import generate_activation_frames
from PIL import Image
from mlp_task import MLPModel
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