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
from model import CAMCNN
from feature_visualize import *

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import time
# --- 可视化工具函数 ---
os.makedirs("vis", exist_ok=True)


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
    model = CAMCNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("mnist_cnn_cam.pth", map_location=device))

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

# --- 训练函数（实时更新） ---

def train(epochs=5, batch_size=64, lr=1e-3):
    model = CAMCNN()
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = MNIST(root='.', train=True, transform=transform, download=True)
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

    torch.save(model.state_dict(), "mnist_cnn_cam.pth")
    print("Model saved as mnist_cnn_cam.pth")




# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("## Visual Flow")
    gr.Markdown("### CNN")
    with gr.Tab("训练可视化"):
        with gr.Row():
            train_log = gr.Textbox(label="训练日志输出", lines=10)
            out_loss = gr.Image(label="训练 Loss 曲线", height=300)
            with gr.Column():
                out_weights = gr.Image(label="权重可视化", height=300)
                
                epoch_slider = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="训练轮数 (Epochs)")
                train_btn = gr.Button("开始训练模型")

                train_btn.click(
                    fn=train,
                    inputs=[epoch_slider],
                    outputs=[out_loss, out_weights, train_log]
                )


    with gr.Tab("过程可视化"):
        with gr.Row():
            sketchpad = gr.Sketchpad(label="灰度数字绘图板", height=280)

            with gr.Column():
                algo = gr.Dropdown(choices=["CNN", ""], value="CNN", label="选择模型")
                pred_text = gr.Textbox(label="🔢 预测结果")
                gif_out = gr.Image(label="🎞️ Conv1 → Conv2 特征图 GIF", type="filepath", height=280)
                logits_out = gr.Image(label="📊 模型输出 logits", type="filepath", height=200)
                btn = gr.Button("开始预测", variant="secondary")
                btn.click(fn=model_inference, inputs=sketchpad, outputs=[pred_text, gif_out, logits_out])
    
            


if __name__ == "__main__":
    demo.launch()
