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
from models.cnn import CNNModel, model_inference, train 
# from feature_visualize import *

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import time


def cnn_ui():
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


    
            