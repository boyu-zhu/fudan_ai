import gradio as gr
from interpret.lime_interpreter import LimeFudan
import numpy as np
from PIL import Image
# import torchvision.models as models
import torchvision.transforms as transforms
import torch
import os
from models.modelnet_v2 import mobilenet_v2 
from typing import Dict, Callable

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = mobilenet_v2(weights="DEFAULT").to(device)
model.eval()


DEFAULT_CLASS_FILE = "data/imagenet_classes.txt"

MODELS: Dict[str, Callable] = {
    "MobileNetV2": lambda: None,
    # 可以在此添加更多模型，例如：
    # "ResNet50": lambda: models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device).eval(),
}


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def load_default_classes():
    """加载默认的类别标签"""
    if os.path.exists(DEFAULT_CLASS_FILE):
        with open(DEFAULT_CLASS_FILE, 'r') as f:
            return {int(i): line.strip() for i, line in enumerate(f.readlines())}
    else:
        # 如果默认文件不存在，返回一个简单的默认类别
        return {i: f"Class {i}" for i in range(1000)}



def lime_ui():
    with gr.Blocks(title="LIME Explanation") as interface:
        gr.Markdown("## LIME Explanation Interface")
        
        with gr.Row():
            # Input components
            with gr.Column():
                model_dropdown = gr.Dropdown(
                    choices=list(MODELS.keys()),
                    value="MobileNetV2",
                    label="Select Model"
                )
                image_input = gr.Image(label="Upload Image", type="numpy", 
                                     value=np.array(Image.open("data/school_bus.png")) if os.path.exists("data/school_bus.png") else None)
                class_file = gr.File(
            label="Upload Class Labels (Optional)",
            value=None,  # 自动加载默认文件
            type="filepath",  # 直接使用文件路径
            interactive=True
        )
                num_samples = gr.Slider(100, 5000, step=100, value=100, label="Number of Samples")
                explain_button = gr.Button("Explain", variant="primary")
            
            # Output components
            with gr.Column():
                original_image = gr.Image(label="Original Image", interactive=False)
                explanation_image = gr.Image(label="Explanation", interactive=False)
                prediction_text = gr.Textbox(label="Prediction", interactive=False)
        
        # Event bindings
        explain_button.click(
            fn=run_lime_explanation,
            inputs=[model_dropdown, image_input, class_file, num_samples],
            outputs=[original_image, explanation_image, prediction_text]
        )
    
    return interface

def run_lime_explanation(_, image: np.ndarray, class_file, num_samples):

    classes = load_default_classes()
    
    # Define classifier function (replace with your actual model)
    def classifier_fn(image):
        image = Image.fromarray(image.astype(np.uint8))
        image = preprocess(image).unsqueeze(0).to(device).float()
        probs = model(image)
        pred_prob, pred_class = torch.max(probs, dim=1)
        pred_class = pred_class.item()  # 转为Python整数
        pred_prob = pred_prob.item()    # 转为Python浮点数
        return probs.detach().squeeze(0).cpu().numpy()
    
    # image = np.array(image)
    # Create and run explainer
    explainer = LimeFudan(
        image=image,
        classes=classes,
        classifier_fn=classifier_fn,
        num_samples=num_samples
    )
    explainer.explain()
    explainer.show_explanation()
    
    # Get results
    original_img = explainer.saved_images['original']
    explanation_img = Image.open(explainer.saved_images['boundary'])
    prediction = explainer.classes.get(explainer.top_label, str(explainer.top_label))
    
    return original_img, explanation_img, prediction