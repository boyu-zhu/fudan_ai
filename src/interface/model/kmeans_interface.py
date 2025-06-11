import gradio as gr
import numpy as np
import tempfile
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_classification
from models.kmeans import KMeansModel 
from PIL import Image, ImageSequence
import io


def main_process(algorithm, samples, centers):
    frames, gif_path = generate_kmeans(samples, centers)
    return frames, gif_path

def kmeans_ui():
    with gr.Row():
        sample_slider = gr.Slider(50, 500, step=10, value=210, label="样本数")
        center_slider = gr.Slider(1, 10, step=1, value=3, label="聚类中心数")

    start_btn = gr.Button("开始可视化")

    with gr.Row():
        gallery = gr.Gallery(label="可视化过程", columns=2, height="auto", show_label=True)
        gif_view = gr.Image(type="filepath", label="GIF 动画预览")

    start_btn.click(
        fn=lambda samples, centers: main_process("KMeans", samples, centers),
        inputs=[sample_slider, center_slider],
        outputs=[gallery, gif_view]
    )


def kmeans_step_visualization(X, n_clusters=3, max_iter=10):
    kmeans = KMeansModel(n_clusters=n_clusters, max_iter=max_iter)
    history = kmeans.fit(X)
    images = []

    for i, (centers, labels) in enumerate(history):
        fig, ax = plt.subplots()
        for j in range(n_clusters):
            ax.scatter(X[labels == j, 0], X[labels == j, 1], label=f"Cluster {j}", alpha=0.6)
            ax.scatter(*centers[j], c='black', marker='x', s=100)
        ax.set_title(f"KMeans Step {i}")
        ax.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        images.append(Image.open(buf).convert("RGB"))

    return images

def generate_kmeans(n_samples, n_clusters):
    X, _ = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=2, random_state=42)
    frames = kmeans_step_visualization(X, n_clusters)

    # 保存为临时 GIF 文件
    gif_path = tempfile.NamedTemporaryFile(delete=False, suffix=".gif").name
    frames[0].save(gif_path, format='GIF', save_all=True, append_images=frames[1:], duration=600, loop=0)

    return frames, gif_path  # ✅ 现在返回字符串路径


