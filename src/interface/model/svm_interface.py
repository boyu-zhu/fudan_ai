import gradio as gr
import numpy as np
import tempfile
import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体（如黑体）
# plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号
from sklearn.datasets import make_blobs, make_classification
# from sklearn.svm import SVC
from models.svm import SVMModel 
from PIL import Image, ImageSequence
import io


# -----------------------------
# SVM 可视化
# -----------------------------
def plot_svm_step_visualization(n_samples, n_steps=10):
    X, y = make_classification(
        n_samples=n_samples, n_features=2, n_informative=2, n_redundant=0,
        n_clusters_per_class=1, flip_y=0, class_sep=2.0, random_state=42
    )

    # 计算全局坐标轴范围
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    indices = np.arange(n_samples)
    step_size = max(n_samples // n_steps, 1)
    images = []

    for i in range(1, n_steps + 1):
        selected = indices[:i * step_size]
        X_step = X[selected]
        y_step = y[selected]

        model = SVMModel()
        model.fit(X_step, y_step)

        fig, ax = plt.subplots()
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k', s=60)

        # 决策边界
        w = model.w
        b = model.b
        x_vals = np.linspace(x_min, x_max, 100)
        y_vals = -(w[0] * x_vals + b) / w[1]
        ax.plot(x_vals, y_vals, 'k--', label="Decision boundary")

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"SVM Step {i}")
        ax.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        images.append(Image.open(buf).convert("RGB"))

    return images




# -----------------------------
# Gradio 界面
# -----------------------------
def main_process(samples):
    frames = plot_svm_step_visualization(samples)
    gif_path = tempfile.NamedTemporaryFile(delete=False, suffix=".gif").name
    frames[0].save(gif_path, format='GIF', save_all=True, append_images=frames[1:], duration=600, loop=0)
    return frames, gif_path



def svm_ui():
    start_btn = gr.Button("开始可视化")
    sample_slider = gr.Slider(50, 500, step=10, value=200, label="样本数")
    with gr.Row():
        gallery = gr.Gallery(label="可视化过程", columns=2, height="auto", show_label=True)
        gif_view = gr.Image(type="filepath", label="GIF 动画预览")

    start_btn.click(
        fn=lambda samples: main_process(samples),
        inputs=[sample_slider],
        outputs=[gallery, gif_view]
    )

