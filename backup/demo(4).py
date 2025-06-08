import gradio as gr
import numpy as np
import tempfile
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体（如黑体）
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号
from sklearn.datasets import make_blobs, make_classification
from sklearn.svm import SVC
from PIL import Image, ImageSequence
import io


# -----------------------------
# KMeans 可视化 + 动画保存
# -----------------------------
def kmeans_step_visualization(X, n_clusters=3, max_iter=10):
    images = []
    rng = np.random.RandomState(42)
    centers = X[rng.choice(len(X), n_clusters, replace=False)]

    for i in range(max_iter):
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), axis=1)

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

        new_centers = np.array([
            X[labels == j].mean(axis=0) if np.any(labels == j) else centers[j]
            for j in range(n_clusters)
        ])
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    return images


def generate_kmeans(n_samples, n_clusters):
    X, _ = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=2, random_state=42)
    frames = kmeans_step_visualization(X, n_clusters)

    # 保存为临时 GIF 文件
    gif_path = tempfile.NamedTemporaryFile(delete=False, suffix=".gif").name
    frames[0].save(gif_path, format='GIF', save_all=True, append_images=frames[1:], duration=600, loop=0)

    return frames, gif_path  # ✅ 现在返回字符串路径


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

        model = SVC(kernel='linear')
        model.fit(X_step, y_step)

        fig, ax = plt.subplots()
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k', s=60)

        # 决策边界
        w = model.coef_[0]
        b = model.intercept_[0]
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
def main_process(algorithm, samples, centers):
    if algorithm == "KMeans":
        frames, gif_path = generate_kmeans(samples, centers)
    else:
        frames = plot_svm_step_visualization(samples)
        gif_path = tempfile.NamedTemporaryFile(delete=False, suffix=".gif").name
        frames[0].save(gif_path, format='GIF', save_all=True, append_images=frames[1:], duration=600, loop=0)
    return frames, gif_path


with gr.Blocks() as demo:
    gr.Markdown("## VisualFlow")
    # gr.Markdown("### KMeans")

    # algo = "KMeans"
    with gr.Row():
        algo = gr.Dropdown(choices=["KMeans", "SVM"], value="KMeans", label="选择算法")
        sample_slider = gr.Slider(50, 500, step=10, value=200, label="样本数")
        center_slider = gr.Slider(1, 10, step=1, value=3, label="聚类中心数（仅 KMeans）")
    start_btn = gr.Button("开始可视化")

    with gr.Row():
        gallery = gr.Gallery(label="可视化过程", columns=2, height="auto", show_label=True)
        gif_view = gr.Image(type="filepath", label="GIF 动画预览")

    def toggle_centers(algorithm):
        return gr.update(visible=(algorithm == "KMeans"))

    algo.change(toggle_centers, inputs=algo, outputs=center_slider)
    start_btn.click(
        fn=main_process,
        inputs=[algo, sample_slider, center_slider],
        outputs=[gallery, gif_view]
    )
    
demo.launch()
