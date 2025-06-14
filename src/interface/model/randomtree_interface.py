# utils/random_tree_vis.py

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
from models.randomtree import RandomTreeModel
from sklearn.datasets import make_classification
import gradio as gr

def plot_decision_boundary(tree, X, y, title="Random Tree"):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]

    Z = tree.predict(grid)
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    ax.set_title(title)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def generate_random_tree_vis(n_samples=200, max_depth=4):
    X, y = make_classification(n_samples=n_samples, n_features=2, n_informative=2,
                               n_redundant=0, n_clusters_per_class=1, random_state=42)

    tree = RandomTreeModel(max_depth=max_depth)
    tree.fit(X, y)
    image = plot_decision_boundary(tree, X, y, title=f"Random Tree (depth={max_depth})")
    return image

def random_tree_ui():
    with gr.Row():
        sample_slider = gr.Slider(50, 500, step=10, value=200, label="样本数")
        depth_slider = gr.Slider(1, 10, step=1, value=4, label="最大深度")

    start_btn = gr.Button("开始可视化")
    output_img = gr.Image(type="pil", label="随机树决策边界")

    start_btn.click(
        fn=lambda n, d: generate_random_tree_vis(n_samples=n, max_depth=d),
        inputs=[sample_slider, depth_slider],
        outputs=output_img
    )