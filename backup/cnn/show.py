import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

def visualize_model_weights(model, save_dir='model_weights_vis'):
    os.makedirs(save_dir, exist_ok=True)

    def plot_avg_kernels(weight, layer_name):
        avg_kernels = weight.mean(dim=1)
        out_channels, kH, kW = avg_kernels.shape
        num_cols = 8
        num_rows = (out_channels + num_cols - 1) // num_cols
        plt.figure(figsize=(num_cols * 1.5, num_rows * 1.5))
        for i in range(out_channels):
            kernel = avg_kernels[i, :, :]
            plt.subplot(num_rows, num_cols, i + 1)
            plt.imshow(kernel, cmap='gray')
            plt.axis('off')
        plt.suptitle(f"{layer_name} (Avg over in_channels)", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        filename = f"{save_dir}/{layer_name.replace('.', '_')}_avg_kernels_gray.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"✅ Saved {filename}")

    def plot_matrix(weight, layer_name):
        plt.figure(figsize=(10, 6))
        plt.imshow(weight, cmap='gray', aspect='auto')
        plt.title(f"{layer_name} (Linear Weights)", fontsize=14)
        plt.xlabel('Input Features')
        plt.ylabel('Output Features')
        plt.axis('off')  # 可选：隐藏坐标轴
        plt.tight_layout()
        filename = f"{save_dir}/{layer_name.replace('.', '_')}_weights_gray.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"✅ Saved {filename}")

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            weight = module.weight.data.clone().cpu()
            print(f"Visualizing Conv2d layer '{name}' with shape {weight.shape}")
            plot_avg_kernels(weight, name)
        elif isinstance(module, nn.Linear):
            weight = module.weight.data.clone().cpu()
            print(f"Visualizing Linear layer '{name}' with shape {weight.shape}")
            plot_matrix(weight, name)

if __name__ == '__main__':
    from model import CAMCNN
    model = CAMCNN()
    model.load_state_dict(torch.load('mnist_cnn_cam.pth', map_location='cpu'))
    model.eval()

    visualize_model_weights(model)
