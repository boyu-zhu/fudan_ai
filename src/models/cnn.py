import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)
        self.feature_map1 = None
        self.feature_map2 = None

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        self.feature_map1 = x.detach().cpu()
        x = self.pool(F.relu(self.conv2(x)))
        self.feature_map2 = x.detach().cpu()
        gap = self.gap(x).view(x.size(0), -1)
        out = self.fc(gap)
        return out

def train(epochs=5, batch_size=64, lr=1e-3):
    model = CNNModel()
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