from matplotlib import pyplot as plt
from sklearn.datasets import make_circles
import torch
import torch.nn as nn
import json
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import os
trained_model=None
class MLPModel(nn.Module):
    def __init__(self, layers):
        super(MLPModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = layers
        module = []
        input_dim = layers[0]
        for n in layers[1:-1]:
            module.append(nn.Linear(input_dim, n))
            module.append(nn.ReLU())
            input_dim = n
        module.append(nn.Linear(input_dim, layers[-1]))
        self.net = nn.Sequential(*module).to(self.device)
        print(f"Using device: {self.device}")

    def forward(self, x):
        return self.net(x)

    def get_device(self):
        return self.device

def load_circle_dataset():
    X, y = make_circles(n_samples=1000, noise=0.1, factor=0.5, random_state=42)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    return X, y


def save_weights_to_json(model, epoch, layer_sizes):
    """将模型权重保存到JSON文件"""
    weights_dict = {}
    weight_idx = 0
    
    for i, layer in enumerate(model.net):
        if isinstance(layer, nn.Linear):
            weights = layer.weight.data.cpu().numpy().tolist()
            biases = layer.bias.data.cpu().numpy().tolist()
            weights_dict[f'layer_{weight_idx}'] = {
                'weights': weights,
                'biases': biases,
                'shape': (layer_sizes[weight_idx], layer_sizes[weight_idx+1])
            }
            weight_idx += 1
    
    os.makedirs('weights', exist_ok=True)
    filename = f'weights/epoch_{epoch+1}.json'
    with open(filename, 'w') as f:
        json.dump(weights_dict, f, indent=2)

def train_with_activation_visual(dataset_name, layer_config, epochs, lr):
    weight_dir = "weights"
    if os.path.exists(weight_dir):
        for file in os.listdir(weight_dir):
            file_path = os.path.join(weight_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(weight_dir)
    transform = transforms.ToTensor()
    layer_config_=None
    layer_sizes=None
    if dataset_name == "MNIST":
        dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        layer_config_= [28*28]+[int(n) for n in layer_config.split(',')]+[10]
        layer_sizes = [28*28]+[int(n) for n in layer_config.split(',')] + [10]
    elif dataset_name == "Circle":
        X, y = load_circle_dataset()
        dataset = torch.utils.data.TensorDataset(X, y)
        layer_config_=[2]+[int(n) for n in layer_config.split(',')]+[2]
        layer_sizes = layer_config_
    else:
        return "Unsupported dataset", None, []

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = MLPModel(layer_config_)
    device = next(model.parameters()).device
    
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    losses = []
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        epoch_loss = 0
        model.train()
        
        for i, (data, target) in enumerate(dataloader):
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        save_weights_to_json(model, epoch, layer_sizes)
    global trained_model
    trained_model=model
    torch.save(model.state_dict(), f"{dataset_name}_saved_model.pth")
    # 绘制损失曲线
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), losses, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Curve', fontsize=14, pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=120, bbox_inches='tight')
    plt.close()
    loss_img = Image.open("loss_curve.png")

    return loss_img

