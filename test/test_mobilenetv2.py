import sys
import os

# 添加 src 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import torch
from models.modelnet_v2 import mobilenet_v2 

def test_custom_mobilenetv2():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Loading custom MobileNetV2 with pretrained weights...")
    model = mobilenet_v2(weights="DEFAULT").to(device)
    model.eval()

    print("Generating dummy input...")
    x = torch.randn(1, 3, 224, 224).to(device)

    print("Running forward pass...")
    with torch.no_grad():
        output = model(x)

    print(f"Output shape: {output.shape}")
    assert output.shape == (1, 1000), "Output shape is incorrect, expected (1, 1000)"
    print("✅ Model forward pass successful with expected output shape.")

if __name__ == "__main__":
    test_custom_mobilenetv2()
