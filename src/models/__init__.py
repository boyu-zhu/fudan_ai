import importlib
from pathlib import Path

# 自动注册所有模型类
_model_classes = {}

for file in Path(__file__).parent.glob("*.py"):
    if file.name != "__init__.py":
        module_name = f"models.{file.stem}"
        module = importlib.import_module(module_name)
        for name, cls in module.__dict__.items():
            if isinstance(cls, type) and name.endswith("Model"):
                _model_classes[file.stem] = cls

# 提供统一访问接口
def get_model(model_name):
    return _model_classes.get(model_name)

print(f"Available models: {list(_model_classes.keys())}")