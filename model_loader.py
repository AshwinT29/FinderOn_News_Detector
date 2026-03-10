import torch
import torch.nn as nn
from torchvision import models

_model = None

def load_model():
    global _model

    if _model is None:
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load("model_weight.pth", map_location="cpu"))
        model.eval()
        _model = model

    return _model