import torch
import torch.nn as nn
import torchvision.models as models

def load_model():
    # Load pretrained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Replace final layer for 2 classes
    model.fc = nn.Linear(model.fc.in_features, 2)

    # If you have trained weights, load here
    try:
        model.load_state_dict(torch.load("model_weight.pth", map_location="cpu"))
        print("Custom model weights loaded.")
    except:
        print("Using pretrained ImageNet weights (no custom weights found).")

    model.eval()
    return model