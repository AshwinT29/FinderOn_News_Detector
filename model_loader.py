import torch
import torchvision.models as models

def load_model():
    # DO NOT download weights
    model = models.resnet18(weights=None)

    # Change final layer
    model.fc = torch.nn.Linear(model.fc.in_features, 2)

    # Load your trained weights
    model.load_state_dict(torch.load("model_weights.pth", map_location="cpu"))

    model.eval()
    return model