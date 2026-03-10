from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import base64
import io

app = FastAPI()

# Load ResNet18
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model_weight.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class ImageRequest(BaseModel):
    image: str

@app.post("/analyze")
async def analyze(request: ImageRequest):

    image_bytes = base64.b64decode(request.image.split(",")[1])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    label = "real" if predicted.item() == 0 else "ai_generated"

    return {
        "prediction": label,
        "confidence": round(confidence.item() * 100, 2)
    }