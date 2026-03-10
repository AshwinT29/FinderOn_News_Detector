from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
from torchvision import transforms
import base64
import io
import torch

from model_loader import load_model
from face_detector import detect_faces
from fact_check import analyze_text
from predict_news import classify_news
from gradcam import generate_heatmap

app = FastAPI()

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class ImageRequest(BaseModel):
    image: str

@app.post("/")
async def analyze(request: ImageRequest):

    image_bytes = base64.b64decode(request.image.split(",")[1])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    label = "Real" if predicted.item() == 0 else "AI Generated"

    # Extra modules
    face_info = detect_faces(image)
    text_info = analyze_text(image)
    news_info = classify_news(image)

    return {
        "prediction": label,
        "confidence": round(confidence.item() * 100, 2),
        "face_analysis": face_info,
        "text_analysis": text_info,
        "news_analysis": news_info
    }