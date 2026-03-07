with torch.no_grad():
    output = model(tensor)
    confidence = float(output.item())  # value between 0 and 1

if confidence > 0.5:
    result = "Fake"
else:
    result = "Real"

confidence_percent = round(confidence * 100, 2)