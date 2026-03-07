from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import cv2

def generate_gradcam(model, input_tensor, original_img):

    target_layer = model.layer4[-1]

    cam = GradCAM(model=model, target_layers=[target_layer])

    grayscale_cam = cam(input_tensor=input_tensor)[0]

    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    original_img = original_img.astype(np.float32) / 255.0

    visualization = show_cam_on_image(original_img, grayscale_cam, use_rgb=True)
    visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)

    return visualization