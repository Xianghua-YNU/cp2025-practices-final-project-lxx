import numpy as np
from PIL import Image
from torchvision import transforms

def preprocess_image(data, target_size=(224, 224)):
    """预处理图像数据"""
    normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    image = Image.fromarray((normalized_data * 255).astype(np.uint8))
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)
