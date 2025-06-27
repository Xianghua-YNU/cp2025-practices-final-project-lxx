import torch
import numpy as np
from tqdm import tqdm

def extract_features(model, preprocessed_images, device):
    """
    使用预训练模型提取图像特征
    Args:
        model: 预训练模型
        preprocessed_images: 预处理后的图像列表
        device: 计算设备 (cuda/cpu)
    Returns:
        numpy.ndarray: 提取的特征数组
    """
    features = []
    model.eval()
    print("\n正在提取特征:")
    with torch.no_grad():
        for image in tqdm(preprocessed_images, desc="特征提取"):
            image = image.unsqueeze(0).to(device)
            feature = model.forward_features(image)
            features.append(feature.cpu().numpy().flatten())
    return np.array(features)
