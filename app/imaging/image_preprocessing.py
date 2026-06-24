import torch
from torchvision import transforms
from PIL import Image
import numpy as np

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])

def preprocess_image(image_file):
    """
    تاخد file object وترجع tensor جاهز للموديل
    """
    img = Image.open(image_file).convert('RGB')
    transform = get_transforms()
    tensor = transform(img).unsqueeze(0)  # إضافة batch dimension
    return tensor, img

def tensor_to_numpy(tensor):
    """
    تحويل tensor لـ numpy array للعرض
    """
    img = tensor.squeeze(0).permute(1, 2, 0).numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    return img