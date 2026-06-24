import torch
import torch.nn as nn
import torchvision.models as models
import json

def load_class_names(path: str = "models/class_names.json"):
    with open(path, 'r') as f:
        return json.load(f)

def build_model(num_classes: int = 3):
    model = models.efficientnet_b0(pretrained=False)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    return model

def load_cnn_model(weights_path: str = "models/cnn_weights.pth"):
    class_names = load_class_names()
    num_classes = len(class_names)
    
    model = build_model(num_classes)
    
    device = torch.device('cpu')
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, class_names