import torch
import torch.nn.functional as F
from app.imaging.cnn_model import load_cnn_model
from app.imaging.image_preprocessing import preprocess_image

def predict_image(image_file):
    """
    تاخد صورة وترجع التشخيص + نسبة الثقة
    """
    model, class_names = load_cnn_model()
    tensor, original_img = preprocess_image(image_file)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    probs = {class_names[i]: round(float(probabilities[i]) * 100, 2) 
             for i in range(len(class_names))}
    
    predicted_idx = probabilities.argmax().item()
    predicted_class = class_names[predicted_idx]
    confidence = probs[predicted_class]

    # Label مع emoji
    emoji_map = {
        'benign': '🟢 Benign',
        'malignant': '🔴 Malignant', 
        'normal': '✅ Normal'
    }
    label = emoji_map.get(predicted_class, predicted_class)

    return {
        "prediction": predicted_class,
        "label": label,
        "confidence": confidence,
        "probabilities": probs,
        "original_image": original_img
    }