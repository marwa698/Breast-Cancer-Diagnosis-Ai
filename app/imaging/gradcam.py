import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import io
import base64
from app.imaging.cnn_model import load_cnn_model
from app.imaging.image_preprocessing import preprocess_image

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Hook على آخر layer في الـ features
        target_layer = model.features[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, tensor, class_idx):
        self.model.zero_grad()
        output = self.model(tensor)
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def generate_gradcam(image_file):
    """
    ترجع صورة Grad-CAM كـ base64
    """
    model, class_names = load_cnn_model()
    tensor, original_img = preprocess_image(image_file)
    tensor.requires_grad_(True)

    gradcam = GradCAM(model)

    with torch.enable_grad():
        outputs = model(tensor)
        pred_idx = outputs.argmax(dim=1).item()

    cam = gradcam.generate(tensor, pred_idx)

    # Overlay على الصورة الأصلية
    img_np = np.array(original_img.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.6 * img_np + 0.4 * heatmap).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    axes[1].imshow(overlay)
    axes[1].set_title("Grad-CAM Explanation")
    axes[1].axis('off')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return img_base64, class_names[pred_idx]