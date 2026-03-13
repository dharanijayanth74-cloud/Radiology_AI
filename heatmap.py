import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def generate_gradcam(model: nn.Module, 
                     tensor: torch.Tensor, 
                     target_layer: nn.Module, 
                     class_idx: int = None) -> np.ndarray:

    targets = [ClassifierOutputTarget(class_idx)] if class_idx is not None else None

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=tensor, targets=targets)

    # Extract batch index 0
    cam = grayscale_cam[0]

    # Normalize heatmap
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)

    return cam
    
def overlay_heatmap(cam: np.ndarray, 
                    original_image, 
                    alpha: float = 0.5, 
                    colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Overlays a Grad-CAM heatmap onto the original image.
    
    Args:
        cam: The generated 2D grayscale CAM array [0, 1].
        original_image: A PIL image or a numpy array representing the original image.
        alpha: Opacity representing how strongly the heatmap blends.
        colormap: OpenCV colormap (for the standalone display logic).
        
    Returns:
        A 3D numpy array representing the overlaid RGB image in uint8 format.
    """
    # Ensure standard dimensions are kept
    if isinstance(original_image, Image.Image):
        original_image = np.array(original_image.resize((224, 224)).convert("RGB"))
        
    # Resize cam in case there is a mismatch
    h, w = original_image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    
    # Normalize original image to [0, 1] for show_cam_on_image
    original_normalized = original_image.astype(np.float32) / 255.0
    
    # Convert overlay leveraging pytorch_grad_cam's utility
    blended = show_cam_on_image(original_normalized, cam_resized, use_rgb=True, colormap=colormap, image_weight=1.0 - alpha)
    
    return blended

def cam_to_heatmap(cam: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Converts a raw 2D float CAM array to a bright colorized visual representation.
    """
    cam_u8 = (cam * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(cam_u8, colormap)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# ── Required Fallback Utilities (Shared UI Helpers) ───────────────────────────
# Kept for compatibility with `app.py` UI metrics expectations.

def apply_colormap(grayscale: np.ndarray, colormap: int = cv2.COLORMAP_HOT) -> np.ndarray:
    """Applies a colormap to a 0-1 grayscale float array directly."""
    gray_u8 = np.clip(grayscale * 255, 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(gray_u8, colormap)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def blend_images(base: np.ndarray, overlay: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Weighted alpha blend of UI maps."""
    if base.shape[:2] != overlay.shape[:2]:
        overlay = cv2.resize(overlay, (base.shape[1], base.shape[0]))
    return cv2.addWeighted(base, 1 - alpha, overlay, alpha, 0)
