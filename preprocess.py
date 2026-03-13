import os
import io
import pydicom
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

def load_image(source):
    """
    Loads a medical image in DICOM, PNG, or JPG format.
    Accepts a file path or a file-like object (e.g., Streamlit UploadedFile).
    Returns a numpy array representing the image.
    """
    if hasattr(source, "name"):
        filename = source.name.lower()
        file_bytes = source.read()
        source.seek(0)
    else:
        filename = str(source).lower()
        with open(source, "rb") as f:
            file_bytes = f.read()

    ext = os.path.splitext(filename)[-1]

    if ext in ['.dcm', '.dicom']:
        dicom = pydicom.dcmread(io.BytesIO(file_bytes))
        img = dicom.pixel_array
        # Handle potential DICOM polarity inversion
        if getattr(dicom, 'PhotometricInterpretation', '') == 'MONOCHROME1':
            img = np.max(img) - img
        return img
    elif ext in ['.png', '.jpg', '.jpeg']:
        # Load with cv2 using an in-memory numpy buffer
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not decode image: {filename}")
        return img
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def convert_to_grayscale(img):
    """
    Converts images to grayscale.
    """
    if len(img.shape) == 3:
        if img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            return img[:, :, 0]
    return img

def preprocess_image(source):
    """
    Main pipeline function that performs preprocessing steps for TorchXRayVision:
    - Loads medical images in DICOM, PNG, or JPG format
    - Converts images to grayscale
    - Prepares the standard medical deep learning transforms (Compose)
    - Returns both the processed tensor and the original image (as PIL.Image)
    """
    # 1. Load image
    original_img_np = load_image(source)
    
    # Obtain original image as PIL to return as part of the output
    if len(original_img_np.shape) == 2:
        img_min, img_max = original_img_np.min(), original_img_np.max()
        if img_max > img_min:
            disp_img = ((original_img_np - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            disp_img = np.zeros_like(original_img_np, dtype=np.uint8)
        original_pil = Image.fromarray(disp_img, mode='L')
    else:
        rgb_img = cv2.cvtColor(original_img_np, cv2.COLOR_BGR2RGB)
        original_pil = Image.fromarray(rgb_img)

    # 2. Convert to grayscale
    gray_img = convert_to_grayscale(original_img_np)
    
    # 3. Create a PIL Image for TorchVision transforms
    gray_pil = Image.fromarray(gray_img)

    # 4. Standard Compose pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Apply standard medical normalization transforms
    tensor_img = transform(gray_pil).unsqueeze(0).float() # (1, 1, 224, 224)
    
    # 6. Return both processed and original
    return tensor_img, original_pil
