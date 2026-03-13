import os
import io
import cv2
import numpy as np
import pydicom
import torch
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

def resize_image(img, size=(224, 224)):
    """
    Resizes images to the specified dimensions (default: 224x224).
    """
    img = img.astype(np.float32)
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def normalize_image(img):
    """
    Normalizes pixel values between 0 and 1.
    """
    min_val = np.min(img)
    max_val = np.max(img)
    if max_val > min_val:
        return (img - min_val) / (max_val - min_val)
    return img - min_val

def to_tensor(img):
    """
    Converts the image into a PyTorch tensor with shape (1, H, W).
    """
    return torch.from_numpy(img).unsqueeze(0).float()

def preprocess_image(source):
    """
    Main pipeline function that performs all required preprocessing steps:
    - Loads medical images in DICOM, PNG, or JPG format
    - Converts images to grayscale
    - Resizes images to 224x224
    - Normalizes pixel values between 0 and 1
    - Converts the image into a PyTorch tensor
    - Returns both the processed tensor and the original image (as PIL.Image)
    
    Args:
        source: File path or file-like object (like Streamlit upload)
        
    Returns:
        tuple: (processed_tensor, original_pil_image)
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
    
    # 3. Resize to 224x224
    resized_img = resize_image(gray_img, (224, 224))
    
    # 4. Normalize between 0 and 1
    normalized_img = normalize_image(resized_img)
    
    # 5. Convert to tensor
    tensor_img = to_tensor(normalized_img)
    
    # 6. Return both processed and original
    return tensor_img, original_pil
