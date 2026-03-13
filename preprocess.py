import torch
import numpy as np
from PIL import Image
import io

def preprocess_image(source):
    """
    Loads and preprocesses a chest X-ray image for the CheXNet pipeline.
    Returns both the processed tensor (1, 1, 224, 224) and the original PIL image.
    """
    # 1. Load image
    if hasattr(source, "read"):
        file_bytes = source.read()
        source.seek(0)
        img_pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    else:
        img_pil = Image.open(source).convert("RGB")
    
    # 2. Convert to grayscale for tensor
    img_gray = img_pil.convert("L")
    
    # 3. Resize specifically for the model
    img_resized = img_gray.resize((224, 224))
    
    # 4. Normalize to [0, 1]
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    
    # 5. Convert to tensor (1, 1, 224, 224) as expected by cnn_analyzer
    tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
    
    # Return both as expected by app.py UI
    return tensor, img_pil