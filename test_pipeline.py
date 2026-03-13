import torch
import torchvision
print("Torch", torch.__version__)
print("Torchvision", torchvision.__version__)

try:
    import torchxrayvision as xrv
    print("XRV loaded", xrv.__version__)
except Exception as e:
    print("XRV Error:", e)

from PIL import Image
import numpy as np
import io

import preprocess
import cnn_analyzer

try:
    print("Loading Mask Model...")
    mask_model = cnn_analyzer.build_lung_mask(device='cpu')
    print("Mask Model Loaded")

    print("Loading DenseNet Model...")
    cnn_model = cnn_analyzer.build_densenet(device='cpu')
    print("DenseNet Model Loaded")

    print("Testing Preprocessing...")
    # Make a dummy Image
    dummy_img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
    # Preprocess
    img_byte_arr = io.BytesIO()
    dummy_img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    img_byte_arr.name = "dummy.png"

    tensor_img, pil_out = preprocess.preprocess_image(img_byte_arr)
    print("Preprocess output tensor shape:", tensor_img.shape)

    print("Testing Lung Mask...")
    lung_tensor = cnn_analyzer.apply_lung_mask(mask_model, tensor_img)
    print("Lung Tensor shape:", lung_tensor.shape)
    
    print("Testing DNN inference...")
    results = cnn_analyzer.run_inference(cnn_model, lung_tensor)
    print("Inference results:")
    print("Top prediction:", results['top_labels'][0], results['top_probs'][0])
    
    print("Testing get_target_layer...")
    target_layer = cnn_analyzer.get_target_layer(cnn_model)
    print("Target layer extracted successfully.")

    print("SUCCESS")
except Exception as e:
    print("ERROR:", e)
    import traceback
    traceback.print_exc()
