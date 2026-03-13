import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torchvision import models

CHEXNET_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural Thickening", "Hernia"
]
NUM_CLASSES = len(CHEXNET_LABELS)

class ModifiedDenseNet(nn.Module):
    """
    Modified DenseNet121 that extracts feature maps and feature vectors representing anatomical structures.
    """
    def __init__(self, original_model):
        super().__init__()
        # Inherit the feature extraction part
        self.features = original_model.features
        # Inherit the classification head
        self.classifier = original_model.classifier
        # Behavior flag (useful for compatibility with tools like Grad-CAM that expect single output)
        self.return_features = False

    def forward(self, x):
        # Extract feature maps from the final convolutional layer
        feature_maps = self.features(x)
        
        # Produce feature vector representing anatomical structures
        out = F.relu(feature_maps, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        feature_vector = torch.flatten(out, 1)
        
        # Produce prediction scores
        logits = self.classifier(feature_vector)
        
        if self.return_features:
            return feature_maps, feature_vector, logits
        return logits

def build_densenet(weights_path=None, device=None):
    """
    Load a pretrained DenseNet121 model using torchvision.
    Initialize its classifier head, and wrap it into ModifiedDenseNet.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pretrained DenseNet121
    base_model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    in_features = base_model.classifier.in_features
    
    # Modify classifier to output the required number of classes representing anatomical structures
    base_model.classifier = nn.Sequential(
        nn.Linear(in_features, NUM_CLASSES),
        nn.Sigmoid()
    )
    
    # Wrap in our ModifiedDenseNet
    model = ModifiedDenseNet(base_model)
    
    if weights_path and Path(weights_path).exists():
        state = torch.load(weights_path, map_location=device)
        if "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        print(f"[DenseNet121] Loaded weights from {weights_path}")
    else:
        print("[DenseNet121] Using ImageNet pretrained weights.")
        
    model = model.to(device)
    model.eval()
    return model

@torch.no_grad()
def run_inference(model, tensor, top_k=5):
    """
    Accepts a processed image tensor.
    Runs inference using DenseNet121.
    Returns extracted feature maps and prediction scores inside a dictionary for compatibility.
    """
    device = next(model.parameters()).device
    tensor = tensor.to(device)
    
    # Temporarily set the flag to obtain the multiple returned vectors
    model.return_features = True
    feature_maps, feature_vector, probs_tensor = model(tensor)
    model.return_features = False
    
    probs = probs_tensor.squeeze().cpu().numpy()
    feature_maps = feature_maps.cpu()
    feature_vector = feature_vector.cpu()
    
    all_probs = {label: float(prob) for label, prob in zip(CHEXNET_LABELS, probs)}
    
    sorted_indices = np.argsort(probs)[::-1]
    top_k = min(top_k, NUM_CLASSES)
    top_idx_list = sorted_indices[:top_k].tolist()
    
    top_labels = [CHEXNET_LABELS[i] for i in top_idx_list]
    top_probs = [float(probs[i]) for i in top_idx_list]
    
    # Return dictionary that includes the exact requested deliverables:
    return {
        "feature_maps": feature_maps,
        "feature_vector": feature_vector,
        "all_probs": all_probs,
        "top_labels": top_labels,
        "top_probs": top_probs,
        "top_idx": int(sorted_indices[0]),
    }

def get_target_layer(model):
    """
    Return the last convolutional block of DenseNet121.
    """
    return model.features.denseblock4
