import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Provide a default anomaly threshold for inference workflows
DEFAULT_ANOMALY_THRESHOLD = 0.015

class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for chest X-ray image reconstruction.
    """
    def __init__(self):
        super().__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Decoder layers 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Pass input through encoder then decoder to reconstruct the image."""
        return self.decoder(self.encoder(x))

def process_and_evaluate(model, x):
    """
    Accepts an input image tensor.
    Reconstructs the image.
    Computes reconstruction error.
    Generates an anomaly map highlighting regions with high reconstruction error.
    
    Returns:
    1. reconstructed image (tensor)
    2. anomaly heatmap (numy array)
    3. reconstruction loss value (float)
    """
    model.eval()
    device = next(model.parameters()).device
    x = x.to(device)
    
    with torch.no_grad():
        # Reconstruct the image
        reconstructed = model(x)
        
        # Compute squared reconstruction error pixel-wise
        error = (x - reconstructed) ** 2
        
        # Generate anomaly heatmap (squeeze to shape [H, W])
        anomaly_heatmap = error.squeeze().cpu().numpy().astype(np.float32)
        
        # Compute scalar reconstruction loss (mean squared error)
        loss_value = float(error.mean().item())
        
    return reconstructed, anomaly_heatmap, loss_value


# ── Existing App Integration Functions ────────────────────────────────────────

def load_autoencoder(weights_path=None, device=None):
    """Loads the autoencoder. Optionally loads downloaded weights."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvAutoencoder().to(device)
    
    if weights_path and Path(weights_path).exists():
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        print(f"[Autoencoder] Loaded weights from {weights_path}")
    else:
        print("[Autoencoder] No weights provided — using random initialisation.")
        
    model.eval()
    return model

def compute_anomaly_map(model, tensor):
    """Wraps process_and_evaluate for app.py compatibility."""
    _, anomaly_heatmap, _ = process_and_evaluate(model, tensor)
    return anomaly_heatmap

def anomaly_score(error_map):
    """Compute mean anomaly score from an error map."""
    return float(error_map.mean())

def is_anomalous(error_map, threshold=DEFAULT_ANOMALY_THRESHOLD):
    """Check if the computed anomaly score exceeds the set threshold."""
    return anomaly_score(error_map) > threshold
