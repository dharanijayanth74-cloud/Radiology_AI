"""
cnn_analyzer.py
---------------
Module 3 – CNN Anatomical Feature Analyzer

Uses a DenseNet121 backbone (ImageNet-pretrained, torchvision) with a
14-class head matching the CheXNet pathology labels.

If `weights_path` points to a CheXNet-compatible checkpoint the model will
load those weights; otherwise it falls back to ImageNet weights and the
probabilities will reflect visual similarity to ImageNet classes rather than
pathology — clearly communicated to the user in the Streamlit UI.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torchvision import models

# ── CheXNet / NIH ChestX-ray14 Pathology Labels ──────────────────────────────
CHEXNET_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural Thickening",
    "Hernia",
]

NUM_CLASSES = len(CHEXNET_LABELS)   # 14


# ── Model Building ────────────────────────────────────────────────────────────

def build_densenet(weights_path: str | None = None,
                   device: str | None = None) -> nn.Module:
    """
    Build DenseNet121 with a 14-class classification head.

    Parameters
    ----------
    weights_path : str | None
        Path to a CheXNet .pth checkpoint. If None, ImageNet weights are used.
    device : str | None
        'cuda', 'cpu', or None (auto-detect).

    Returns
    -------
    nn.Module  in eval mode on the selected device.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ── Backbone ──────────────────────────────────────────────────────────────
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

    # ── Replace classifier head for 14 pathology classes ─────────────────────
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, NUM_CLASSES),
        nn.Sigmoid(),           # multi-label: each class 0–1 independently
    )

    # ── Optionally load CheXNet weights ──────────────────────────────────────
    if weights_path and Path(weights_path).exists():
        state = torch.load(weights_path, map_location=device)
        # Handle checkpoints that wrap state_dict under a key
        if "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        print(f"[DenseNet121] Loaded CheXNet weights from {weights_path}")
    else:
        print("[DenseNet121] Using ImageNet pretrained weights (no CheXNet).")

    model = model.to(device)
    model.eval()
    return model


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model: nn.Module,
                  tensor: torch.Tensor,
                  top_k: int = 5) -> dict:
    """
    Run a forward pass and return pathology probabilities.

    Parameters
    ----------
    model  : DenseNet121 built by `build_densenet()`
    tensor : torch.Tensor  shape (1, 3, 224, 224), ImageNet-normalised
    top_k  : int  number of top labels to include

    Returns
    -------
    dict with keys:
        'all_probs'  : dict {label: float}  – all 14 probabilities
        'top_labels' : list[str]            – top-k label names
        'top_probs'  : list[float]          – corresponding probabilities
        'top_idx'    : int                  – index of highest-prob class
    """
    device = next(model.parameters()).device
    tensor = tensor.to(device)

    probs = model(tensor).squeeze().cpu().numpy()          # shape (14,)

    all_probs = {label: float(prob)
                 for label, prob in zip(CHEXNET_LABELS, probs)}

    sorted_indices = np.argsort(probs)[::-1]
    top_k          = min(top_k, NUM_CLASSES)
    top_idx_list   = sorted_indices[:top_k].tolist()

    top_labels = [CHEXNET_LABELS[i] for i in top_idx_list]
    top_probs  = [float(probs[i])   for i in top_idx_list]

    return {
        "all_probs":  all_probs,
        "top_labels": top_labels,
        "top_probs":  top_probs,
        "top_idx":    int(sorted_indices[0]),   # single best class for Grad-CAM
    }


def get_target_layer(model: nn.Module) -> nn.Module:
    """
    Return the last convolutional block of DenseNet121 — used as the
    Grad-CAM target layer.  This is `features.denseblock4`.
    """
    return model.features.denseblock4
