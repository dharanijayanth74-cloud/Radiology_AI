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
def run_inference(model, tensor, top_k=5):
    device = next(model.parameters()).device
    tensor = tensor.to(device)

# FIX
    if tensor.shape[1] == 1:
        tensor = tensor.repeat(1,3,1,1)

    preds = model(tensor)
    probs = preds[0].cpu().numpy()

    threshold = 0.5
    abnormalities = [
        (CHEXNET_LABELS[i], float(probs[i]))
        for i in range(len(probs))
        if probs[i] > threshold
    ]

    abnormality_score = float(np.max(probs))

    sorted_idx = np.argsort(probs)[::-1]

    top_labels = []
    top_probs = []

    for i in sorted_idx[:top_k]:
        top_labels.append(CHEXNET_LABELS[i])
        top_probs.append(float(probs[i]))

    return {
    "abnormal_score": abnormality_score,
    "abnormalities": abnormalities,
    "top_labels": top_labels,
    "top_probs": top_probs,
    "top_idx": int(sorted_idx[0]),
}



def get_target_layer(model: nn.Module) -> nn.Module:
    """
    Return the last convolutional block of DenseNet121 — used as the
    Grad-CAM target layer.  This is `features.denseblock4`.
    """
    return model.features.denseblock4
