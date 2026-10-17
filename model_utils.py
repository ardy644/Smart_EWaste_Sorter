"""
model_utils.py — Smart E-Waste & Recyclable Vision Sorter
=========================================================
Contains:
  • BaselineCNN architecture (4-layer CNN matching the Kaggle-trained checkpoint)
  • EfficientNetV2-S loader (transfer-learning head swap)
  • Image preprocessing (224×224, ImageNet normalisation)
  • Grad-CAM implementation (hooks into final conv layer)
  • Single-image and batch inference helpers
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Paths (relative to repo root — resolved at import time)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent / "ewaste_deploy_bundle"
EFFICIENTNET_WEIGHTS = BASE_DIR / "efficientnet_ewaste.pth"
BASELINE_WEIGHTS = BASE_DIR / "baseline_cnn.pth"
METRICS_JSON = BASE_DIR / "eda_and_metrics.json"
SAMPLE_IMG_DIR = BASE_DIR / "sample_images"

# ---------------------------------------------------------------------------
# Load metadata once
# ---------------------------------------------------------------------------
with open(METRICS_JSON, "r", encoding="utf-8") as _f:
    METRICS: dict = json.load(_f)

CLASSES: List[str] = METRICS["classes"]  # length-21 list, alphabetical
NUM_CLASSES: int = len(CLASSES)

# ---------------------------------------------------------------------------
# Preprocessing transform (matches Kaggle training pipeline)
# ---------------------------------------------------------------------------
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """PIL Image → (1, 3, 224, 224) float32 tensor."""
    return TRANSFORM(image.convert("RGB")).unsqueeze(0)


# ---------------------------------------------------------------------------
# BaselineCNN — must mirror the architecture used during Kaggle training
# ---------------------------------------------------------------------------
class BaselineCNN(nn.Module):
    """4 × (Conv2d → BatchNorm2d → ReLU → MaxPool2d) → GAP → Dropout → FC"""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3 → 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2: 32 → 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3: 64 → 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 4: 128 → 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Model loaders (cached via Streamlit @st.cache_resource externally)
# ---------------------------------------------------------------------------
def load_efficientnet(device: torch.device | str = "cpu") -> nn.Module:
    """Load EfficientNetV2-S with the custom 21-class head."""
    model = models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    state = torch.load(EFFICIENTNET_WEIGHTS, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def load_baseline(device: torch.device | str = "cpu") -> nn.Module:
    """Load the 4-layer BaselineCNN."""
    model = BaselineCNN(NUM_CLASSES)
    state = torch.load(BASELINE_WEIGHTS, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def count_parameters(model: nn.Module) -> int:
    """Total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(
    model: nn.Module,
    tensor: torch.Tensor,
    device: str = "cpu",
) -> Tuple[str, float, np.ndarray, float]:
    """
    Returns (predicted_class, confidence%, full_probs, latency_ms).
    """
    tensor = tensor.to(device)
    t0 = time.perf_counter()
    logits = model(tensor)
    latency_ms = (time.perf_counter() - t0) * 1000
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(probs.argmax())
    return CLASSES[idx], float(probs[idx]) * 100, probs, latency_ms


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
class GradCAM:
    """
    Grad-CAM for any CNN.
    Hooks into `target_layer` to capture activations & gradients.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        # Register hooks
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Returns a (224, 224) heatmap in [0, 255] uint8.
        """
        self.model.zero_grad()
        input_tensor = input_tensor.requires_grad_(True)
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        target = output[0, class_idx]
        target.backward()

        gradients = self._gradients[0]        # (C, H, W)
        activations = self._activations[0]    # (C, H, W)

        # Global-average-pool the gradients → channel weights
        weights = gradients.mean(dim=(1, 2))  # (C,)

        # Weighted combination of activation maps
        cam = (weights[:, None, None] * activations).sum(dim=0)  # (H, W)
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam_np = cam.cpu().numpy()
        cam_resized = cv2.resize(cam_np, (224, 224))
        cam_uint8 = np.uint8(255 * cam_resized)
        return cam_uint8

    def remove_hooks(self) -> None:
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def get_gradcam_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """Return the correct final-conv target layer for Grad-CAM."""
    if model_name == "EfficientNetV2-S":
        return model.features[-1]
    else:  # BaselineCNN
        # features[-2] is the last BatchNorm; we want the last Conv block's ReLU output
        # The last conv block is features[12:16] → Conv,BN,ReLU,Pool
        # Hook into the ReLU (index 14) for best spatial resolution before pooling
        return model.features[-2]  # MaxPool2d of last block — still has spatial dims


def generate_gradcam_overlay(
    model: nn.Module,
    model_name: str,
    input_tensor: torch.Tensor,
    original_image: Image.Image,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (original_rgb, raw_heatmap_rgb, overlay_rgb) — all 224×224 uint8.
    """
    target_layer = get_gradcam_target_layer(model, model_name)
    cam = GradCAM(model, target_layer)

    heatmap = cam.generate(input_tensor.clone())
    cam.remove_hooks()

    # Coloured heatmap
    heatmap_colour = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colour = cv2.cvtColor(heatmap_colour, cv2.COLOR_BGR2RGB)

    # Original resized
    orig_resized = np.array(original_image.convert("RGB").resize((224, 224)))

    # Overlay
    overlay = cv2.addWeighted(orig_resized, 0.55, heatmap_colour, 0.45, 0)

    return orig_resized, heatmap_colour, overlay
