"""
MapRot Test-Time Augmentation (TTA) Engine.
Ensembles predictions across multi-orientation views to smooth tile boundary seams
and boost segmentation accuracy.
"""

from typing import Callable, List, Optional
import torch
import torch.nn as nn
import numpy as np


class TTAPredictor:
    """
    Applies multi-flip test-time augmentation during neural network inference.
    Combines predictions from Identity, Horizontal Flip, Vertical Flip, and Dual Flip.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

    def predict_patch_tta(self, patch_tensor: torch.Tensor) -> np.ndarray:
        """
        Executes 4-fold test-time augmentation on a single patch tensor (1, C, H, W).
        Returns class mask array (H, W).
        """
        x = patch_tensor.to(self.device)

        # 1. Identity
        p1 = self._forward(x)

        # 2. Horizontal Flip
        x_hflip = torch.flip(x, dims=[3])
        p2_raw = self._forward(x_hflip)
        p2 = torch.flip(p2_raw, dims=[3])

        # 3. Vertical Flip
        x_vflip = torch.flip(x, dims=[2])
        p3_raw = self._forward(x_vflip)
        p3 = torch.flip(p3_raw, dims=[2])

        # 4. Diagonal (H + V Flip)
        x_hvflip = torch.flip(x, dims=[2, 3])
        p4_raw = self._forward(x_hvflip)
        p4 = torch.flip(p4_raw, dims=[2, 3])

        # Average ensemble probabilities
        ensemble_prob = (p1 + p2 + p3 + p4) / 4.0
        pred_class = ensemble_prob.argmax(dim=1).squeeze(0).cpu().numpy()

        return pred_class

    def _forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "predict"):
            out = self.model.predict(tensor)
        else:
            out = self.model(tensor)
        return out
