"""
MapRot Model Factory.
Constructs deep segmentation architectures with custom encoders and pre-trained backbones.
"""

from pathlib import Path
from typing import Optional, Union
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


SUPPORTED_ARCHITECTURES = {
    "unet": smp.Unet,
    "unetplusplus": smp.UnetPlusPlus,
    "fpn": smp.FPN,
    "pspnet": smp.PSPNet,
    "deeplabv3": smp.DeepLabV3,
    "deeplabv3plus": smp.DeepLabV3Plus,
    "pan": smp.PAN,
    "linknet": smp.Linknet,
    "manet": smp.MAnet,
}


def build_segmentation_model(
    arch: str = "Unet",
    encoder_name: str = "efficientnet-b0",
    encoder_weights: Optional[str] = "imagenet",
    classes: int = 4,
    activation: Optional[str] = "softmax2d",
) -> nn.Module:
    """
    Instantiates a PyTorch segmentation model via segmentation_models_pytorch.

    Args:
        arch: Model architecture name (e.g., Unet, FPN, DeepLabV3Plus).
        encoder_name: Backbone encoder (e.g., resnet34, efficientnet-b0).
        encoder_weights: Pretrained weights name (e.g., 'imagenet' or None).
        classes: Number of output prediction classes.
        activation: Final activation layer (e.g. 'softmax2d', 'sigmoid', or None).
    """
    key = arch.lower()
    if key not in SUPPORTED_ARCHITECTURES:
        available = ", ".join(SUPPORTED_ARCHITECTURES.keys())
        raise ValueError(f"Architecture '{arch}' not supported by MapRot. Available: {available}")

    model_cls = SUPPORTED_ARCHITECTURES[key]
    model = model_cls(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        classes=classes,
        activation=activation,
    )
    return model


def load_model_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> nn.Module:
    """
    Safely loads a serialized PyTorch model checkpoint.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"MapRot model weights not found at: {path}")

    device = torch.device(device)
    model = torch.load(path, map_location=device, weights_only=False)
    model.eval()
    return model
