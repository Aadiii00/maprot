"""
MapRot Image Augmentation and Preprocessing Pipeline.
Provides Albumentations transformations for satellite image normalization and augmentation.
"""

from typing import Callable, Optional
import albumentations as album
import numpy as np


def to_chw_tensor(x: np.ndarray, **kwargs) -> np.ndarray:
    """Converts HWC array to CHW float32 tensor array."""
    return x.transpose(2, 0, 1).astype("float32")


def get_training_augmentation(p_flip: float = 0.5) -> album.Compose:
    """
    Returns standard geospatial augmentation pipeline:
    Random horizontal and vertical reflections suitable for satellite nadir imagery.
    """
    transforms = [
        album.HorizontalFlip(p=p_flip),
        album.VerticalFlip(p=p_flip),
        album.RandomRotate90(p=0.5),
    ]
    return album.Compose(transforms)


def get_preprocessing_pipeline(
    preprocessing_fn: Optional[Callable] = None,
) -> album.Compose:
    """
    Constructs model-specific preprocessing (e.g. ImageNet normalization)
    and formats data for PyTorch tensor ingestion.
    """
    transforms = []
    if preprocessing_fn is not None:
        transforms.append(album.Lambda(image=preprocessing_fn))

    transforms.append(
        album.Lambda(image=to_chw_tensor, mask=to_chw_tensor)
    )
    return album.Compose(transforms)
