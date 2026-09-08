"""
MapRot Geospatial Segmentation Dataset.
Implements PyTorch Dataset for aerial and satellite land-cover imagery.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as album


class MapRotSegmentationDataset(Dataset):
    """
    MapRot PyTorch Dataset for multi-class surface segmentation.

    Args:
        images_dir: Directory containing input images (e.g., .tif, .png).
        masks_dir: Directory containing corresponding ground truth label masks.
        all_classes: Master list of all recognized classes in the dataset.
        classes: Subset of classes to select and train/evaluate on.
        augmentation: Albumentations augmentation pipeline.
        preprocessing: Albumentations preprocessing / tensor conversion pipeline.
    """

    def __init__(
        self,
        images_dir: Union[str, Path],
        masks_dir: Optional[Union[str, Path]] = None,
        all_classes: Optional[List[str]] = None,
        classes: Optional[List[str]] = None,
        augmentation: Optional[album.Compose] = None,
        preprocessing: Optional[album.Compose] = None,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir) if masks_dir else None

        self.ids = [
            f.name for f in sorted(self.images_dir.iterdir())
            if f.suffix.lower() in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
        ]
        self.image_paths = [self.images_dir / img_id for img_id in self.ids]
        self.mask_paths = [self.masks_dir / img_id for img_id in self.ids] if self.masks_dir else []

        self.all_classes = [c.lower() for c in (all_classes or ["background", "building", "woodland", "water", "road"])]
        self.target_classes = [c.lower() for c in (classes or self.all_classes)]
        self.class_values = [self.all_classes.index(cls) for cls in self.target_classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        # Read RGB image
        img_path = str(self.image_paths[idx])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0

        mask = None
        if self.masks_dir and idx < len(self.mask_paths) and self.mask_paths[idx].exists():
            mask_path = str(self.mask_paths[idx])
            raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if raw_mask is None:
                raise FileNotFoundError(f"Could not load mask: {mask_path}")

            # Extract binary channels for each chosen class
            masks_per_class = [(raw_mask == v) for v in self.class_values]
            mask = np.stack(masks_per_class, axis=-1).astype("float32")

        # Apply spatial augmentations
        if self.augmentation:
            if mask is not None:
                augmented = self.augmentation(image=image, mask=mask)
                image, mask = augmented["image"], augmented["mask"]
            else:
                augmented = self.augmentation(image=image)
                image = augmented["image"]

        # Apply preprocessing (normalization and channel transpose)
        if self.preprocessing:
            if mask is not None:
                preprocessed = self.preprocessing(image=image, mask=mask)
                image, mask = preprocessed["image"], preprocessed["mask"]
            else:
                preprocessed = self.preprocessing(image=image)
                image = preprocessed["image"]

        return (image, mask) if mask is not None else (image, np.empty(0))
