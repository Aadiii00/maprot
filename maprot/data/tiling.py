"""
MapRot Geospatial Raster Tiler.
Provides patch extraction, overlapping sliding windows, unpatching reconstruction,
and dataset filtering for large satellite / aerial rasters.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import math
import cv2
import numpy as np
from patchify import patchify, unpatchify
from tqdm import tqdm


class RasterTiler:
    """
    Handles sliding-window patch generation and seamless mask reconstruction.
    """

    def __init__(self, patch_size: int = 512):
        self.patch_size = patch_size

    def tile_directory(
        self,
        src_dir: Union[str, Path],
        dst_dir: Union[str, Path],
        file_extension: str = ".tif",
        step: Optional[int] = None,
    ) -> int:
        """
        Tiles all matching rasters in src_dir and writes cropped patches into dst_dir.
        """
        src_path = Path(src_dir)
        dst_path = Path(dst_dir)
        dst_path.mkdir(parents=True, exist_ok=True)

        step_size = step or self.patch_size
        file_list = [f for f in src_path.iterdir() if f.suffix.lower() == file_extension.lower()]
        total_created = 0

        for file_item in tqdm(file_list, desc=f"Tiling {src_path.name}"):
            img = cv2.imread(str(file_item), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            # Crop dimensions to divisible patch multiple
            max_h = (img.shape[0] // self.patch_size) * self.patch_size
            max_w = (img.shape[1] // self.patch_size) * self.patch_size
            if max_h == 0 or max_w == 0:
                continue

            cropped = img[:max_h, :max_w]
            is_color = len(img.shape) == 3 and img.shape[2] == 3

            if is_color:
                patches = patchify(cropped, (self.patch_size, self.patch_size, 3), step=step_size)
                for i in range(patches.shape[0]):
                    for j in range(patches.shape[1]):
                        patch = patches[i, j, 0, :, :]
                        out_name = file_item.stem + f"_patch_{i}_{j}" + file_item.suffix
                        cv2.imwrite(str(dst_path / out_name), patch)
                        total_created += 1
            else:
                patches = patchify(cropped, (self.patch_size, self.patch_size), step=step_size)
                for i in range(patches.shape[0]):
                    for j in range(patches.shape[1]):
                        patch = patches[i, j, :, :]
                        out_name = file_item.stem + f"_patch_{i}_{j}" + file_item.suffix
                        cv2.imwrite(str(dst_path / out_name), patch)
                        total_created += 1

        return total_created

    def filter_monotone_patches(
        self,
        images_dir: Union[str, Path],
        masks_dir: Union[str, Path],
        discard_rate: float = 0.95,
        background_class_id: int = 0,
    ) -> int:
        """
        Removes patches where the background class exceeds discard_rate.
        """
        img_dir = Path(images_dir)
        msk_dir = Path(masks_dir)
        discarded = 0

        mask_files = [f for f in msk_dir.iterdir() if f.is_file()]
        for msk_file in tqdm(mask_files, desc="Filtering monotone patches"):
            mask = cv2.imread(str(msk_file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            unique_classes, counts = np.unique(mask, return_counts=True)
            bg_ratio = 0.0
            if background_class_id in unique_classes:
                idx = np.where(unique_classes == background_class_id)[0][0]
                bg_ratio = counts[idx] / counts.sum()

            if bg_ratio > discard_rate:
                corresponding_img = img_dir / msk_file.name
                if corresponding_img.exists():
                    corresponding_img.unlink()
                msk_file.unlink()
                discarded += 1

        return discarded

    def prepare_padded_raster(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Pads image via reflection to perfectly align with patch size grid.
        Returns:
            (padded_image, (original_height, original_width))
        """
        h, w = image.shape[:2]
        pad_h = (math.ceil(h / self.patch_size) * self.patch_size) - h
        pad_w = (math.ceil(w / self.patch_size) * self.patch_size) - w

        if len(image.shape) == 3:
            padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        else:
            padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect")

        return padded, (h, w)

    def extract_inference_patches(
        self,
        padded_image: np.ndarray,
        overlap: bool = True,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Splits image into inference grid.
        Returns patch tensor shape (num_rows, num_cols, H, W, C).
        """
        step = self.patch_size // 2 if overlap else self.patch_size
        patches = patchify(padded_image, (self.patch_size, self.patch_size, 3), step=step)
        # Squeeze singleton patchify channel dimension
        patches = patches[:, :, 0, :, :, :]
        return patches, (patches.shape[0], patches.shape[1])

    def reconstruct_from_patches(
        self,
        mask_patches: np.ndarray,
        padded_shape: Tuple[int, int],
        orig_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Unpatchifies predicted patch grid and unpads to original image dimensions.
        """
        reconstructed = unpatchify(mask_patches, padded_shape)
        orig_h, orig_w = orig_shape
        return reconstructed[:orig_h, :orig_w]
