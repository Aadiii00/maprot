"""
MapRot Model Evaluation & Benchmarking Pipeline.
Computes dataset-wide IoU, Dice coefficients, and precision-recall benchmarks.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np
import torch
from tqdm import tqdm
import segmentation_models_pytorch as smp

from maprot.core.config import MapRotConfig
from maprot.core.logger import get_maprot_logger
from maprot.data.tiling import RasterTiler
from maprot.evaluation.metrics import MapRotMetricsCalculator
from maprot.models.factory import load_model_checkpoint


class MapRotEvaluator:
    """
    Benchmarks MapRot segmentation models against ground truth test rasters.
    """

    def __init__(self, config: MapRotConfig):
        self.cfg = config
        self.logger = get_maprot_logger(
            name="MapRot-Evaluator",
            log_file=self.cfg.logs_dir / "maprot_eval.log",
            level=self.cfg.log_level,
        )

        device_name = self.cfg.device if torch.cuda.is_available() and self.cfg.device != "cpu" else "cpu"
        self.device = torch.device(device_name)
        self.tiler = RasterTiler(patch_size=self.cfg.patch_size)
        self.metrics_calc = MapRotMetricsCalculator(class_names=self.cfg.train_classes)

    def evaluate(self, checkpoint_path: Optional[Path | str] = None) -> Dict[str, any]:
        """
        Runs evaluation over test directory and returns aggregate metric scores.
        """
        model_path = Path(checkpoint_path) if checkpoint_path else self.cfg.checkpoint_path
        self.logger.info(f"Loading model checkpoint from: {model_path}")
        model = load_model_checkpoint(model_path, device=self.device)

        preprocessing_fn = smp.encoders.get_preprocessing_fn(
            self.cfg.encoder, self.cfg.encoder_weights
        )

        test_img_dir = self.cfg.test_dir / "images"
        test_msk_dir = self.cfg.test_dir / "masks"

        img_files = [f for f in test_img_dir.iterdir() if f.suffix.lower() == self.cfg.file_type.lower()]
        self.logger.info(f"Found {len(img_files)} test rasters to benchmark.")

        if not img_files:
            raise FileNotFoundError(f"No test images with extension {self.cfg.file_type} found in {test_img_dir}")

        all_class_ious: Dict[str, List[float]] = {cls: [] for cls in self.cfg.train_classes}
        all_class_dices: Dict[str, List[float]] = {cls: [] for cls in self.cfg.train_classes}

        class_indices = [self.cfg.all_classes.index(cls.lower()) for cls in self.cfg.train_classes]

        for file_item in tqdm(img_files, desc="Benchmarking test rasters"):
            # Load raw image
            image = cv2.imread(str(file_item), cv2.IMREAD_COLOR)
            if image is None:
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Load ground truth mask
            mask_path = test_msk_dir / file_item.name
            if not mask_path.exists():
                self.logger.warning(f"Skipping {file_item.name}: Ground truth mask missing.")
                continue

            raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            gt_masks = [(raw_mask == idx) for idx in class_indices]
            gt_mask = np.stack(gt_masks, axis=-1).astype("float32").argmax(2)

            # Tiling and Prediction
            padded_img, orig_shape = self.tiler.prepare_padded_raster(image_rgb)
            patches, (n_rows, n_cols) = self.tiler.extract_inference_patches(padded_img, overlap=False)
            pred_grid = np.empty((n_rows, n_cols, self.cfg.patch_size, self.cfg.patch_size), dtype=np.int64)

            with torch.no_grad():
                for r in range(n_rows):
                    for c in range(n_cols):
                        patch = patches[r, c, :, :, :]
                        preprocessed = preprocessing_fn(patch).transpose(2, 0, 1).astype("float32")
                        x_tensor = torch.from_numpy(preprocessed).to(self.device).unsqueeze(0)

                        logits = model.predict(x_tensor) if hasattr(model, "predict") else model(x_tensor)
                        pred = logits.squeeze(0).cpu().numpy()
                        pred_class = pred.argmax(0)
                        pred_grid[r, c, :, :] = pred_class

            pred_full = self.tiler.reconstruct_from_patches(
                pred_grid, padded_img.shape[:2], orig_shape
            )

            metrics = self.metrics_calc.compute_all_metrics(gt_mask, pred_full)
            for cls_name in self.cfg.train_classes:
                all_class_ious[cls_name].append(metrics["classes"][cls_name]["iou"])
                all_class_dices[cls_name].append(metrics["classes"][cls_name]["dice"])

        # Summary calculation
        summary = {
            "mean_iou": float(np.mean([np.mean(vals) for vals in all_class_ious.values() if vals])),
            "mean_dice": float(np.mean([np.mean(vals) for vals in all_class_dices.values() if vals])),
            "per_class": {
                cls: {
                    "mean_iou": float(np.mean(all_class_ious[cls])) if all_class_ious[cls] else 0.0,
                    "mean_dice": float(np.mean(all_class_dices[cls])) if all_class_dices[cls] else 0.0,
                }
                for cls in self.cfg.train_classes
            },
        }

        # Save benchmark output
        benchmark_file = self.cfg.outputs_dir / "benchmark_summary.json"
        benchmark_file.parent.mkdir(parents=True, exist_ok=True)
        with open(benchmark_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Evaluation benchmark complete! Saved to {benchmark_file}")
        self.logger.info(f"Mean IoU: {summary['mean_iou']:.4f} | Mean Dice: {summary['mean_dice']:.4f}")
        return summary
