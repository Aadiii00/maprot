"""
MapRot Segmentation Metrics Calculator.
Computes multi-class IoU, F1-Score (Dice coefficient), Precision, Recall,
and generates benchmark report summaries.
"""

from typing import Dict, List, Optional
import numpy as np


class MapRotMetricsCalculator:
    """
    Computes statistical performance metrics for semantic segmentation maps.
    """

    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.num_classes = len(class_names)

    def calculate_iou_per_class(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculates Intersection over Union (IoU) for each class."""
        ious = {}
        for idx, name in enumerate(self.class_names):
            true_mask = (y_true == idx)
            pred_mask = (y_pred == idx)

            intersection = np.logical_and(true_mask, pred_mask).sum()
            union = np.logical_or(true_mask, pred_mask).sum()

            if union == 0:
                ious[name] = 1.0 if intersection == 0 else 0.0
            else:
                ious[name] = float(intersection / union)
        return ious

    def calculate_dice_per_class(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculates Dice coefficient (F1 score) for each class."""
        dice_scores = {}
        for idx, name in enumerate(self.class_names):
            true_mask = (y_true == idx)
            pred_mask = (y_pred == idx)

            intersection = np.logical_and(true_mask, pred_mask).sum()
            total = true_mask.sum() + pred_mask.sum()

            if total == 0:
                dice_scores[name] = 1.0 if intersection == 0 else 0.0
            else:
                dice_scores[name] = float(2.0 * intersection / total)
        return dice_scores

    def calculate_precision_recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Calculates Precision and Recall for each class."""
        results = {}
        for idx, name in enumerate(self.class_names):
            true_mask = (y_true == idx)
            pred_mask = (y_pred == idx)

            tp = np.logical_and(true_mask, pred_mask).sum()
            fp = np.logical_and(~true_mask, pred_mask).sum()
            fn = np.logical_and(true_mask, ~pred_mask).sum()

            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

            results[name] = {"precision": precision, "recall": recall}
        return results

    def compute_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, any]:
        """
        Runs comprehensive evaluation on a pair of masks.
        """
        ious = self.calculate_iou_per_class(y_true, y_pred)
        dices = self.calculate_dice_per_class(y_true, y_pred)
        pr = self.calculate_precision_recall(y_true, y_pred)

        mean_iou = float(np.mean(list(ious.values())))
        mean_dice = float(np.mean(list(dices.values())))

        class_summary = {}
        for name in self.class_names:
            class_summary[name] = {
                "iou": ious[name],
                "dice": dices[name],
                "precision": pr[name]["precision"],
                "recall": pr[name]["recall"],
            }

        return {
            "mean_iou": mean_iou,
            "mean_dice": mean_dice,
            "classes": class_summary,
        }
