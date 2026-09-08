"""
MapRot Visualization Engine.
Provides mask colorization, side-by-side comparative views, alpha-blended overlays,
and publication-ready plot generation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from maprot.core.constants import MAPROT_COLOR_PALETTE, MAPROT_CLASSES


class MapRotVisualizer:
    """
    Renders high-contrast segmentation maps, overlays, and side-by-side comparisons.
    """

    def __init__(self, palette: Optional[Dict[str, Tuple[int, int, int]]] = None):
        self.palette = palette or MAPROT_COLOR_PALETTE

    def colorize_mask(self, mask: np.ndarray, class_names: List[str]) -> np.ndarray:
        """
        Transforms integer class mask (H, W) into an RGB image (H, W, 3) using MapRot palette.
        """
        h, w = mask.shape[:2]
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)

        for idx, name in enumerate(class_names):
            color = self.palette.get(name.lower(), (128, 128, 128))
            color_mask[mask == idx] = color

        return color_mask

    def create_comparison_figure(
        self,
        image: np.ndarray,
        pred_mask: np.ndarray,
        class_names: List[str],
        gt_mask: Optional[np.ndarray] = None,
        title: Optional[str] = None,
        alpha: float = 0.45,
    ) -> plt.Figure:
        """
        Creates a side-by-side comparative figure showing:
        1. Original Satellite Image
        2. Ground Truth Mask (if provided)
        3. Predicted Segmentation Mask
        4. Alpha-Blended Overlay
        """
        # Ensure image is in [0, 1] range for matplotlib
        img_norm = image.copy()
        if img_norm.max() > 1.0:
            img_norm = img_norm / 255.0

        pred_color = self.colorize_mask(pred_mask, class_names)
        pred_color_norm = pred_color / 255.0

        overlay = (1 - alpha) * img_norm + alpha * pred_color_norm

        panels = [
            ("Satellite Imagery", img_norm),
        ]

        if gt_mask is not None:
            gt_color = self.colorize_mask(gt_mask, class_names) / 255.0
            panels.append(("Ground Truth", gt_color))

        panels.append(("MapRot Prediction", pred_color_norm))
        panels.append(("Overlay Blend", overlay))

        n_panels = len(panels)
        fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5), dpi=150)
        if n_panels == 1:
            axes = [axes]

        for ax, (panel_name, panel_img) in zip(axes, panels):
            ax.imshow(panel_img)
            ax.set_title(panel_name, fontsize=13, fontweight="bold", pad=8)
            ax.axis("off")

        # Add legend
        legend_handles = [
            mpatches.Patch(
                color=np.array(self.palette.get(cls_name, (128, 128, 128))) / 255.0,
                label=cls_name.capitalize(),
            )
            for cls_name in class_names
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=len(class_names),
            bbox_to_anchor=(0.5, -0.05),
            frameon=True,
            fontsize=11,
        )

        if title:
            fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)

        plt.tight_layout()
        return fig

    def save_plot(
        self,
        fig: plt.Figure,
        output_path: Union[str, Path],
    ) -> None:
        """Saves Matplotlib figure to disk with proper bounding box padding."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
