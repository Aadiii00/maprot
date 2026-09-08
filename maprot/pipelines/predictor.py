"""
MapRot Geospatial Prediction & Class-Selective Inference Engine.
Performs sliding-window inference on large satellite rasters with selective class filtering,
optional Test-Time Augmentation (TTA), GeoTIFF georeference preservation,
and GIS vector GeoJSON polygon exports.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import cv2
import numpy as np
import torch
from tqdm import tqdm
import segmentation_models_pytorch as smp

from maprot.core.config import MapRotConfig
from maprot.core.logger import get_maprot_logger
from maprot.data.tiling import RasterTiler
from maprot.evaluation.visualization import MapRotVisualizer
from maprot.gis.georaster import read_georaster_meta, write_georeferenced_mask
from maprot.gis.measurements import compute_ground_area
from maprot.gis.vectorizer import mask_to_geojson_features, export_geojson
from maprot.models.factory import load_model_checkpoint
from maprot.models.tta import TTAPredictor


class MapRotPredictor:
    """
    Enterprise-grade inference engine over satellite/aerial rasters with selective class filtering,
    Test-Time Augmentation, CRS preservation, and GeoJSON polygonization.
    """

    def __init__(self, config: MapRotConfig):
        self.cfg = config
        self.logger = get_maprot_logger(
            name="MapRot-Predictor",
            log_file=self.cfg.logs_dir / "maprot_predict.log",
            level=self.cfg.log_level,
        )

        device_name = self.cfg.device if torch.cuda.is_available() and self.cfg.device != "cpu" else "cpu"
        self.device = torch.device(device_name)
        self.tiler = RasterTiler(patch_size=self.cfg.patch_size)
        self.visualizer = MapRotVisualizer()

    def predict_directory(
        self,
        input_dir: Optional[Union[str, Path]] = None,
        selected_classes: Optional[List[str]] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        use_tta: bool = False,
        export_vector_geojson: bool = True,
        save_plots: bool = True,
    ) -> List[Path]:
        """
        Runs batch prediction on all rasters in input_dir with geospatial vectorization.
        """
        img_dir = Path(input_dir) if input_dir else self.cfg.test_dir / "images"
        model_path = Path(checkpoint_path) if checkpoint_path else self.cfg.checkpoint_path
        target_classes = selected_classes or self.cfg.inference_classes

        self.logger.info(f"Loading checkpoint: {model_path} (Device: {self.device})")
        model = load_model_checkpoint(model_path, device=self.device)
        tta_engine = TTAPredictor(model, device=self.device) if use_tta else None

        preprocessing_fn = smp.encoders.get_preprocessing_fn(
            self.cfg.encoder, self.cfg.encoder_weights
        )

        files = [
            f for f in img_dir.iterdir()
            if f.suffix.lower() in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
        ]
        self.logger.info(f"Discovered {len(files)} rasters. Classes: {target_classes} | TTA: {use_tta}")

        pred_mask_dir = self.cfg.outputs_dir / "predictions"
        pred_plot_dir = self.cfg.outputs_dir / "plots"
        pred_vector_dir = self.cfg.outputs_dir / "vectors"
        pred_mask_dir.mkdir(parents=True, exist_ok=True)
        pred_plot_dir.mkdir(parents=True, exist_ok=True)
        if export_vector_geojson:
            pred_vector_dir.mkdir(parents=True, exist_ok=True)

        class_indices = [self.cfg.all_classes.index(cls.lower()) for cls in target_classes]
        output_files = []

        for file_item in tqdm(files, desc="Running MapRot Inference"):
            image = cv2.imread(str(file_item), cv2.IMREAD_COLOR)
            if image is None:
                self.logger.warning(f"Failed to read image {file_item.name}, skipping.")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Inspect geospatial metadata
            geo_meta = None
            try:
                geo_meta = read_georaster_meta(file_item)
            except Exception:
                pass

            # Pad raster
            padded_img, orig_shape = self.tiler.prepare_padded_raster(image_rgb)

            # Extract patches
            patches, (n_rows, n_cols) = self.tiler.extract_inference_patches(padded_img, overlap=False)
            patch_grid = np.empty((n_rows, n_cols, self.cfg.patch_size, self.cfg.patch_size), dtype=np.int64)

            with torch.no_grad():
                for r in range(n_rows):
                    for c in range(n_cols):
                        patch = patches[r, c, :, :, :]
                        preprocessed = preprocessing_fn(patch).transpose(2, 0, 1).astype("float32")
                        x_tensor = torch.from_numpy(preprocessed).unsqueeze(0)

                        if use_tta and tta_engine:
                            pred_class = tta_engine.predict_patch_tta(x_tensor)
                        else:
                            logits = model.predict(x_tensor.to(self.device)) if hasattr(model, "predict") else model(x_tensor.to(self.device))
                            pred = logits.squeeze(0).cpu().numpy().round()
                            pred_class = pred.argmax(0)

                        patch_grid[r, c, :, :] = pred_class

            # Reconstruct full-size mask
            reconstructed_mask = self.tiler.reconstruct_from_patches(
                patch_grid, padded_img.shape[:2], orig_shape
            )

            # Apply selective class filter
            filtered_channels = [(reconstructed_mask == idx) for idx in class_indices]
            final_mask = np.stack(filtered_channels, axis=-1).astype("float32").argmax(2)

            # Save prediction mask with CRS metadata preserved
            out_mask_path = pred_mask_dir / file_item.name
            if geo_meta and geo_meta.has_georeference:
                write_georeferenced_mask(final_mask, out_mask_path, src_meta=geo_meta)
            else:
                cv2.imwrite(str(out_mask_path), final_mask.astype(np.uint8))
            output_files.append(out_mask_path)

            # Export GIS Vector Polygons as GeoJSON
            if export_vector_geojson:
                geojson_data = mask_to_geojson_features(
                    mask=final_mask,
                    class_names=target_classes,
                    meta=geo_meta,
                    simplify_tolerance=1.5,
                )
                geojson_path = pred_vector_dir / f"{file_item.stem}_features.geojson"
                export_geojson(geojson_data, geojson_path)

            # Save visual comparison plot
            if save_plots:
                gt_mask_path = self.cfg.test_dir / "masks" / file_item.name
                gt_mask = None
                if gt_mask_path.exists():
                    raw_gt = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
                    gt_channels = [(raw_gt == idx) for idx in class_indices]
                    gt_mask = np.stack(gt_channels, axis=-1).astype("float32").argmax(2)

                fig = self.visualizer.create_comparison_figure(
                    image=image_rgb,
                    pred_mask=final_mask,
                    class_names=target_classes,
                    gt_mask=gt_mask,
                    title=f"MapRot Surface Segmentation: {file_item.stem}",
                )
                plot_path = pred_plot_dir / f"{file_item.stem}_segmentation.png"
                self.visualizer.save_plot(fig, plot_path)

        self.logger.info(f"Inference complete: {len(output_files)} georeferenced masks generated.")
        return output_files
