"""
MapRot Interactive Web Dashboard & Demo Server.
Serves an interactive web application for aerial raster segmentation,
class-selective filtering, Test-Time Augmentation (TTA), GeoTIFF georeferencing,
and vector GeoJSON polygon exports.
"""

import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from maprot.core.config import MapRotConfig
from maprot.core.constants import MAPROT_CLASSES, MAPROT_COLOR_PALETTE
from maprot.data.tiling import RasterTiler
from maprot.evaluation.visualization import MapRotVisualizer
from maprot.gis.georaster import read_georaster_meta
from maprot.gis.measurements import compute_ground_area
from maprot.gis.vectorizer import mask_to_geojson_features
from maprot.models.factory import load_model_checkpoint
from maprot.models.tta import TTAPredictor
import segmentation_models_pytorch as smp

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Global caches
MODEL = None
CONFIG = None
PREPROCESSING_FN = None
TILER = None
VISUALIZER = None
TTA_ENGINE = None
LATEST_GEOJSON = None


def init_engine():
    global MODEL, CONFIG, PREPROCESSING_FN, TILER, VISUALIZER, TTA_ENGINE
    if MODEL is None:
        config_path = PROJECT_ROOT / "configs" / "maprot_default.yaml"
        CONFIG = MapRotConfig.from_yaml(config_path, root_dir=PROJECT_ROOT)
        device_name = CONFIG.device if torch.cuda.is_available() and CONFIG.device != "cpu" else "cpu"
        device = torch.device(device_name)

        print(f"[MapRot Web] Loading model onto device: {device}...")
        MODEL = load_model_checkpoint(CONFIG.checkpoint_path, device=device)
        PREPROCESSING_FN = smp.encoders.get_preprocessing_fn(CONFIG.encoder, CONFIG.encoder_weights)
        TILER = RasterTiler(patch_size=CONFIG.patch_size)
        VISUALIZER = MapRotVisualizer()
        TTA_ENGINE = TTAPredictor(MODEL, device=device)
        print("[MapRot Web] Engine ready with GIS and TTA capabilities.")


def array_to_base64_png(arr: np.ndarray) -> str:
    """Converts RGB numpy array into a base64 data URI."""
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

    pil_img = Image.fromarray(arr)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/samples", methods=["GET"])
def get_samples():
    init_engine()
    test_dir = CONFIG.test_dir / "images"
    samples = []
    if test_dir.exists():
        for f in test_dir.iterdir():
            if f.suffix.lower() in [".tif", ".tiff", ".png", ".jpg"]:
                samples.append({
                    "id": f.name,
                    "name": f.stem,
                    "filename": f.name,
                    "size_mb": round(f.stat().st_size / (1024 * 1024), 2)
                })
    return jsonify({
        "samples": samples,
        "classes": list(MAPROT_CLASSES),
        "palette": MAPROT_COLOR_PALETTE
    })


@app.route("/api/segment", methods=["POST"])
def run_segmentation():
    global LATEST_GEOJSON
    init_engine()
    t_start = time.time()

    # Determine input raster
    image_rgb = None
    filename = "upload.png"
    geo_meta = None
    use_tta = False

    if "file" in request.files and request.files["file"].filename != "":
        uploaded = request.files["file"]
        filename = uploaded.filename
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        raw_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if raw_img is not None:
            image_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        use_tta = request.form.get("use_tta") in ["true", "1", "True"]

    elif request.is_json and "sample_id" in request.json:
        sample_name = request.json["sample_id"]
        sample_path = CONFIG.test_dir / "images" / sample_name
        if sample_path.exists():
            filename = sample_name
            raw_img = cv2.imread(str(sample_path), cv2.IMREAD_COLOR)
            if raw_img is not None:
                image_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            try:
                geo_meta = read_georaster_meta(sample_path)
            except Exception:
                pass
        use_tta = bool(request.json.get("use_tta", False))

    if image_rgb is None:
        return jsonify({"error": "No valid image provided"}), 400

    # Parse target classes
    selected_classes = None
    if request.is_json and "classes" in request.json:
        selected_classes = request.json["classes"]
    elif "classes" in request.form:
        selected_classes = request.form.getlist("classes")

    if not selected_classes:
        selected_classes = ["background", "building", "water"]

    # Target class index mapping
    class_indices = [CONFIG.all_classes.index(cls.lower()) for cls in selected_classes if cls.lower() in CONFIG.all_classes]
    if 0 not in class_indices:
        class_indices.insert(0, 0)  # Always include background

    # Pad raster for tiling
    padded_img, orig_shape = TILER.prepare_padded_raster(image_rgb)
    patches, (n_rows, n_cols) = TILER.extract_inference_patches(padded_img, overlap=False)
    patch_grid = np.empty((n_rows, n_cols, CONFIG.patch_size, CONFIG.patch_size), dtype=np.int64)

    device = next(MODEL.parameters()).device
    with torch.no_grad():
        for r in range(n_rows):
            for c in range(n_cols):
                patch = patches[r, c, :, :, :]
                preprocessed = PREPROCESSING_FN(patch).transpose(2, 0, 1).astype("float32")
                x_tensor = torch.from_numpy(preprocessed).unsqueeze(0)

                if use_tta and TTA_ENGINE:
                    pred_class = TTA_ENGINE.predict_patch_tta(x_tensor)
                else:
                    logits = MODEL.predict(x_tensor.to(device)) if hasattr(MODEL, "predict") else MODEL(x_tensor.to(device))
                    pred = logits.squeeze(0).cpu().numpy().round()
                    pred_class = pred.argmax(0)

                patch_grid[r, c, :, :] = pred_class

    # Reconstruct full-size mask
    reconstructed = TILER.reconstruct_from_patches(patch_grid, padded_img.shape[:2], orig_shape)

    # Class filter
    filtered_channels = [(reconstructed == idx) for idx in class_indices]
    final_mask = np.stack(filtered_channels, axis=-1).astype("float32").argmax(2)

    active_class_names = [CONFIG.all_classes[idx] for idx in class_indices]

    # Vectorize to GeoJSON
    try:
        LATEST_GEOJSON = mask_to_geojson_features(
            mask=final_mask,
            class_names=active_class_names,
            meta=geo_meta,
            simplify_tolerance=1.5,
        )
    except Exception as e:
        print(f"[MapRot Web] GeoJSON extraction warning: {e}")
        LATEST_GEOJSON = {"type": "FeatureCollection", "features": []}

    # Calculate land-cover statistics & real ground area (hectares, m2)
    total_pixels = final_mask.size
    coverage_stats = []
    for local_idx, orig_class_idx in enumerate(class_indices):
        class_name = CONFIG.all_classes[orig_class_idx]
        pixel_count = int((final_mask == local_idx).sum())
        percentage = round((pixel_count / total_pixels) * 100, 2)
        color_rgb = MAPROT_COLOR_PALETTE.get(class_name, (128, 128, 128))
        color_hex = "#{:02x}{:02x}{:02x}".format(*color_rgb)

        ground_metrics = compute_ground_area(pixel_count, meta=geo_meta)

        coverage_stats.append({
            "class": class_name,
            "pixels": pixel_count,
            "percentage": percentage,
            "hectares": ground_metrics["hectares"],
            "sq_meters": ground_metrics["sq_meters"],
            "acres": ground_metrics["acres"],
            "color": color_hex,
            "color_rgb": color_rgb,
        })

    # Total area metrics
    total_ground_metrics = compute_ground_area(total_pixels, meta=geo_meta)

    # Render colorized mask and blended overlay
    color_mask = VISUALIZER.colorize_mask(final_mask, active_class_names)

    img_norm = image_rgb.astype("float32") / 255.0
    color_norm = color_mask.astype("float32") / 255.0
    alpha = 0.50
    overlay = ((1 - alpha) * img_norm + alpha * color_norm) * 255.0

    # Extract pixel-space contours for Three.js 3D Digital Twin extrusion
    contours_3d = []
    for local_idx, orig_class_idx in enumerate(class_indices):
        class_name = CONFIG.all_classes[orig_class_idx]
        if class_name.lower() == "background":
            continue
        bin_m = (final_mask == local_idx).astype(np.uint8)
        cnts, _ = cv2.findContours(bin_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < 25:
                continue
            approx = cv2.approxPolyDP(cnt, epsilon=1.8, closed=True)
            if len(approx) < 3:
                continue
            pts = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]
            contours_3d.append({
                "class": class_name,
                "area": float(area),
                "points": pts,
            })

    t_elapsed = round(time.time() - t_start, 2)

    return jsonify({
        "status": "success",
        "filename": filename,
        "dimensions": {"width": orig_shape[1], "height": orig_shape[0]},
        "time_seconds": t_elapsed,
        "use_tta": use_tta,
        "crs": geo_meta.crs if (geo_meta and geo_meta.crs) else "Projected (Standard)",
        "total_area_hectares": total_ground_metrics["hectares"],
        "selected_classes": active_class_names,
        "stats": coverage_stats,
        "num_polygons": len(LATEST_GEOJSON.get("features", [])),
        "contours_3d": contours_3d,
        "geojson": LATEST_GEOJSON,
        "images": {
            "original": array_to_base64_png(image_rgb),
            "mask": array_to_base64_png(color_mask),
            "overlay": array_to_base64_png(overlay),
        }
    })


@app.route("/api/export_geojson", methods=["GET"])
def export_geojson_endpoint():
    global LATEST_GEOJSON
    if LATEST_GEOJSON is None:
        return jsonify({"error": "No segmented features available. Run segmentation first."}), 400

    response_data = json.dumps(LATEST_GEOJSON, indent=2)
    return Response(
        response_data,
        mimetype="application/geo+json",
        headers={"Content-Disposition": "attachment;filename=maprot_vector_features.geojson"},
    )


def start_server(host="127.0.0.1", port=5000):
    init_engine()
    print("\n=======================================================")
    print("MapRot Interactive Web Dashboard is running at:")
    print(f"--> http://{host}:{port}")
    print("=======================================================\n")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    start_server(port=port)
