"""
MapRot Geospatial Vectorizer.
Converts pixel segmentation masks into GIS vector polygons (GeoJSON FeatureCollections).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import rasterio.features
from shapely.geometry import shape, mapping
from shapely.ops import transform as shapely_transform

from maprot.gis.georaster import GeoRasterMeta


def mask_to_geojson_features(
    mask: np.ndarray,
    class_names: List[str],
    meta: Optional[GeoRasterMeta] = None,
    simplify_tolerance: float = 1.0,
    min_area_pixels: int = 15,
) -> Dict[str, Any]:
    """
    Converts a multi-class raster segmentation mask into standard GeoJSON features.

    Args:
        mask: 2D integer class mask (H, W).
        class_names: List of class labels corresponding to mask indices.
        meta: Optional GeoRasterMeta for real-world geospatial coordinates.
        simplify_tolerance: Douglas-Peucker polygon simplification tolerance in map units.
        min_area_pixels: Minimum blob size to include (filters out single-pixel noise).
    """
    features = []
    affine_transform = meta.transform if (meta and meta.has_georeference) else None

    for class_idx, class_name in enumerate(class_names):
        if class_name.lower() == "background":
            continue  # Omit background polygons to keep GeoJSON focused on assets

        binary_mask = (mask == class_idx).astype(np.uint8)
        if not binary_mask.any():
            continue

        # Extract shapes from binary mask
        shapes_gen = rasterio.features.shapes(
            binary_mask,
            mask=(binary_mask == 1),
            transform=affine_transform,
        )

        for geom_dict, val in shapes_gen:
            poly = shape(geom_dict)
            if not poly.is_valid:
                poly = poly.buffer(0)

            # Filter out tiny noise artifacts
            if poly.area < (min_area_pixels * (meta.pixel_size_x * meta.pixel_size_y if meta else 1)):
                continue

            # Simplify geometry for clean vector boundaries
            if simplify_tolerance > 0:
                poly = poly.simplify(simplify_tolerance, preserve_topology=True)

            feature = {
                "type": "Feature",
                "properties": {
                    "class": class_name,
                    "class_id": class_idx,
                    "area_map_units": round(poly.area, 2),
                },
                "geometry": mapping(poly),
            }
            features.append(feature)

    geojson_doc = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": meta.crs if (meta and meta.crs) else "urn:ogc:def:crs:OGC:1.3:CRS84"}
        },
        "features": features,
    }
    return geojson_doc


def export_geojson(
    geojson_data: Dict[str, Any],
    output_path: Union[str, Path],
) -> Path:
    """Writes GeoJSON document to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, indent=2)
    return path
