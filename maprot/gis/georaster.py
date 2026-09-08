"""
MapRot GeoTIFF Georeferencing Engine.
Reads geospatial coordinates and writes QGIS/ArcGIS-compatible georeferenced rasters.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import rasterio
from rasterio.transform import Affine


@dataclass
class GeoRasterMeta:
    """Stores geospatial metadata extracted from aerial/satellite GeoTIFFs."""
    crs: Optional[str]
    transform: Affine
    width: int
    height: int
    pixel_size_x: float
    pixel_size_y: float
    bounds: Tuple[float, float, float, float]
    has_georeference: bool


def read_georaster_meta(raster_path: Union[str, Path]) -> GeoRasterMeta:
    """
    Inspects a GeoTIFF raster and extracts spatial reference systems and affine transforms.
    """
    path = Path(raster_path)
    with rasterio.open(path) as src:
        crs_str = src.crs.to_string() if src.crs else None
        transform = src.transform
        res_x, res_y = abs(transform.a), abs(transform.e)
        bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
        has_geo = src.crs is not None and not transform.is_identity

        return GeoRasterMeta(
            crs=crs_str,
            transform=transform,
            width=src.width,
            height=src.height,
            pixel_size_x=res_x if res_x > 0 else 1.0,
            pixel_size_y=res_y if res_y > 0 else 1.0,
            bounds=bounds,
            has_georeference=has_geo,
        )


def write_georeferenced_mask(
    mask: np.ndarray,
    output_path: Union[str, Path],
    src_meta: Optional[GeoRasterMeta] = None,
) -> Path:
    """
    Saves a segmentation mask as a compressed GeoTIFF, copying CRS and affine bounds.
    Directly compatible with QGIS, ArcGIS, and Google Earth Engine.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = mask.shape[:2]
    dtype = rasterio.uint8 if mask.max() <= 255 else rasterio.uint16

    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": dtype,
        "compress": "lzw",
    }

    if src_meta and src_meta.has_georeference:
        profile["crs"] = src_meta.crs
        profile["transform"] = src_meta.transform

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask.astype(profile["dtype"]), 1)

    return out_path
