"""
MapRot Geospatial Surface Measurements.
Calculates ground surface areas in real-world units (m², hectares, acres).
"""

from typing import Dict, List, Optional
import numpy as np
from maprot.gis.georaster import GeoRasterMeta


def compute_ground_area(
    pixel_count: int,
    meta: Optional[GeoRasterMeta] = None,
    default_gsd_meters: float = 0.5,
) -> Dict[str, float]:
    """
    Converts pixel count to real-world surface metrics.

    Args:
        pixel_count: Number of segmented pixels.
        meta: Optional GeoRasterMeta providing spatial resolution.
        default_gsd_meters: Fallback Ground Sample Distance (meters per pixel).
    """
    if meta and meta.has_georeference:
        pixel_area_m2 = meta.pixel_size_x * meta.pixel_size_y
    else:
        pixel_area_m2 = default_gsd_meters * default_gsd_meters

    area_sq_meters = float(pixel_count * pixel_area_m2)
    area_hectares = float(area_sq_meters / 10000.0)
    area_acres = float(area_hectares * 2.47105)

    return {
        "sq_meters": round(area_sq_meters, 2),
        "hectares": round(area_hectares, 3),
        "acres": round(area_acres, 3),
    }
