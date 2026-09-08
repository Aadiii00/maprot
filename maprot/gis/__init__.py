"""
MapRot Geospatial Information Systems (GIS) Subsystem.
Provides GeoTIFF georeferencing preservation, coordinate transformation,
vector polygonization, and real-world ground measurement analytics.
"""

from maprot.gis.georaster import (
    GeoRasterMeta,
    read_georaster_meta,
    write_georeferenced_mask,
)
from maprot.gis.vectorizer import mask_to_geojson_features, export_geojson
from maprot.gis.measurements import compute_ground_area

__all__ = [
    "GeoRasterMeta",
    "read_georaster_meta",
    "write_georeferenced_mask",
    "mask_to_geojson_features",
    "export_geojson",
    "compute_ground_area",
]
