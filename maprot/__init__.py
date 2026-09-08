"""
MapRot: Geospatial Surface Analytics & Semantic Segmentation Framework.

A high-performance PyTorch framework for land-cover classification,
satellite raster tiling, and selective multi-class semantic segmentation.
"""

__version__ = "2.0.0"
__author__ = "MapRot Core Team"

from maprot.core.constants import MAPROT_CLASSES, MAPROT_COLOR_PALETTE
from maprot.core.config import MapRotConfig

# Lazy imports for heavy pipelines/models to allow fast CLI startup
def get_trainer():
    from maprot.pipelines.trainer import MapRotTrainer
    return MapRotTrainer

def get_evaluator():
    from maprot.pipelines.evaluator import MapRotEvaluator
    return MapRotEvaluator

def get_predictor():
    from maprot.pipelines.predictor import MapRotPredictor
    return MapRotPredictor

def get_model_factory():
    from maprot.models.factory import build_segmentation_model
    return build_segmentation_model

__all__ = [
    "MAPROT_CLASSES",
    "MAPROT_COLOR_PALETTE",
    "MapRotConfig",
    "get_trainer",
    "get_evaluator",
    "get_predictor",
    "get_model_factory",
]
