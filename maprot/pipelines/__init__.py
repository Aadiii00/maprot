"""
MapRot Pipelines Package.
Contains end-to-end task pipelines for training, evaluation, and inference.
"""

from maprot.pipelines.trainer import MapRotTrainer
from maprot.pipelines.evaluator import MapRotEvaluator
from maprot.pipelines.predictor import MapRotPredictor

__all__ = ["MapRotTrainer", "MapRotEvaluator", "MapRotPredictor"]
