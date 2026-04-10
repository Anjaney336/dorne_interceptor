"""Product-style core module facade."""

from drone_interceptor.core.control import InterceptionController
from drone_interceptor.core.optimization import InterceptionCostModel
from drone_interceptor.core.perception import TargetDetector
from drone_interceptor.core.prediction import TargetPredictor
from drone_interceptor.core.tracking import TargetTracker

__all__ = [
    "InterceptionController",
    "InterceptionCostModel",
    "TargetDetector",
    "TargetPredictor",
    "TargetTracker",
]
