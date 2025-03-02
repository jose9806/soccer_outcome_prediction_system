"""
AI Models package for soccer match prediction.

This package contains various machine learning and deep learning models for predicting
match outcomes and in-game events.
"""

from .base_model import BaseModel
from .ml_models import (
    MatchOutcomeModel,
    ExpectedGoalsModel,
    EventPredictionModel,
    BTTSModel,
)

try:
    from .dl_models import DeepMatchOutcomeModel, DeepExpectedGoalsModel

    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False

__all__ = [
    "BaseModel",
    "MatchOutcomeModel",
    "ExpectedGoalsModel",
    "EventPredictionModel",
    "BTTSModel",
]

if DEEP_LEARNING_AVAILABLE:
    __all__.extend(["DeepMatchOutcomeModel", "DeepExpectedGoalsModel"])
