"""
Core interfaces and abstract base classes for the ML system.

This module defines the fundamental contracts that all components must follow,
ensuring consistency and enabling dependency injection throughout the system.
"""

from .interfaces import (
    Predictor,
    DataProcessor,
    FeatureEngineer,
    BacktestEngine,
    ModelEvaluator
)

from .exceptions import (
    MLSystemError,
    DataProcessingError,
    ModelTrainingError,
    BacktestingError,
    ValidationError
)

from .types import (
    MatchData,
    MatchFeatures,
    Prediction,
    ModelMetrics,
    BacktestResult
)

__all__ = [
    # Interfaces
    "Predictor",
    "DataProcessor", 
    "FeatureEngineer",
    "BacktestEngine",
    "ModelEvaluator",
    
    # Exceptions
    "MLSystemError",
    "DataProcessingError",
    "ModelTrainingError", 
    "BacktestingError",
    "ValidationError",
    
    # Types
    "MatchData",
    "MatchFeatures", 
    "Prediction",
    "ModelMetrics",
    "BacktestResult"
]