"""
ML Models package for soccer outcome prediction.

Provides production-ready predictors including XGBoost, LightGBM, RandomForest
and ensemble methods for betting strategy optimization.
"""

from .predictors import (
    XGBoostPredictor,
    LightGBMPredictor, 
    RandomForestPredictor
)

from .ensemble import (
    EnsemblePredictor,
    MetaLearner,
    ModelSelector
)

from .trainer import (
    ModelTrainer,
    CrossValidator,
    HyperparameterOptimizer
)

__all__ = [
    # Individual predictors
    "XGBoostPredictor",
    "LightGBMPredictor", 
    "RandomForestPredictor",
    
    # Ensemble methods
    "EnsemblePredictor",
    "MetaLearner", 
    "ModelSelector",
    
    # Training utilities
    "ModelTrainer",
    "CrossValidator",
    "HyperparameterOptimizer"
]