"""
Machine Learning module for soccer outcome prediction and betting strategy.

This module provides the foundation for the betting system including:
- Data pipeline optimization
- Feature engineering
- ML model implementations
- Backtesting infrastructure
"""

__version__ = "1.0.0"
__author__ = "Betting System AI"

from .core import *
from .data import *
from .features import *
# from .models import *  # Not implemented yet
# from .backtesting import *  # Not implemented yet

__all__ = [
    # Core interfaces
    "Predictor",
    "DataProcessor", 
    "FeatureEngineer",
    "BacktestEngine",
    
    # Data pipeline
    "DataLoader",
    "DataValidator",
    "DataTransformer",
    "DataCache",
    
    # Feature engineering
    "TemporalFeatures",
    "ContextualFeatures", 
    "AdvancedMetrics",
    "FeatureSelector",
    
    # ML Models
    "XGBoostPredictor",
    "LightGBMPredictor", 
    "RandomForestPredictor",
    "EnsemblePredictor",
    
    # Backtesting
    "PerformanceAnalyzer",
    "RiskAnalyzer", 
    "ReportGenerator"
]