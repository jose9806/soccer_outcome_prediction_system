"""
Abstract base classes defining the core interfaces for the ML betting system.

Following the Interface Segregation Principle (ISP), each interface is focused
on a single responsibility to avoid forcing classes to implement methods they don't need.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

from .types import MatchData, MatchFeatures, Prediction, ModelMetrics, BacktestResult


class DataProcessor(ABC):
    """
    Interface for data processing components.
    
    Following Single Responsibility Principle - handles only data processing tasks.
    """
    
    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process raw data and return cleaned/transformed version."""
        pass
    
    @abstractmethod
    def validate(self, data: pd.DataFrame) -> bool:
        """Validate data quality and completeness."""
        pass


class FeatureEngineer(ABC):
    """
    Interface for feature engineering components.
    
    Separate from DataProcessor to follow SRP - focused only on feature creation.
    """
    
    @abstractmethod
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create new features from raw data."""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Return list of feature names this engineer creates."""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Return feature importance scores if available."""
        pass


class Predictor(ABC):
    """
    Interface for prediction models.
    
    Focused solely on making predictions - training/evaluation handled separately.
    """
    
    @abstractmethod
    def predict(self, features: MatchFeatures) -> Prediction:
        """Make prediction for a single match."""
        pass
    
    @abstractmethod
    def predict_batch(self, features: List[MatchFeatures]) -> List[Prediction]:
        """Make predictions for multiple matches efficiently."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata and configuration."""
        pass


class ModelTrainer(ABC):
    """
    Interface for model training components.
    
    Separate from Predictor to follow SRP - handles only training logic.
    """
    
    @abstractmethod
    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        """Train the model on provided data."""
        pass
    
    @abstractmethod
    def save_model(self, path: str) -> None:
        """Save trained model to disk."""
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> None:
        """Load trained model from disk."""
        pass


class ModelEvaluator(ABC):
    """
    Interface for model evaluation components.
    
    Dedicated to model performance assessment.
    """
    
    @abstractmethod
    def evaluate(self, predictions: List[Prediction], 
                actual_results: List[str]) -> ModelMetrics:
        """Evaluate model performance against actual results."""
        pass
    
    @abstractmethod
    def cross_validate(self, model: Predictor, 
                      features: pd.DataFrame, 
                      targets: pd.Series, 
                      cv_folds: int = 5) -> ModelMetrics:
        """Perform cross-validation evaluation."""
        pass


class BacktestEngine(ABC):
    """
    Interface for backtesting components.
    
    Focused on simulating historical betting performance.
    """
    
    @abstractmethod
    def run_backtest(self, 
                    predictor: Predictor,
                    historical_data: pd.DataFrame,
                    betting_strategy: 'BettingStrategy') -> BacktestResult:
        """Run complete backtest simulation."""
        pass
    
    @abstractmethod
    def analyze_performance(self, results: BacktestResult) -> Dict[str, float]:
        """Analyze backtest results and return key metrics."""
        pass


class BettingStrategy(ABC):
    """
    Interface for betting strategy components.
    
    Defines how predictions are converted into actual betting decisions.
    """
    
    @abstractmethod
    def should_bet(self, prediction: Prediction, odds: Dict[str, float]) -> bool:
        """Determine if a bet should be placed based on prediction and odds."""
        pass
    
    @abstractmethod
    def calculate_stake(self, prediction: Prediction, 
                       odds: Dict[str, float], 
                       bankroll: float) -> float:
        """Calculate appropriate stake size for the bet."""
        pass
    
    @abstractmethod
    def get_strategy_params(self) -> Dict[str, Any]:
        """Return strategy configuration parameters."""
        pass


class CacheManager(ABC):
    """
    Interface for caching components.
    
    Handles storage and retrieval of computed data to improve performance.
    """
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Retrieve cached data by key."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store data in cache with optional time-to-live."""
        pass
    
    @abstractmethod
    def invalidate(self, pattern: str) -> None:
        """Invalidate cached entries matching pattern."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cached data."""
        pass