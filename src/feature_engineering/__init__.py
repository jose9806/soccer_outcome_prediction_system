"""
Feature engineering module for soccer match prediction.

This package contains utilities for extracting, transforming, and selecting
features from processed soccer match data to use in prediction models.
"""

from src.feature_engineering.pipelines.feature_pipeline import FeaturePipeline
from src.feature_engineering.extractors.team_performance_features import (
    TeamPerformanceFeatures,
)
from src.feature_engineering.extractors.match_context_features import (
    MatchContextFeatures,
)
from src.feature_engineering.extractors.advanced_metrics import AdvancedMetrics
from src.feature_engineering.extractors.temporal_features import TemporalFeatures
from src.feature_engineering.selectors.feature_selector import FeatureSelector

__all__ = [
    "FeaturePipeline",
    "TeamPerformanceFeatures",
    "MatchContextFeatures",
    "AdvancedMetrics",
    "TemporalFeatures",
    "FeatureSelector",
]
