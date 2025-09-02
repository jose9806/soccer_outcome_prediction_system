"""
Feature Engineering module for soccer outcome prediction.

This module provides comprehensive feature engineering capabilities including:
- Temporal features (rolling statistics, momentum, form)
- Contextual features (home advantage, rivalries, travel)
- Advanced metrics (xG gaps, efficiency ratios, clutch performance)
- Automatic feature selection and importance analysis
"""

from .temporal import (
    TemporalFeaturesEngine,
    RollingStatsCalculator,
    MomentumAnalyzer,
    FormCalculator,
    StreakDetector
)

from .contextual import (
    ContextualFeaturesEngine,
    HomeAdvantageCalculator,
    RivalryAnalyzer,
    TravelFatigueCalculator,
    SeasonalAdjuster
)

from .advanced import (
    AdvancedMetricsEngine,
    ExpectedGoalsAnalyzer,
    EfficiencyCalculator,
    ClutchPerformanceAnalyzer,
    StyleAnalyzer
)

from .selector import (
    FeatureSelector,
    ImportanceAnalyzer,
    CorrelationAnalyzer,
    RedundancyRemover
)

from .engine import (
    FeatureEngineeringEngine,
    FeatureConfig,
    FeaturePipeline
)

__all__ = [
    # Temporal features
    "TemporalFeaturesEngine",
    "RollingStatsCalculator",
    "MomentumAnalyzer", 
    "FormCalculator",
    "StreakDetector",
    
    # Contextual features
    "ContextualFeaturesEngine",
    "HomeAdvantageCalculator",
    "RivalryAnalyzer",
    "TravelFatigueCalculator",
    "SeasonalAdjuster",
    
    # Advanced metrics
    "AdvancedMetricsEngine",
    "ExpectedGoalsAnalyzer",
    "EfficiencyCalculator",
    "ClutchPerformanceAnalyzer",
    "StyleAnalyzer",
    
    # Feature selection
    "FeatureSelector",
    "ImportanceAnalyzer",
    "CorrelationAnalyzer",
    "RedundancyRemover",
    
    # Main engine
    "FeatureEngineeringEngine",
    "FeatureConfig",
    "FeaturePipeline"
]