"""
Feature Engineering Engine for soccer match analysis.

Main orchestrator that coordinates all feature engineering components
to create a comprehensive feature pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
import time

from src.config.logging_config import get_logger
from ..core.interfaces import FeatureEngineer
from ..core.exceptions import FeatureEngineeringError

from .temporal import TemporalFeaturesEngine, TemporalConfig
from .contextual import ContextualFeaturesEngine, ContextualConfig  
from .advanced import AdvancedMetricsEngine, AdvancedMetricsConfig
from .selector import FeatureSelector, FeatureSelectorConfig


@dataclass
class FeatureConfig:
    """Configuration for the complete feature engineering pipeline."""
    
    # Component enable/disable flags
    enable_temporal: bool = True
    enable_contextual: bool = True
    enable_advanced: bool = True
    enable_selection: bool = True
    
    # Individual component configs
    temporal_config: Optional[TemporalConfig] = None
    contextual_config: Optional[ContextualConfig] = None
    advanced_config: Optional[AdvancedMetricsConfig] = None
    selector_config: Optional[FeatureSelectorConfig] = None
    
    # Pipeline settings
    cache_features: bool = True
    parallel_processing: bool = True
    feature_validation: bool = True
    performance_monitoring: bool = True
    
    def __post_init__(self):
        if self.temporal_config is None:
            self.temporal_config = TemporalConfig()
        if self.contextual_config is None:
            self.contextual_config = ContextualConfig()
        if self.advanced_config is None:
            self.advanced_config = AdvancedMetricsConfig()
        if self.selector_config is None:
            self.selector_config = FeatureSelectorConfig()


class FeaturePipeline:
    """
    Pipeline for sequential feature engineering processing.
    
    Manages the execution order and data flow between feature components.
    """
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.logger = get_logger("FeaturePipeline")
        
        # Initialize components based on configuration
        self.components = {}
        
        if config.enable_temporal:
            self.components['temporal'] = TemporalFeaturesEngine(config.temporal_config)
        
        if config.enable_contextual:
            self.components['contextual'] = ContextualFeaturesEngine(config.contextual_config)
        
        if config.enable_advanced:
            self.components['advanced'] = AdvancedMetricsEngine(config.advanced_config)
        
        if config.enable_selection:
            self.components['selector'] = FeatureSelector(config.selector_config)
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process data through the complete feature engineering pipeline.
        
        Args:
            data: Input DataFrame with match data
            
        Returns:
            DataFrame with all engineered features
        """
        try:
            self.logger.info(f"Starting pipeline processing for {len(data)} matches")
            start_time = time.time()
            
            result = data.copy()
            processing_stats = {}
            
            # Process through each component in order
            for component_name, component in self.components.items():
                if component_name == 'selector':
                    # Feature selection is handled separately
                    continue
                
                component_start = time.time()
                self.logger.info(f"Processing {component_name} features...")
                
                try:
                    result = component.create_features(result)
                    component_time = time.time() - component_start
                    processing_stats[component_name] = {
                        'processing_time': component_time,
                        'features_added': len(component.get_feature_names()),
                        'success': True
                    }
                    
                    self.logger.info(
                        f"{component_name} completed in {component_time:.2f}s, "
                        f"added {len(component.get_feature_names())} features"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Error in {component_name} processing: {e}")
                    processing_stats[component_name] = {
                        'processing_time': time.time() - component_start,
                        'features_added': 0,
                        'success': False,
                        'error': str(e)
                    }
            
            # Apply feature selection if enabled
            if 'selector' in self.components:
                selector_start = time.time()
                self.logger.info("Applying feature selection...")
                
                try:
                    result = self.components['selector'].create_features(result)
                    selector_time = time.time() - selector_start
                    
                    selected_count = len(self.components['selector'].get_feature_names())
                    processing_stats['selector'] = {
                        'processing_time': selector_time,
                        'features_selected': selected_count,
                        'success': True
                    }
                    
                    self.logger.info(f"Feature selection completed, {selected_count} features selected")
                    
                except Exception as e:
                    self.logger.error(f"Error in feature selection: {e}")
                    processing_stats['selector'] = {
                        'processing_time': time.time() - selector_start,
                        'features_selected': 0,
                        'success': False,
                        'error': str(e)
                    }
            
            total_time = time.time() - start_time
            self.logger.info(f"Pipeline processing completed in {total_time:.2f}s")
            
            # Store processing statistics
            if self.config.performance_monitoring:
                self._log_processing_stats(processing_stats, total_time, result.shape)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Pipeline processing failed: {e}")
            raise FeatureEngineeringError(f"Feature pipeline failed: {e}")
    
    def _log_processing_stats(self, stats: Dict[str, Any], total_time: float, 
                            final_shape: tuple) -> None:
        """Log detailed processing statistics."""
        self.logger.info("=== FEATURE PIPELINE STATISTICS ===")
        self.logger.info(f"Total processing time: {total_time:.2f}s")
        self.logger.info(f"Final dataset shape: {final_shape}")
        
        for component, component_stats in stats.items():
            if component_stats['success']:
                if 'features_added' in component_stats:
                    self.logger.info(
                        f"{component}: {component_stats['features_added']} features "
                        f"in {component_stats['processing_time']:.2f}s"
                    )
                elif 'features_selected' in component_stats:
                    self.logger.info(
                        f"{component}: {component_stats['features_selected']} features selected "
                        f"in {component_stats['processing_time']:.2f}s"
                    )
            else:
                self.logger.error(f"{component}: FAILED - {component_stats.get('error', 'Unknown error')}")
        
        self.logger.info("=== END STATISTICS ===")


class FeatureEngineeringEngine(FeatureEngineer):
    """
    Main feature engineering orchestrator.
    
    Coordinates all feature engineering components and provides unified interface.
    Implements the Facade pattern for the complete feature engineering subsystem.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.logger = get_logger("FeatureEngineeringEngine")
        
        # Initialize pipeline
        self.pipeline = FeaturePipeline(self.config)
        
        # Feature tracking
        self._feature_names = None
        self._feature_importance = None
        self._processing_stats = {}
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive features using all enabled components.
        
        Args:
            data: DataFrame with match data
            
        Returns:
            DataFrame with all engineered features
        """
        try:
            self.logger.info(f"Creating features for {len(data)} matches")
            
            if data.empty:
                raise FeatureEngineeringError("Empty dataset provided")
            
            # Validate input data
            if self.config.feature_validation:
                self._validate_input_data(data)
            
            # Process through pipeline
            result = self.pipeline.process(data)
            
            # Update feature tracking
            self._update_feature_tracking()
            
            # Validate output
            if self.config.feature_validation:
                self._validate_output_data(result)
            
            self.logger.info(f"Feature engineering completed: {result.shape[1]} total columns")
            return result
            
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}")
            raise FeatureEngineeringError(f"Feature engineering failed: {e}")
    
    def get_feature_names(self) -> List[str]:
        """Return list of all feature names created by the engine."""
        if self._feature_names is None:
            self._update_feature_tracking()
        return self._feature_names or []
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Return combined feature importance scores."""
        if self._feature_importance is None:
            self._update_feature_tracking()
        return self._feature_importance
    
    def get_component_features(self, component_name: str) -> List[str]:
        """Get feature names for a specific component."""
        if component_name in self.pipeline.components:
            component = self.pipeline.components[component_name]
            return component.get_feature_names()
        return []
    
    def get_processing_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive processing report.
        
        Returns:
            Dict with processing statistics and feature information
        """
        report = {
            'configuration': {
                'temporal_enabled': self.config.enable_temporal,
                'contextual_enabled': self.config.enable_contextual,
                'advanced_enabled': self.config.enable_advanced,
                'selection_enabled': self.config.enable_selection
            },
            'feature_counts': {},
            'processing_stats': self._processing_stats,
            'total_features': len(self.get_feature_names()),
            'feature_importance_available': self._feature_importance is not None
        }
        
        # Get feature counts by component
        for component_name in self.pipeline.components:
            feature_count = len(self.get_component_features(component_name))
            report['feature_counts'][component_name] = feature_count
        
        # Add selection report if available
        if 'selector' in self.pipeline.components:
            selector = self.pipeline.components['selector']
            if hasattr(selector, 'get_selection_report'):
                report['selection_report'] = selector.get_selection_report()
        
        return report
    
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """Validate input data requirements."""
        required_columns = ['date', 'home_team', 'away_team', 'home_score', 'away_score']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise FeatureEngineeringError(f"Missing required columns: {missing_columns}")
        
        # Check for minimum data requirements
        if len(data) < 10:
            self.logger.warning(f"Very small dataset: {len(data)} matches")
        
        # Check date format
        try:
            pd.to_datetime(data['date'])
        except Exception as e:
            raise FeatureEngineeringError(f"Invalid date format in 'date' column: {e}")
    
    def _validate_output_data(self, data: pd.DataFrame) -> None:
        """Validate output data quality."""
        # Check for excessive missing values
        missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        if missing_ratio > 0.5:
            self.logger.warning(f"High missing value ratio in output: {missing_ratio:.2%}")
        
        # Check for infinite values
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        inf_counts = np.isinf(data[numeric_cols]).sum().sum()
        if inf_counts > 0:
            self.logger.warning(f"Found {inf_counts} infinite values in output")
        
        # Check feature variance
        low_variance_features = []
        for col in numeric_cols:
            if data[col].var() < 1e-10:
                low_variance_features.append(col)
        
        if low_variance_features:
            self.logger.warning(f"Found {len(low_variance_features)} low-variance features")
    
    def _update_feature_tracking(self) -> None:
        """Update internal feature tracking information."""
        all_feature_names = []
        combined_importance = {}
        
        # Collect feature names and importance from all components
        for component_name, component in self.pipeline.components.items():
            if component_name == 'selector':
                # For selector, get selected features
                selected_features = component.get_feature_names()
                all_feature_names.extend(selected_features)
                
                # Get importance for selected features
                importance = component.get_feature_importance()
                if importance:
                    combined_importance.update(importance)
            else:
                # For other components, get all features
                component_features = component.get_feature_names()
                
                # If selector is enabled, only include selected features
                if 'selector' in self.pipeline.components:
                    selector = self.pipeline.components['selector']
                    selected_features = selector.get_feature_names()
                    component_features = [f for f in component_features if f in selected_features]
                
                all_feature_names.extend(component_features)
                
                # Get importance scores
                importance = component.get_feature_importance()
                if importance:
                    for feature in component_features:
                        if feature in importance:
                            combined_importance[feature] = importance[feature]
        
        self._feature_names = list(set(all_feature_names))  # Remove duplicates
        self._feature_importance = combined_importance if combined_importance else None
    
    def create_feature_summary(self) -> pd.DataFrame:
        """
        Create a summary DataFrame of all features with their properties.
        
        Returns:
            DataFrame with feature information
        """
        feature_names = self.get_feature_names()
        importance_scores = self.get_feature_importance() or {}
        
        summary_data = []
        
        for feature in feature_names:
            # Determine feature category
            category = 'Unknown'
            if any(pattern in feature.lower() for pattern in ['rolling', 'momentum', 'form', 'streak']):
                category = 'Temporal'
            elif any(pattern in feature.lower() for pattern in ['home_', 'h2h_', 'rivalry', 'travel', 'seasonal']):
                category = 'Contextual'
            elif any(pattern in feature.lower() for pattern in ['xg_', 'eff_', 'clutch_', 'style_']):
                category = 'Advanced'
            
            # Get importance score
            importance = importance_scores.get(feature, 0.0)
            
            summary_data.append({
                'feature_name': feature,
                'category': category,
                'importance_score': importance,
                'importance_rank': 0  # Will be filled after sorting
            })
        
        # Create DataFrame and add ranking
        summary_df = pd.DataFrame(summary_data)
        if not summary_df.empty:
            summary_df = summary_df.sort_values('importance_score', ascending=False)
            summary_df['importance_rank'] = range(1, len(summary_df) + 1)
        
        return summary_df