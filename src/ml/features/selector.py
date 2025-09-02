"""
Feature selection components for soccer match analysis.

Provides automated feature selection including importance analysis,
correlation analysis, and redundancy removal.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime
from dataclasses import dataclass
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

from src.config.logging_config import get_logger
from ..core.interfaces import FeatureEngineer
from ..core.exceptions import FeatureEngineeringError


@dataclass
class FeatureSelectorConfig:
    """Configuration for feature selection."""
    importance_threshold: float = 0.01
    correlation_threshold: float = 0.95
    max_features: int = 100
    min_features: int = 20
    variance_threshold: float = 0.01
    mutual_info_threshold: float = 0.1
    
    def __post_init__(self):
        pass


class ImportanceAnalyzer:
    """
    Analyzes feature importance using multiple methods.
    
    Following Single Responsibility Principle - focused only on importance analysis.
    """
    
    def __init__(self, importance_threshold: float = 0.01):
        self.importance_threshold = importance_threshold
        self.logger = get_logger("ImportanceAnalyzer")
    
    def analyze_feature_importance(self, 
                                 features: pd.DataFrame,
                                 targets: pd.Series,
                                 feature_names: List[str]) -> Dict[str, float]:
        """
        Analyze feature importance using multiple methods.
        
        Returns:
            Dict with importance scores for each feature
        """
        try:
            self.logger.info(f"Analyzing importance for {len(feature_names)} features")
            
            # Ensure features have correct column names
            if features.shape[1] != len(feature_names):
                self.logger.warning(f"Feature count mismatch: {features.shape[1]} vs {len(feature_names)}")
                feature_names = feature_names[:features.shape[1]]
            
            features.columns = feature_names
            
            # Remove features with no variance
            variance_mask = features.var() > self.importance_threshold
            valid_features = features.loc[:, variance_mask]
            valid_feature_names = [name for name, valid in zip(feature_names, variance_mask) if valid]
            
            if len(valid_features.columns) == 0:
                self.logger.error("No valid features found after variance filtering")
                return {name: 0.0 for name in feature_names}
            
            # Convert targets to categorical for classification
            y_categorical = self._prepare_targets(targets)
            
            importance_scores = {}
            
            # Method 1: Random Forest importance
            rf_scores = self._calculate_rf_importance(valid_features, y_categorical, valid_feature_names)
            
            # Method 2: Mutual information
            mi_scores = self._calculate_mutual_info(valid_features, y_categorical, valid_feature_names)
            
            # Method 3: F-score (ANOVA)
            f_scores = self._calculate_f_scores(valid_features, y_categorical, valid_feature_names)
            
            # Method 4: Lasso regularization
            lasso_scores = self._calculate_lasso_importance(valid_features, y_categorical, valid_feature_names)
            
            # Combine importance scores
            for feature in valid_feature_names:
                combined_score = (
                    rf_scores.get(feature, 0) * 0.4 +
                    mi_scores.get(feature, 0) * 0.3 +
                    f_scores.get(feature, 0) * 0.2 +
                    lasso_scores.get(feature, 0) * 0.1
                )
                importance_scores[feature] = combined_score
            
            # Add zero scores for removed features
            for feature in feature_names:
                if feature not in importance_scores:
                    importance_scores[feature] = 0.0
            
            # Normalize scores
            max_score = max(importance_scores.values()) if importance_scores else 1.0
            if max_score > 0:
                importance_scores = {k: v / max_score for k, v in importance_scores.items()}
            
            self.logger.info(f"Analyzed importance for {len(importance_scores)} features")
            return importance_scores
            
        except Exception as e:
            self.logger.error(f"Error analyzing feature importance: {e}")
            return {name: 0.5 for name in feature_names}  # Default importance
    
    def _prepare_targets(self, targets: pd.Series) -> pd.Series:
        """Convert targets to categorical format."""
        if targets.dtype == 'object':
            # Already categorical
            return targets
        
        # Convert numeric targets to categories (Win/Draw/Loss based on goal difference)
        categorical_targets = []
        for target in targets:
            if isinstance(target, (int, float)):
                if target > 0:
                    categorical_targets.append('Win')
                elif target == 0:
                    categorical_targets.append('Draw')
                else:
                    categorical_targets.append('Loss')
            else:
                categorical_targets.append(str(target))
        
        return pd.Series(categorical_targets)
    
    def _calculate_rf_importance(self, features: pd.DataFrame, 
                               targets: pd.Series, 
                               feature_names: List[str]) -> Dict[str, float]:
        """Calculate Random Forest feature importance."""
        try:
            # Handle missing values
            features_clean = features.fillna(0)
            
            rf = RandomForestClassifier(
                n_estimators=50, 
                random_state=42, 
                max_depth=10,
                min_samples_split=10,
                n_jobs=-1
            )
            
            rf.fit(features_clean, targets)
            importances = rf.feature_importances_
            
            return dict(zip(feature_names, importances))
            
        except Exception as e:
            self.logger.error(f"Error calculating RF importance: {e}")
            return {name: 0.1 for name in feature_names}
    
    def _calculate_mutual_info(self, features: pd.DataFrame, 
                             targets: pd.Series, 
                             feature_names: List[str]) -> Dict[str, float]:
        """Calculate mutual information scores."""
        try:
            features_clean = features.fillna(0)
            
            mi_scores = mutual_info_classif(
                features_clean, targets, 
                discrete_features=False, 
                random_state=42
            )
            
            # Normalize scores
            max_score = max(mi_scores) if len(mi_scores) > 0 and max(mi_scores) > 0 else 1.0
            normalized_scores = mi_scores / max_score
            
            return dict(zip(feature_names, normalized_scores))
            
        except Exception as e:
            self.logger.error(f"Error calculating mutual information: {e}")
            return {name: 0.1 for name in feature_names}
    
    def _calculate_f_scores(self, features: pd.DataFrame, 
                          targets: pd.Series, 
                          feature_names: List[str]) -> Dict[str, float]:
        """Calculate F-scores (ANOVA)."""
        try:
            features_clean = features.fillna(0)
            
            selector = SelectKBest(f_classif, k='all')
            selector.fit(features_clean, targets)
            
            f_scores = selector.scores_
            
            # Handle potential NaN values
            f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Normalize scores
            max_score = max(f_scores) if len(f_scores) > 0 and max(f_scores) > 0 else 1.0
            normalized_scores = f_scores / max_score
            
            return dict(zip(feature_names, normalized_scores))
            
        except Exception as e:
            self.logger.error(f"Error calculating F-scores: {e}")
            return {name: 0.1 for name in feature_names}
    
    def _calculate_lasso_importance(self, features: pd.DataFrame, 
                                  targets: pd.Series, 
                                  feature_names: List[str]) -> Dict[str, float]:
        """Calculate Lasso regularization importance."""
        try:
            from sklearn.preprocessing import LabelEncoder
            
            features_clean = features.fillna(0)
            
            # Encode categorical targets
            le = LabelEncoder()
            targets_encoded = le.fit_transform(targets)
            
            lasso = LassoCV(cv=3, random_state=42, max_iter=1000)
            lasso.fit(features_clean, targets_encoded)
            
            # Use absolute coefficients as importance
            coefficients = np.abs(lasso.coef_)
            
            # Normalize scores
            max_coef = max(coefficients) if len(coefficients) > 0 and max(coefficients) > 0 else 1.0
            normalized_scores = coefficients / max_coef
            
            return dict(zip(feature_names, normalized_scores))
            
        except Exception as e:
            self.logger.error(f"Error calculating Lasso importance: {e}")
            return {name: 0.1 for name in feature_names}


class CorrelationAnalyzer:
    """
    Analyzes feature correlations to identify redundant features.
    
    Focuses on correlation detection and multicollinearity analysis.
    """
    
    def __init__(self, correlation_threshold: float = 0.95):
        self.correlation_threshold = correlation_threshold
        self.logger = get_logger("CorrelationAnalyzer")
    
    def analyze_correlations(self, features: pd.DataFrame, feature_names: List[str]) -> Dict[str, Any]:
        """
        Analyze feature correlations and identify redundant features.
        
        Returns:
            Dict with correlation matrix and redundant feature recommendations
        """
        try:
            self.logger.info(f"Analyzing correlations for {len(feature_names)} features")
            
            # Ensure features have correct column names
            if features.shape[1] != len(feature_names):
                feature_names = feature_names[:features.shape[1]]
            
            features.columns = feature_names
            
            # Handle missing values
            features_clean = features.fillna(0)
            
            # Calculate correlation matrix
            corr_matrix = features_clean.corr()
            
            # Find highly correlated pairs
            high_corr_pairs = self._find_high_correlation_pairs(corr_matrix)
            
            # Identify redundant features
            redundant_features = self._identify_redundant_features(corr_matrix, high_corr_pairs)
            
            # Calculate feature clusters
            feature_clusters = self._create_feature_clusters(corr_matrix)
            
            # Analyze multicollinearity
            multicollinearity_scores = self._calculate_multicollinearity(features_clean)
            
            return {
                'correlation_matrix': corr_matrix,
                'high_correlation_pairs': high_corr_pairs,
                'redundant_features': redundant_features,
                'feature_clusters': feature_clusters,
                'multicollinearity_scores': multicollinearity_scores,
                'recommended_removals': list(redundant_features)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing correlations: {e}")
            return {
                'correlation_matrix': pd.DataFrame(),
                'high_correlation_pairs': [],
                'redundant_features': set(),
                'feature_clusters': [],
                'multicollinearity_scores': {},
                'recommended_removals': []
            }
    
    def _find_high_correlation_pairs(self, corr_matrix: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """Find pairs of features with high correlation."""
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                feature1 = corr_matrix.columns[i]
                feature2 = corr_matrix.columns[j]
                correlation = corr_matrix.iloc[i, j]
                
                if abs(correlation) >= self.correlation_threshold:
                    high_corr_pairs.append((feature1, feature2, correlation))
        
        # Sort by correlation strength
        high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        return high_corr_pairs
    
    def _identify_redundant_features(self, corr_matrix: pd.DataFrame, 
                                   high_corr_pairs: List[Tuple[str, str, float]]) -> Set[str]:
        """Identify features that are redundant due to high correlation."""
        redundant_features = set()
        processed_features = set()
        
        for feature1, feature2, correlation in high_corr_pairs:
            if feature1 in processed_features and feature2 in processed_features:
                continue
            
            # Keep the feature with higher variance (more informative)
            var1 = corr_matrix[feature1].var()
            var2 = corr_matrix[feature2].var()
            
            if var1 >= var2:
                redundant_features.add(feature2)
                processed_features.add(feature2)
            else:
                redundant_features.add(feature1)
                processed_features.add(feature1)
            
            processed_features.add(feature1)
            processed_features.add(feature2)
        
        return redundant_features
    
    def _create_feature_clusters(self, corr_matrix: pd.DataFrame) -> List[List[str]]:
        """Create clusters of related features."""
        features = list(corr_matrix.columns)
        clusters = []
        processed = set()
        
        for feature in features:
            if feature in processed:
                continue
            
            # Find all features highly correlated with this one
            cluster = [feature]
            for other_feature in features:
                if (other_feature != feature and 
                    other_feature not in processed and
                    abs(corr_matrix.loc[feature, other_feature]) >= self.correlation_threshold * 0.8):
                    cluster.append(other_feature)
            
            if len(cluster) > 1:
                clusters.append(cluster)
                processed.update(cluster)
        
        return clusters
    
    def _calculate_multicollinearity(self, features: pd.DataFrame) -> Dict[str, float]:
        """Calculate multicollinearity scores (VIF approximation)."""
        multicollinearity_scores = {}
        
        try:
            # Simple approximation of VIF using R-squared
            for feature in features.columns:
                other_features = [col for col in features.columns if col != feature]
                
                if len(other_features) == 0:
                    multicollinearity_scores[feature] = 1.0
                    continue
                
                # Calculate correlation with all other features
                correlations = []
                for other_feature in other_features:
                    corr = features[feature].corr(features[other_feature])
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
                
                # Approximate VIF using maximum correlation
                max_corr = max(correlations) if correlations else 0.0
                vif_approx = 1 / max(1 - max_corr**2, 0.01)  # Avoid division by zero
                
                multicollinearity_scores[feature] = min(vif_approx, 10.0)  # Cap at 10
                
        except Exception as e:
            self.logger.error(f"Error calculating multicollinearity: {e}")
            # Return default scores
            for feature in features.columns:
                multicollinearity_scores[feature] = 2.0
        
        return multicollinearity_scores


class RedundancyRemover:
    """
    Removes redundant features based on multiple criteria.
    
    Combines importance, correlation, and domain knowledge to select optimal features.
    """
    
    def __init__(self, max_features: int = 100, min_features: int = 20):
        self.max_features = max_features
        self.min_features = min_features
        self.logger = get_logger("RedundancyRemover")
    
    def remove_redundant_features(self, 
                                features: pd.DataFrame,
                                importance_scores: Dict[str, float],
                                correlation_analysis: Dict[str, Any],
                                feature_names: List[str]) -> List[str]:
        """
        Remove redundant features using multiple criteria.
        
        Returns:
            List of selected feature names
        """
        try:
            self.logger.info(f"Removing redundancy from {len(feature_names)} features")
            
            # Start with all features
            selected_features = set(feature_names)
            
            # Step 1: Remove features with very low importance
            low_importance_threshold = 0.01
            low_importance_features = {
                feature for feature, score in importance_scores.items() 
                if score < low_importance_threshold
            }
            selected_features -= low_importance_features
            self.logger.info(f"Removed {len(low_importance_features)} low-importance features")
            
            # Step 2: Remove highly correlated features
            redundant_features = set(correlation_analysis.get('redundant_features', []))
            selected_features -= redundant_features
            self.logger.info(f"Removed {len(redundant_features)} redundant features")
            
            # Step 3: Remove features with high multicollinearity
            multicollinearity_scores = correlation_analysis.get('multicollinearity_scores', {})
            high_multicollinearity_features = {
                feature for feature, score in multicollinearity_scores.items() 
                if score > 5.0  # VIF > 5 indicates multicollinearity
            }
            # Only remove if we still have enough features
            if len(selected_features) - len(high_multicollinearity_features) >= self.min_features:
                selected_features -= high_multicollinearity_features
                self.logger.info(f"Removed {len(high_multicollinearity_features)} multicollinear features")
            
            # Step 4: Apply feature importance ranking if we still have too many
            if len(selected_features) > self.max_features:
                # Rank remaining features by importance
                remaining_with_scores = [
                    (feature, importance_scores.get(feature, 0)) 
                    for feature in selected_features
                ]
                remaining_with_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Keep top features
                selected_features = {
                    feature for feature, _ in remaining_with_scores[:self.max_features]
                }
                self.logger.info(f"Selected top {len(selected_features)} features by importance")
            
            # Step 5: Ensure minimum features
            if len(selected_features) < self.min_features:
                # Add back highest importance features
                all_features_by_importance = [
                    (feature, importance_scores.get(feature, 0)) 
                    for feature in feature_names
                ]
                all_features_by_importance.sort(key=lambda x: x[1], reverse=True)
                
                for feature, _ in all_features_by_importance:
                    selected_features.add(feature)
                    if len(selected_features) >= self.min_features:
                        break
                
                self.logger.info(f"Added features to reach minimum of {self.min_features}")
            
            # Step 6: Apply domain knowledge priorities
            selected_features = self._apply_domain_priorities(selected_features, importance_scores)
            
            final_features = list(selected_features)
            self.logger.info(f"Final selection: {len(final_features)} features")
            
            return final_features
            
        except Exception as e:
            self.logger.error(f"Error removing redundant features: {e}")
            # Return top features by importance as fallback
            sorted_features = sorted(
                feature_names, 
                key=lambda x: importance_scores.get(x, 0), 
                reverse=True
            )
            return sorted_features[:min(self.max_features, len(sorted_features))]
    
    def _apply_domain_priorities(self, selected_features: Set[str], 
                               importance_scores: Dict[str, float]) -> Set[str]:
        """Apply domain knowledge to prioritize certain types of features."""
        # High-priority feature patterns
        high_priority_patterns = [
            'home_advantage', 'h2h_', 'xg_', 'form_', 'momentum_',
            'rivalry_intensity', 'recent_', 'rolling_'
        ]
        
        # Ensure we keep high-priority features
        priority_features = set()
        for feature in selected_features:
            for pattern in high_priority_patterns:
                if pattern in feature.lower():
                    priority_features.add(feature)
                    break
        
        # Low-priority feature patterns that can be removed if needed
        low_priority_patterns = [
            'seasonal_month', 'travel_distance', 'style_tempo', 'eff_adaptability'
        ]
        
        # If we need to make room, remove low-priority features first
        if len(selected_features) > self.max_features:
            removable_features = set()
            for feature in selected_features:
                for pattern in low_priority_patterns:
                    if pattern in feature.lower():
                        removable_features.add(feature)
                        break
            
            # Remove lowest importance among removable features
            removable_with_scores = [
                (feature, importance_scores.get(feature, 0)) 
                for feature in removable_features
            ]
            removable_with_scores.sort(key=lambda x: x[1])
            
            features_to_remove = len(selected_features) - self.max_features
            for feature, _ in removable_with_scores[:features_to_remove]:
                selected_features.discard(feature)
        
        return selected_features


class FeatureSelector(FeatureEngineer):
    """
    Main orchestrator for feature selection.
    
    Combines importance analysis, correlation analysis, and redundancy removal.
    Follows Facade pattern to provide unified interface to feature selection.
    """
    
    def __init__(self, config: Optional[FeatureSelectorConfig] = None):
        self.config = config or FeatureSelectorConfig()
        self.logger = get_logger("FeatureSelector")
        
        # Initialize components
        self.importance_analyzer = ImportanceAnalyzer(self.config.importance_threshold)
        self.correlation_analyzer = CorrelationAnalyzer(self.config.correlation_threshold)
        self.redundancy_remover = RedundancyRemover(self.config.max_features, self.config.min_features)
        
        self._selected_features = None
        self._importance_scores = None
        self._correlation_analysis = None
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        This method performs feature selection rather than creation.
        
        Args:
            data: DataFrame with all features and targets
            
        Returns:
            DataFrame with only selected features
        """
        try:
            self.logger.info(f"Selecting features from {data.shape[1]} columns")
            
            if data.empty:
                raise FeatureEngineeringError("Empty dataset provided")
            
            # Identify feature and target columns
            target_columns = ['home_score', 'away_score', 'result', 'home_team', 'away_team', 'date']
            feature_columns = [col for col in data.columns if col not in target_columns]
            
            if len(feature_columns) == 0:
                raise FeatureEngineeringError("No feature columns found")
            
            # Prepare features and targets
            features = data[feature_columns].copy()
            
            # Create targets for analysis (goal difference)
            if 'home_score' in data.columns and 'away_score' in data.columns:
                targets = data['home_score'] - data['away_score']
            else:
                # If no scores, create dummy targets
                targets = pd.Series([0] * len(data))
            
            # Perform feature selection
            selected_feature_names = self.select_features(features, targets, feature_columns)
            
            # Return data with selected features plus essential columns
            essential_columns = [col for col in target_columns if col in data.columns]
            result_columns = essential_columns + selected_feature_names
            
            result = data[result_columns].copy()
            
            self.logger.info(f"Selected {len(selected_feature_names)} features")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in feature selection: {e}")
            raise FeatureEngineeringError(f"Feature selection failed: {e}")
    
    def select_features(self, 
                       features: pd.DataFrame, 
                       targets: pd.Series, 
                       feature_names: List[str]) -> List[str]:
        """
        Select optimal features using comprehensive analysis.
        
        Returns:
            List of selected feature names
        """
        try:
            self.logger.info("Starting comprehensive feature selection")
            
            # Step 1: Analyze feature importance
            self._importance_scores = self.importance_analyzer.analyze_feature_importance(
                features, targets, feature_names
            )
            
            # Step 2: Analyze correlations
            self._correlation_analysis = self.correlation_analyzer.analyze_correlations(
                features, feature_names
            )
            
            # Step 3: Remove redundant features
            selected_features = self.redundancy_remover.remove_redundant_features(
                features, self._importance_scores, self._correlation_analysis, feature_names
            )
            
            # Step 4: Final validation and optimization
            selected_features = self._validate_selection(selected_features, features, targets)
            
            self._selected_features = selected_features
            
            self.logger.info(f"Feature selection complete: {len(selected_features)} features selected")
            return selected_features
            
        except Exception as e:
            self.logger.error(f"Error selecting features: {e}")
            # Return top features by name as fallback
            return feature_names[:min(50, len(feature_names))]
    
    def _validate_selection(self, selected_features: List[str], 
                          features: pd.DataFrame, 
                          targets: pd.Series) -> List[str]:
        """Validate and optimize the feature selection."""
        try:
            # Ensure we have a reasonable number of features
            if len(selected_features) < self.config.min_features:
                self.logger.warning(f"Too few features selected: {len(selected_features)}")
                # Add highest importance features
                all_features_by_importance = sorted(
                    self._importance_scores.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                for feature, _ in all_features_by_importance:
                    if feature not in selected_features:
                        selected_features.append(feature)
                    if len(selected_features) >= self.config.min_features:
                        break
            
            # Remove features that don't exist in the data
            valid_features = [f for f in selected_features if f in features.columns]
            
            if len(valid_features) != len(selected_features):
                self.logger.warning(f"Removed {len(selected_features) - len(valid_features)} invalid features")
            
            return valid_features
            
        except Exception as e:
            self.logger.error(f"Error validating selection: {e}")
            return selected_features
    
    def get_feature_names(self) -> List[str]:
        """Return list of selected feature names."""
        if self._selected_features is None:
            return []
        return self._selected_features
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Return importance scores for selected features."""
        if self._importance_scores is None or self._selected_features is None:
            return None
        
        return {
            feature: self._importance_scores.get(feature, 0.0) 
            for feature in self._selected_features
        }
    
    def get_selection_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive feature selection report.
        
        Returns:
            Dict with selection statistics and analysis
        """
        if self._selected_features is None:
            return {}
        
        report = {
            'total_features_analyzed': len(self._importance_scores) if self._importance_scores else 0,
            'features_selected': len(self._selected_features),
            'selection_ratio': len(self._selected_features) / max(len(self._importance_scores), 1),
            'average_importance': np.mean([
                self._importance_scores.get(f, 0) for f in self._selected_features
            ]) if self._importance_scores else 0,
            'top_features': sorted(
                [(f, self._importance_scores.get(f, 0)) for f in self._selected_features],
                key=lambda x: x[1], reverse=True
            )[:10] if self._importance_scores else [],
            'correlation_analysis': self._correlation_analysis,
            'redundant_features_removed': len(
                self._correlation_analysis.get('redundant_features', [])
            ) if self._correlation_analysis else 0
        }
        
        return report