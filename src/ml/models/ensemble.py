"""
Ensemble methods and meta-learning for soccer outcome prediction.

Combines multiple predictors using stacking, weighted voting, and
meta-learning approaches for optimal performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import pickle
from pathlib import Path

from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder

from src.config.logging_config import get_logger
from ..core.interfaces import Predictor, ModelTrainer
from ..core.exceptions import ModelTrainingError, ModelPredictionError
from ..core.types import MatchFeatures, Prediction, ModelMetrics

from .predictors import XGBoostPredictor, LightGBMPredictor, RandomForestPredictor


class MetaLearner:
    """
    Meta-learning component that learns when to trust each model.
    
    Uses model performance history and confidence scores to dynamically
    weight predictions from different models.
    """
    
    def __init__(self, base_models: List[Predictor], random_state: int = 42):
        self.base_models = base_models
        self.random_state = random_state
        self.meta_model = LogisticRegression(random_state=random_state)
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.model_weights = None
        self.logger = get_logger("MetaLearner")
        
    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        """Train meta-learner using base model predictions as features."""
        try:
            self.logger.info("Training meta-learner with base model predictions")
            
            # Prepare targets
            y_encoded = self.label_encoder.fit_transform(targets)
            
            # Generate meta-features using cross-validation
            meta_features = self._generate_meta_features(features, y_encoded)
            
            # Train meta-model
            self.meta_model.fit(meta_features, y_encoded)
            self.is_trained = True
            
            # Calculate model weights based on individual performance
            self._calculate_model_weights(features, y_encoded)
            
            self.logger.info("Meta-learner training completed")
            
        except Exception as e:
            self.logger.error(f"Meta-learner training failed: {e}")
            raise ModelTrainingError(f"Meta-learner training failed: {e}")
    
    def predict_meta(self, features: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Generate meta-prediction with model confidence scores.
        
        Returns:
            Tuple of (prediction_probabilities, model_confidences)
        """
        if not self.is_trained:
            raise ModelPredictionError("Meta-learner must be trained first")
        
        try:
            # Get predictions from all base models
            base_predictions = []
            model_confidences = {}
            
            for i, model in enumerate(self.base_models):
                if model.is_trained:
                    prediction = model.predict(features.iloc[0])  # Single prediction
                    base_predictions.extend([
                        prediction.home_win_prob,
                        prediction.draw_prob, 
                        prediction.away_win_prob,
                        prediction.confidence
                    ])
                    model_confidences[model.model_name] = prediction.confidence
                else:
                    # Use default values for untrained models
                    base_predictions.extend([0.33, 0.33, 0.33, 0.5])
                    model_confidences[model.model_name] = 0.5
            
            # Create meta-features
            meta_features = np.array(base_predictions).reshape(1, -1)
            
            # Get meta-prediction
            meta_proba = self.meta_model.predict_proba(meta_features)[0]
            
            return meta_proba, model_confidences
            
        except Exception as e:
            self.logger.error(f"Meta-prediction failed: {e}")
            raise ModelPredictionError(f"Meta-prediction failed: {e}")
    
    def _generate_meta_features(self, features: pd.DataFrame, targets: np.ndarray) -> np.ndarray:
        """Generate meta-features using cross-validated predictions."""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        meta_features = []
        
        for model in self.base_models:
            if hasattr(model, 'model') and model.model is not None:
                # Use existing trained model for cross-validation
                try:
                    cv_predictions = cross_val_predict(
                        model.model, features.fillna(0), targets,
                        cv=cv, method='predict_proba'
                    )
                    
                    # Add prediction probabilities and confidence
                    meta_features.append(cv_predictions)
                    meta_features.append(np.max(cv_predictions, axis=1).reshape(-1, 1))
                    
                except Exception as e:
                    self.logger.warning(f"Failed to generate CV predictions for {model.model_name}: {e}")
                    # Use default predictions
                    n_classes = len(np.unique(targets))
                    default_proba = np.ones((len(features), n_classes)) / n_classes
                    meta_features.append(default_proba)
                    meta_features.append(np.ones((len(features), 1)) * 0.5)
            else:
                # Model not trained - use default features
                n_classes = len(np.unique(targets))
                default_proba = np.ones((len(features), n_classes)) / n_classes
                meta_features.append(default_proba)
                meta_features.append(np.ones((len(features), 1)) * 0.5)
        
        return np.concatenate(meta_features, axis=1)
    
    def _calculate_model_weights(self, features: pd.DataFrame, targets: np.ndarray) -> None:
        """Calculate dynamic weights for each base model."""
        self.model_weights = {}
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        for model in self.base_models:
            if hasattr(model, 'model') and model.model is not None:
                try:
                    cv_scores = cross_val_score(
                        model.model, features.fillna(0), targets,
                        cv=cv, scoring='accuracy'
                    )
                    weight = cv_scores.mean()
                except Exception:
                    weight = 0.33  # Default weight
            else:
                weight = 0.33  # Default weight for untrained models
            
            self.model_weights[model.model_name] = weight
        
        # Normalize weights
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            self.model_weights = {k: v/total_weight for k, v in self.model_weights.items()}


class ModelSelector:
    """
    Dynamic model selector that chooses the best model for each prediction.
    
    Uses context-aware selection based on match characteristics and
    historical model performance.
    """
    
    def __init__(self, base_models: List[Predictor]):
        self.base_models = base_models
        self.selection_rules = {}
        self.performance_history = {}
        self.logger = get_logger("ModelSelector")
        
    def select_best_model(self, features: pd.DataFrame, context: Dict[str, Any] = None) -> Predictor:
        """
        Select the best model based on match context and historical performance.
        
        Args:
            features: Match features for prediction
            context: Additional context (e.g., league, teams, etc.)
            
        Returns:
            Selected predictor model
        """
        try:
            # Default to first available trained model
            best_model = None
            best_score = -1
            
            for model in self.base_models:
                if model.is_trained:
                    # Calculate selection score based on multiple factors
                    score = self._calculate_selection_score(model, features, context)
                    
                    if score > best_score:
                        best_score = score
                        best_model = model
            
            if best_model is None:
                # No trained models available
                raise ModelPredictionError("No trained models available for selection")
            
            self.logger.debug(f"Selected {best_model.model_name} with score {best_score:.3f}")
            return best_model
            
        except Exception as e:
            self.logger.error(f"Model selection failed: {e}")
            # Fall back to first trained model
            for model in self.base_models:
                if model.is_trained:
                    return model
            raise ModelPredictionError("No trained models available")
    
    def _calculate_selection_score(self, model: Predictor, 
                                 features: pd.DataFrame, 
                                 context: Dict[str, Any] = None) -> float:
        """Calculate selection score for a model."""
        score = 0.5  # Base score
        
        # Historical performance weight
        if model.model_name in self.performance_history:
            historical_perf = self.performance_history[model.model_name]
            score += historical_perf.get('accuracy', 0.5) * 0.4
        
        # Model-specific bonuses based on data characteristics
        if hasattr(model, 'training_metrics'):
            metrics = model.training_metrics
            
            # XGBoost excels with many features
            if 'XGBoost' in model.model_name and len(features.columns) > 50:
                score += 0.1
            
            # LightGBM is fast and works well with sparse data
            if 'LightGBM' in model.model_name:
                sparsity = features.isnull().sum().sum() / (features.shape[0] * features.shape[1])
                if sparsity > 0.1:  # Sparse data
                    score += 0.1
            
            # RandomForest is robust to outliers
            if 'RandomForest' in model.model_name:
                score += 0.05  # Generally robust baseline
        
        # Context-specific adjustments
        if context:
            # Add context-specific selection logic here
            pass
        
        return min(max(score, 0.0), 1.0)  # Clamp between 0 and 1
    
    def update_performance_history(self, model_name: str, performance_metrics: Dict[str, float]) -> None:
        """Update performance history for a model."""
        self.performance_history[model_name] = performance_metrics
        self.logger.debug(f"Updated performance history for {model_name}")


class EnsemblePredictor(Predictor, ModelTrainer):
    """
    Advanced ensemble predictor combining multiple models with meta-learning.
    
    Implements stacking, weighted voting, and dynamic model selection
    for optimal prediction performance.
    """
    
    def __init__(self, 
                 base_models: List[Predictor] = None,
                 ensemble_method: str = 'stacking',
                 random_state: int = 42):
        """
        Initialize ensemble predictor.
        
        Args:
            base_models: List of base predictors to ensemble
            ensemble_method: 'stacking', 'voting', or 'dynamic'
            random_state: Random state for reproducibility
        """
        self.ensemble_method = ensemble_method
        self.random_state = random_state
        self.is_trained = False
        self.logger = get_logger("EnsemblePredictor")
        
        # Initialize base models if not provided
        if base_models is None:
            self.base_models = [
                XGBoostPredictor(random_state=random_state),
                LightGBMPredictor(random_state=random_state),
                RandomForestPredictor(random_state=random_state)
            ]
        else:
            self.base_models = base_models
        
        # Initialize ensemble components
        self.meta_learner = MetaLearner(self.base_models, random_state)
        self.model_selector = ModelSelector(self.base_models)
        self.voting_classifier = None
        self.feature_names = None
        self.training_metrics = {}
        
    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        """Train ensemble with all base models and meta-learner."""
        try:
            self.logger.info(f"Training ensemble with {len(self.base_models)} base models")
            
            self.feature_names = list(features.columns)
            
            # Train all base models
            trained_models = []
            for i, model in enumerate(self.base_models):
                try:
                    self.logger.info(f"Training base model {i+1}/{len(self.base_models)}: {model.model_name}")
                    model.train(features, targets)
                    trained_models.append(model)
                    self.logger.info(f"✅ {model.model_name} training completed")
                except Exception as e:
                    self.logger.error(f"❌ Failed to train {model.model_name}: {e}")
                    continue
            
            if not trained_models:
                raise ModelTrainingError("No base models trained successfully")
            
            self.base_models = trained_models
            
            # Setup voting classifier for voting method
            if self.ensemble_method == 'voting':
                estimators = [(model.model_name, model.model) for model in self.base_models 
                            if hasattr(model, 'model') and model.model is not None]
                if estimators:
                    self.voting_classifier = VotingClassifier(
                        estimators=estimators,
                        voting='soft'  # Use probability voting
                    )
                    self.voting_classifier.fit(features.fillna(0), targets)
            
            # Train meta-learner for stacking method
            if self.ensemble_method == 'stacking':
                self.meta_learner.train(features, targets)
            
            # Update model selector performance
            for model in self.base_models:
                if hasattr(model, 'training_metrics'):
                    self.model_selector.update_performance_history(
                        model.model_name, 
                        model.training_metrics
                    )
            
            self.is_trained = True
            self._calculate_ensemble_metrics(features, targets)
            
            self.logger.info("✅ Ensemble training completed successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Ensemble training failed: {e}")
            raise ModelTrainingError(f"Ensemble training failed: {e}")
    
    def predict(self, features: MatchFeatures) -> Prediction:
        """Make ensemble prediction using configured method."""
        if not self.is_trained:
            raise ModelPredictionError("Ensemble must be trained before making predictions")
        
        try:
            # Convert to DataFrame if needed
            if isinstance(features, dict):
                features_df = pd.DataFrame([features])
            elif isinstance(features, pd.Series):
                features_df = features.to_frame().T
            else:
                features_df = features
            
            # Ensure correct feature order
            if self.feature_names:
                features_df = features_df[self.feature_names]
            
            # Make prediction based on ensemble method
            if self.ensemble_method == 'stacking':
                return self._predict_stacking(features_df)
            elif self.ensemble_method == 'voting':
                return self._predict_voting(features_df)
            elif self.ensemble_method == 'dynamic':
                return self._predict_dynamic(features_df)
            else:
                raise ModelPredictionError(f"Unknown ensemble method: {self.ensemble_method}")
                
        except Exception as e:
            self.logger.error(f"Ensemble prediction failed: {e}")
            raise ModelPredictionError(f"Ensemble prediction failed: {e}")
    
    def predict_batch(self, features: List[MatchFeatures]) -> List[Prediction]:
        """Make batch predictions using ensemble method."""
        return [self.predict(f) for f in features]
    
    def _predict_stacking(self, features: pd.DataFrame) -> Prediction:
        """Stacking prediction using meta-learner."""
        meta_proba, model_confidences = self.meta_learner.predict_meta(features)
        
        # Get the predicted class
        predicted_class_idx = np.argmax(meta_proba)
        predicted_class = self.meta_learner.label_encoder.inverse_transform([predicted_class_idx])[0]
        
        # Calculate ensemble confidence
        ensemble_confidence = float(np.max(meta_proba))
        
        return Prediction(
            home_win_prob=float(meta_proba[self._get_class_index('H', self.meta_learner.label_encoder)]),
            draw_prob=float(meta_proba[self._get_class_index('D', self.meta_learner.label_encoder)]),
            away_win_prob=float(meta_proba[self._get_class_index('A', self.meta_learner.label_encoder)]),
            predicted_outcome=predicted_class,
            confidence=ensemble_confidence,
            model_name="EnsembleStacking"
        )
    
    def _predict_voting(self, features: pd.DataFrame) -> Prediction:
        """Voting prediction using VotingClassifier."""
        if self.voting_classifier is None:
            raise ModelPredictionError("Voting classifier not initialized")
        
        # Get prediction probabilities
        proba = self.voting_classifier.predict_proba(features.fillna(0))[0]
        
        # Get predicted class
        predicted_class_idx = np.argmax(proba)
        predicted_class = self.voting_classifier.classes_[predicted_class_idx]
        
        return Prediction(
            home_win_prob=float(proba[self._get_class_index('H', None, self.voting_classifier.classes_)]),
            draw_prob=float(proba[self._get_class_index('D', None, self.voting_classifier.classes_)]),
            away_win_prob=float(proba[self._get_class_index('A', None, self.voting_classifier.classes_)]),
            predicted_outcome=predicted_class,
            confidence=float(np.max(proba)),
            model_name="EnsembleVoting"
        )
    
    def _predict_dynamic(self, features: pd.DataFrame) -> Prediction:
        """Dynamic prediction using model selector."""
        selected_model = self.model_selector.select_best_model(features)
        prediction = selected_model.predict(features)
        
        # Update model name to indicate ensemble selection
        prediction.model_name = f"EnsembleDynamic({selected_model.model_name})"
        
        return prediction
    
    def _get_class_index(self, class_label: str, encoder=None, classes=None) -> int:
        """Get index of class label."""
        try:
            if encoder is not None:
                return list(encoder.classes_).index(class_label)
            elif classes is not None:
                return list(classes).index(class_label)
            else:
                return 0
        except (ValueError, IndexError):
            return 0
    
    def _calculate_ensemble_metrics(self, features: pd.DataFrame, targets: pd.Series) -> None:
        """Calculate ensemble training metrics."""
        try:
            # Get individual model metrics
            model_metrics = {}
            for model in self.base_models:
                if hasattr(model, 'training_metrics'):
                    model_metrics[model.model_name] = model.training_metrics
            
            # Calculate ensemble-specific metrics
            self.training_metrics = {
                'ensemble_method': self.ensemble_method,
                'n_base_models': len(self.base_models),
                'trained_models': [m.model_name for m in self.base_models if m.is_trained],
                'base_model_metrics': model_metrics,
                'feature_count': len(self.feature_names) if self.feature_names else 0,
                'sample_count': len(features)
            }
            
            # Add method-specific metrics
            if self.ensemble_method == 'stacking' and self.meta_learner.is_trained:
                self.training_metrics['meta_learner_weights'] = self.meta_learner.model_weights
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate ensemble metrics: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return ensemble model information."""
        base_model_info = {}
        for model in self.base_models:
            base_model_info[model.model_name] = model.get_model_info()
        
        return {
            'model_name': 'EnsemblePredictor',
            'ensemble_method': self.ensemble_method,
            'is_trained': self.is_trained,
            'n_base_models': len(self.base_models),
            'base_models': base_model_info,
            'training_metrics': self.training_metrics
        }
    
    def save_model(self, path: str) -> None:
        """Save ensemble model."""
        if not self.is_trained:
            raise ModelTrainingError("Cannot save untrained ensemble")
        
        # Save individual models
        model_dir = Path(path).parent / f"{Path(path).stem}_models"
        model_dir.mkdir(exist_ok=True)
        
        for model in self.base_models:
            model_path = model_dir / f"{model.model_name}.pkl"
            model.save_model(str(model_path))
        
        # Save ensemble configuration
        ensemble_data = {
            'ensemble_method': self.ensemble_method,
            'random_state': self.random_state,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'meta_learner': self.meta_learner if self.ensemble_method == 'stacking' else None,
            'voting_classifier': self.voting_classifier if self.ensemble_method == 'voting' else None,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(ensemble_data, f)
        
        self.logger.info(f"Ensemble model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load ensemble model."""
        try:
            # Load ensemble configuration
            with open(path, 'rb') as f:
                ensemble_data = pickle.load(f)
            
            self.ensemble_method = ensemble_data['ensemble_method']
            self.random_state = ensemble_data['random_state']
            self.feature_names = ensemble_data['feature_names']
            self.training_metrics = ensemble_data['training_metrics']
            
            # Load individual models
            model_dir = Path(path).parent / f"{Path(path).stem}_models"
            
            loaded_models = []
            for model in self.base_models:
                model_path = model_dir / f"{model.model_name}.pkl"
                if model_path.exists():
                    model.load_model(str(model_path))
                    loaded_models.append(model)
            
            self.base_models = loaded_models
            
            # Restore ensemble components
            if self.ensemble_method == 'stacking' and ensemble_data.get('meta_learner'):
                self.meta_learner = ensemble_data['meta_learner']
            
            if self.ensemble_method == 'voting' and ensemble_data.get('voting_classifier'):
                self.voting_classifier = ensemble_data['voting_classifier']
            
            self.is_trained = True
            self.logger.info(f"Ensemble model loaded from {path}")
            
        except Exception as e:
            raise ModelTrainingError(f"Failed to load ensemble model: {e}")