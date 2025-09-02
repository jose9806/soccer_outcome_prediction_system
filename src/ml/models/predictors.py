"""
Production-ready ML predictors for soccer match outcome prediction.

Implements XGBoost, LightGBM, and RandomForest predictors with automatic
hyperparameter optimization and cross-validation capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import pickle
import json
from pathlib import Path

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import optuna
from optuna.samplers import TPESampler

from src.config.logging_config import get_logger
from ..core.interfaces import Predictor, ModelTrainer
from ..core.exceptions import ModelTrainingError, ModelPredictionError
from ..core.types import MatchFeatures, Prediction, ModelMetrics


class BasePredictor(Predictor):
    """
    Base class for all predictors with common functionality.
    
    Following Template Method pattern for consistent prediction workflow.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.label_encoder = LabelEncoder()
        self.training_metrics = {}
        self.logger = get_logger(f"{model_name}Predictor")
        
    def predict(self, features: MatchFeatures) -> Prediction:
        """Make prediction for a single match."""
        if not self.is_trained:
            raise ModelPredictionError("Model must be trained before making predictions")
        
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
                missing_features = set(self.feature_names) - set(features_df.columns)
                if missing_features:
                    raise ModelPredictionError(f"Missing features: {missing_features}")
                features_df = features_df[self.feature_names]
            
            # Get prediction probabilities
            probabilities = self._predict_proba(features_df)
            
            # Convert to class prediction
            predicted_class_idx = np.argmax(probabilities)
            predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            
            # Get confidence score
            confidence = float(np.max(probabilities))
            
            # Safely get probabilities for each class
            home_win_prob = 0.0
            draw_prob = 0.0
            away_win_prob = 0.0
            
            if self._has_class('H'):
                class_idx = self._get_class_index('H')
                if class_idx < len(probabilities):
                    home_win_prob = float(probabilities[class_idx])
                    
            if self._has_class('D'):
                class_idx = self._get_class_index('D')
                if class_idx < len(probabilities):
                    draw_prob = float(probabilities[class_idx])
                    
            if self._has_class('A'):
                class_idx = self._get_class_index('A')
                if class_idx < len(probabilities):
                    away_win_prob = float(probabilities[class_idx])
            
            # Ensure probabilities sum to 1 if we have partial classes
            total_prob = home_win_prob + draw_prob + away_win_prob
            if total_prob > 0 and total_prob != 1.0:
                home_win_prob /= total_prob
                draw_prob /= total_prob  
                away_win_prob /= total_prob
            elif total_prob == 0:
                # Fallback to equal probabilities if no classes match
                home_win_prob = draw_prob = away_win_prob = 1.0/3.0
            
            return Prediction(
                home_win_prob=home_win_prob,
                draw_prob=draw_prob,
                away_win_prob=away_win_prob,
                predicted_outcome=predicted_class,
                confidence=confidence,
                model_name=self.model_name
            )
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            raise ModelPredictionError(f"Prediction failed: {e}")
    
    def predict_batch(self, features: List[MatchFeatures]) -> List[Prediction]:
        """Make predictions for multiple matches efficiently."""
        if not self.is_trained:
            raise ModelPredictionError("Model must be trained before making predictions")
        
        try:
            # Convert to DataFrame
            if isinstance(features[0], dict):
                features_df = pd.DataFrame(features)
            else:
                features_df = pd.concat(features, ignore_index=True)
            
            # Ensure correct feature order
            if self.feature_names:
                features_df = features_df[self.feature_names]
            
            # Get predictions
            probabilities = self._predict_proba(features_df)
            predicted_classes = self.label_encoder.inverse_transform(np.argmax(probabilities, axis=1))
            confidences = np.max(probabilities, axis=1)
            
            # Create prediction objects
            predictions = []
            for i in range(len(features)):
                prob_row = probabilities[i]
                
                # Safely get probabilities for each class
                home_win_prob = 0.0
                draw_prob = 0.0
                away_win_prob = 0.0
                
                if self._has_class('H'):
                    class_idx = self._get_class_index('H')
                    if class_idx < len(prob_row):
                        home_win_prob = float(prob_row[class_idx])
                        
                if self._has_class('D'):
                    class_idx = self._get_class_index('D')
                    if class_idx < len(prob_row):
                        draw_prob = float(prob_row[class_idx])
                        
                if self._has_class('A'):
                    class_idx = self._get_class_index('A')
                    if class_idx < len(prob_row):
                        away_win_prob = float(prob_row[class_idx])
                
                # Ensure probabilities sum to 1
                total_prob = home_win_prob + draw_prob + away_win_prob
                if total_prob > 0 and total_prob != 1.0:
                    home_win_prob /= total_prob
                    draw_prob /= total_prob  
                    away_win_prob /= total_prob
                elif total_prob == 0:
                    home_win_prob = draw_prob = away_win_prob = 1.0/3.0
                
                predictions.append(Prediction(
                    home_win_prob=home_win_prob,
                    draw_prob=draw_prob,
                    away_win_prob=away_win_prob,
                    predicted_outcome=predicted_classes[i],
                    confidence=float(confidences[i]),
                    model_name=self.model_name
                ))
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Batch prediction error: {e}")
            raise ModelPredictionError(f"Batch prediction failed: {e}")
    
    def _predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Abstract method for probability prediction."""
        raise NotImplementedError("Subclasses must implement _predict_proba")
    
    def _get_class_index(self, class_label: str) -> int:
        """Get index of class label."""
        try:
            return list(self.label_encoder.classes_).index(class_label)
        except ValueError:
            return 0
    
    def _has_class(self, class_label: str) -> bool:
        """Check if class exists in label encoder."""
        return class_label in self.label_encoder.classes_
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata and configuration."""
        return {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'classes': list(self.label_encoder.classes_) if self.is_trained else [],
            'training_metrics': self.training_metrics,
            'model_type': self.__class__.__name__
        }
    
    def save_model(self, path: str) -> None:
        """Save trained model to disk."""
        if not self.is_trained:
            raise ModelTrainingError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'label_encoder': self.label_encoder,
            'training_metrics': self.training_metrics,
            'model_info': self.get_model_info()
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load trained model from disk."""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.label_encoder = model_data['label_encoder']
            self.training_metrics = model_data.get('training_metrics', {})
            self.is_trained = True
            
            self.logger.info(f"Model loaded from {path}")
            
        except Exception as e:
            raise ModelTrainingError(f"Failed to load model: {e}")


class XGBoostPredictor(BasePredictor, ModelTrainer):
    """
    XGBoost predictor with automatic hyperparameter optimization.
    
    Optimized for gradient boosting with excellent performance on structured data.
    """
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        super().__init__("XGBoost")
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.best_params = None
        
    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        """Train XGBoost model with hyperparameter optimization."""
        try:
            self.logger.info(f"Training XGBoost model with {len(features)} samples")
            
            # Prepare data
            self.feature_names = list(features.columns)
            y_encoded = self.label_encoder.fit_transform(targets)
            
            # Handle missing values
            X_train = features.fillna(0)
            
            # Hyperparameter optimization
            if self.best_params is None:
                self.logger.info("Starting hyperparameter optimization...")
                self.best_params = self._optimize_hyperparameters(X_train, y_encoded)
            
            # Train final model
            self.model = xgb.XGBClassifier(
                **self.best_params,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0
            )
            
            self.model.fit(X_train, y_encoded)
            self.is_trained = True
            
            # Calculate training metrics
            self._calculate_training_metrics(X_train, y_encoded, targets)
            
            self.logger.info("XGBoost training completed successfully")
            
        except Exception as e:
            self.logger.error(f"XGBoost training failed: {e}")
            raise ModelTrainingError(f"XGBoost training failed: {e}")
    
    def _predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        X = features.fillna(0)
        return self.model.predict_proba(X)
    
    def _optimize_hyperparameters(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Optimize XGBoost hyperparameters using Optuna."""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            }
            
            model = xgb.XGBClassifier(
                **params,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0
            )
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X, y, 
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                scoring='accuracy',
                n_jobs=-1
            )
            
            return cv_scores.mean()
        
        # Run optimization
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=50, show_progress_bar=False)
        
        self.logger.info(f"Best hyperparameters found with CV score: {study.best_value:.4f}")
        return study.best_params
    
    def _calculate_training_metrics(self, X: pd.DataFrame, y_encoded: np.ndarray, y_original: pd.Series) -> None:
        """Calculate and store training metrics."""
        # Predictions
        y_pred_proba = self.model.predict_proba(X)
        y_pred = self.model.predict(X)
        y_pred_original = self.label_encoder.inverse_transform(y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(
            self.model, X, y_encoded,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring='accuracy'
        )
        
        # Feature importance
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        self.training_metrics = {
            'accuracy': float(accuracy_score(y_original, y_pred_original)),
            'cv_accuracy_mean': float(cv_scores.mean()),
            'cv_accuracy_std': float(cv_scores.std()),
            'n_classes': len(self.label_encoder.classes_),
            'n_features': len(self.feature_names),
            'n_samples': len(X),
            'best_params': self.best_params,
            'feature_importance': feature_importance
        }


class LightGBMPredictor(BasePredictor, ModelTrainer):
    """
    LightGBM predictor optimized for fast training and inference.
    
    Excellent for large datasets with efficient memory usage.
    """
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        super().__init__("LightGBM")
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.best_params = None
        
    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        """Train LightGBM model with hyperparameter optimization."""
        try:
            self.logger.info(f"Training LightGBM model with {len(features)} samples")
            
            # Prepare data
            self.feature_names = list(features.columns)
            y_encoded = self.label_encoder.fit_transform(targets)
            
            # Handle missing values
            X_train = features.fillna(0)
            
            # Hyperparameter optimization
            if self.best_params is None:
                self.logger.info("Starting hyperparameter optimization...")
                self.best_params = self._optimize_hyperparameters(X_train, y_encoded)
            
            # Train final model
            self.model = lgb.LGBMClassifier(
                **self.best_params,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=-1
            )
            
            self.model.fit(X_train, y_encoded)
            self.is_trained = True
            
            # Calculate training metrics
            self._calculate_training_metrics(X_train, y_encoded, targets)
            
            self.logger.info("LightGBM training completed successfully")
            
        except Exception as e:
            self.logger.error(f"LightGBM training failed: {e}")
            raise ModelTrainingError(f"LightGBM training failed: {e}")
    
    def _predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        X = features.fillna(0)
        return self.model.predict_proba(X)
    
    def _optimize_hyperparameters(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Optimize LightGBM hyperparameters using Optuna."""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            }
            
            model = lgb.LGBMClassifier(
                **params,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=-1
            )
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X, y,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                scoring='accuracy',
                n_jobs=-1
            )
            
            return cv_scores.mean()
        
        # Run optimization
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=50, show_progress_bar=False)
        
        self.logger.info(f"Best hyperparameters found with CV score: {study.best_value:.4f}")
        return study.best_params
    
    def _calculate_training_metrics(self, X: pd.DataFrame, y_encoded: np.ndarray, y_original: pd.Series) -> None:
        """Calculate and store training metrics."""
        # Predictions
        y_pred_proba = self.model.predict_proba(X)
        y_pred = self.model.predict(X)
        y_pred_original = self.label_encoder.inverse_transform(y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(
            self.model, X, y_encoded,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring='accuracy'
        )
        
        # Feature importance
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        self.training_metrics = {
            'accuracy': float(accuracy_score(y_original, y_pred_original)),
            'cv_accuracy_mean': float(cv_scores.mean()),
            'cv_accuracy_std': float(cv_scores.std()),
            'n_classes': len(self.label_encoder.classes_),
            'n_features': len(self.feature_names),
            'n_samples': len(X),
            'best_params': self.best_params,
            'feature_importance': feature_importance
        }


class RandomForestPredictor(BasePredictor, ModelTrainer):
    """
    Random Forest predictor for robust ensemble learning.
    
    Excellent baseline with good interpretability and robustness to overfitting.
    """
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        super().__init__("RandomForest")
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.best_params = None
        
    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        """Train Random Forest model with hyperparameter optimization."""
        try:
            self.logger.info(f"Training Random Forest model with {len(features)} samples")
            
            # Prepare data
            self.feature_names = list(features.columns)
            y_encoded = self.label_encoder.fit_transform(targets)
            
            # Handle missing values
            X_train = features.fillna(0)
            
            # Hyperparameter optimization
            if self.best_params is None:
                self.logger.info("Starting hyperparameter optimization...")
                self.best_params = self._optimize_hyperparameters(X_train, y_encoded)
            
            # Train final model
            self.model = RandomForestClassifier(
                **self.best_params,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            self.model.fit(X_train, y_encoded)
            self.is_trained = True
            
            # Calculate training metrics
            self._calculate_training_metrics(X_train, y_encoded, targets)
            
            self.logger.info("Random Forest training completed successfully")
            
        except Exception as e:
            self.logger.error(f"Random Forest training failed: {e}")
            raise ModelTrainingError(f"Random Forest training failed: {e}")
    
    def _predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        X = features.fillna(0)
        return self.model.predict_proba(X)
    
    def _optimize_hyperparameters(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Optimize Random Forest hyperparameters using Optuna."""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            }
            
            model = RandomForestClassifier(
                **params,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X, y,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                scoring='accuracy',
                n_jobs=-1
            )
            
            return cv_scores.mean()
        
        # Run optimization
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=50, show_progress_bar=False)
        
        self.logger.info(f"Best hyperparameters found with CV score: {study.best_value:.4f}")
        return study.best_params
    
    def _calculate_training_metrics(self, X: pd.DataFrame, y_encoded: np.ndarray, y_original: pd.Series) -> None:
        """Calculate and store training metrics."""
        # Predictions
        y_pred_proba = self.model.predict_proba(X)
        y_pred = self.model.predict(X)
        y_pred_original = self.label_encoder.inverse_transform(y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(
            self.model, X, y_encoded,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring='accuracy'
        )
        
        # Feature importance
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        self.training_metrics = {
            'accuracy': float(accuracy_score(y_original, y_pred_original)),
            'cv_accuracy_mean': float(cv_scores.mean()),
            'cv_accuracy_std': float(cv_scores.std()),
            'n_classes': len(self.label_encoder.classes_),
            'n_features': len(self.feature_names),
            'n_samples': len(X),
            'best_params': self.best_params,
            'feature_importance': feature_importance
        }