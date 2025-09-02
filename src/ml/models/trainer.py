"""
Model training utilities and cross-validation system.

Provides comprehensive model training, evaluation, and hyperparameter 
optimization capabilities for soccer outcome prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import pickle
import json
from pathlib import Path
import time

from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, TimeSeriesSplit, 
    GridSearchCV, RandomizedSearchCV, train_test_split
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, log_loss, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
import optuna
from optuna.samplers import TPESampler
import mlflow
import mlflow.sklearn

from src.config.logging_config import get_logger
from ..core.interfaces import ModelTrainer as IModelTrainer, Predictor
from ..core.exceptions import ModelTrainingError, ValidationError
from ..core.types import ModelMetrics

from .predictors import XGBoostPredictor, LightGBMPredictor, RandomForestPredictor
from .ensemble import EnsemblePredictor


class CrossValidator:
    """
    Advanced cross-validation system for time-series aware model evaluation.
    
    Implements multiple CV strategies suitable for temporal soccer data.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.logger = get_logger("CrossValidator")
        
    def evaluate_model(self, 
                      model: Predictor,
                      features: pd.DataFrame,
                      targets: pd.Series,
                      cv_strategy: str = 'temporal',
                      cv_folds: int = 5) -> ModelMetrics:
        """
        Evaluate model using specified cross-validation strategy.
        
        Args:
            model: Predictor to evaluate
            features: Feature matrix
            targets: Target labels
            cv_strategy: 'temporal', 'stratified', or 'blocked'
            cv_folds: Number of CV folds
            
        Returns:
            ModelMetrics with comprehensive evaluation results
        """
        try:
            self.logger.info(f"Evaluating {model.__class__.__name__} using {cv_strategy} CV")
            
            # Prepare data
            X = features.fillna(0)
            y = targets
            
            # Choose CV strategy
            cv = self._get_cv_strategy(cv_strategy, cv_folds, X, y)
            
            # Prepare model for sklearn compatibility
            sklearn_model = self._prepare_sklearn_model(model, X, y)
            
            # Calculate cross-validation scores
            cv_results = self._calculate_cv_scores(sklearn_model, X, y, cv)
            
            # Detailed evaluation on full dataset  
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            sklearn_model.fit(X, y_encoded)
            detailed_metrics = self._calculate_detailed_metrics(sklearn_model, X, y, label_encoder)
            
            # Combine results
            metrics = ModelMetrics(
                accuracy=cv_results['accuracy_mean'],
                precision=cv_results['precision_mean'],
                recall=cv_results['recall_mean'],
                f1_score=cv_results['f1_mean'],
                cv_scores=cv_results,
                detailed_metrics=detailed_metrics,
                model_name=model.__class__.__name__,
                cv_strategy=cv_strategy,
                evaluation_timestamp=datetime.now().isoformat()
            )
            
            self.logger.info(f"Evaluation completed: {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")
            raise ValidationError(f"Model evaluation failed: {e}")
    
    def _get_cv_strategy(self, strategy: str, n_folds: int, X: pd.DataFrame, y: pd.Series):
        """Get cross-validation strategy object."""
        if strategy == 'temporal':
            # Time-aware splitting for temporal data
            return TimeSeriesSplit(n_splits=n_folds)
        elif strategy == 'stratified':
            # Stratified for balanced class distribution
            return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        elif strategy == 'blocked':
            # Custom blocked CV for grouped data
            return self._create_blocked_cv(X, y, n_folds)
        else:
            raise ValidationError(f"Unknown CV strategy: {strategy}")
    
    def _create_blocked_cv(self, X: pd.DataFrame, y: pd.Series, n_folds: int):
        """Create blocked cross-validation for temporal data."""
        n_samples = len(X)
        fold_size = n_samples // n_folds
        
        for i in range(n_folds):
            # Use past data for training, future for validation
            val_start = i * fold_size
            val_end = min((i + 1) * fold_size, n_samples)
            
            if val_start == 0:
                # First fold: use second block for training
                train_idx = list(range(val_end, min(val_end + fold_size * 2, n_samples)))
                val_idx = list(range(val_start, val_end))
            else:
                # Use all previous data for training
                train_idx = list(range(0, val_start))
                val_idx = list(range(val_start, val_end))
            
            if len(train_idx) > 0 and len(val_idx) > 0:
                yield train_idx, val_idx
    
    def _prepare_sklearn_model(self, model: Predictor, X: pd.DataFrame, y: pd.Series):
        """Prepare model for sklearn compatibility."""
        # Clone model if it has sklearn-compatible interface
        if hasattr(model, 'model') and model.model is not None:
            return model.model
        
        # Train model if not trained
        if not model.is_trained:
            model.train(X, y)
        
        return model.model if hasattr(model, 'model') else model
    
    def _calculate_cv_scores(self, model, X: pd.DataFrame, y: pd.Series, cv) -> Dict[str, float]:
        """Calculate comprehensive cross-validation scores."""
        from sklearn.preprocessing import LabelEncoder
        
        # Encode labels for sklearn compatibility
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        cv_results = {}
        
        for metric in scoring_metrics:
            try:
                scores = cross_val_score(model, X, y_encoded, cv=cv, scoring=metric, n_jobs=-1)
                cv_results[f"{metric}_mean"] = float(scores.mean())
                cv_results[f"{metric}_std"] = float(scores.std())
                cv_results[f"{metric}_scores"] = scores.tolist()
            except Exception as e:
                self.logger.warning(f"Failed to calculate {metric}: {e}")
                cv_results[f"{metric}_mean"] = 0.0
                cv_results[f"{metric}_std"] = 0.0
                cv_results[f"{metric}_scores"] = []
        
        return cv_results
    
    def _calculate_detailed_metrics(self, model, X: pd.DataFrame, y: pd.Series, label_encoder=None) -> Dict[str, Any]:
        """Calculate detailed metrics on full dataset."""
        try:
            from sklearn.preprocessing import LabelEncoder
            
            # Use provided encoder or create new one
            if label_encoder is None:
                label_encoder = LabelEncoder()
                y_encoded = label_encoder.fit_transform(y)
            else:
                y_encoded = label_encoder.transform(y)
            
            # Predictions (encoded)
            y_pred_encoded = model.predict(X)
            y_pred_proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
            
            # Basic metrics
            accuracy = accuracy_score(y_encoded, y_pred_encoded)
            precision, recall, f1, support = precision_recall_fscore_support(y_encoded, y_pred_encoded, average='macro')
            
            # Get per-class support for compatibility
            _, _, _, per_class_support = precision_recall_fscore_support(y_encoded, y_pred_encoded, average=None)
            
            # For display, convert back to original labels
            y_pred_original = label_encoder.inverse_transform(y_pred_encoded)
            
            # Classification report (with original labels for readability)
            class_report = classification_report(y, y_pred_original, output_dict=True)
            
            # Confusion matrix (with encoded labels for consistency)
            conf_matrix = confusion_matrix(y_encoded, y_pred_encoded)
            
            # Probability-based metrics
            detailed = {
                'accuracy': float(accuracy),
                'precision_macro': float(precision),
                'recall_macro': float(recall),
                'f1_macro': float(f1),
                'classification_report': class_report,
                'confusion_matrix': conf_matrix.tolist(),
                'support_by_class': per_class_support.tolist() if per_class_support is not None else []
            }
            
            # Add probability-based metrics if available
            if y_pred_proba is not None:
                try:
                    # Multi-class log loss (use encoded labels)
                    detailed['log_loss'] = float(log_loss(y_encoded, y_pred_proba))
                    
                    # AUC for binary classification or multiclass
                    if len(np.unique(y_encoded)) == 2:
                        detailed['auc_roc'] = float(roc_auc_score(y_encoded, y_pred_proba[:, 1]))
                    else:
                        detailed['auc_roc_multiclass'] = float(roc_auc_score(y_encoded, y_pred_proba, multi_class='ovo', average='macro'))
                except Exception as e:
                    self.logger.warning(f"Failed to calculate probability metrics: {e}")
            
            return detailed
            
        except Exception as e:
            self.logger.error(f"Failed to calculate detailed metrics: {e}")
            return {'error': str(e)}


class HyperparameterOptimizer:
    """
    Advanced hyperparameter optimization using Optuna.
    
    Provides efficient hyperparameter search with early stopping
    and advanced optimization strategies.
    """
    
    def __init__(self, n_trials: int = 100, random_state: int = 42):
        self.n_trials = n_trials
        self.random_state = random_state
        self.logger = get_logger("HyperparameterOptimizer")
        
    def optimize_model(self, 
                      model_type: str,
                      features: pd.DataFrame,
                      targets: pd.Series,
                      cv_folds: int = 5) -> Dict[str, Any]:
        """
        Optimize hyperparameters for specified model type.
        
        Args:
            model_type: 'xgboost', 'lightgbm', 'randomforest', or 'ensemble'
            features: Feature matrix
            targets: Target labels
            cv_folds: Number of CV folds for evaluation
            
        Returns:
            Dict with best parameters and optimization results
        """
        try:
            self.logger.info(f"Starting hyperparameter optimization for {model_type}")
            
            # Prepare data
            X = features.fillna(0)
            y = targets
            
            # Create optimization objective
            objective = self._create_objective(model_type, X, y, cv_folds)
            
            # Setup Optuna study
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=self.random_state)
            )
            
            # Run optimization
            study.optimize(
                objective, 
                n_trials=self.n_trials,
                show_progress_bar=True
            )
            
            # Extract results
            results = {
                'best_params': study.best_params,
                'best_value': study.best_value,
                'n_trials': len(study.trials),
                'optimization_history': [trial.value for trial in study.trials if trial.value is not None],
                'best_trial_number': study.best_trial.number,
                'model_type': model_type
            }
            
            self.logger.info(f"Optimization completed: {study.best_value:.4f} with {len(study.trials)} trials")
            return results
            
        except Exception as e:
            self.logger.error(f"Hyperparameter optimization failed: {e}")
            raise ModelTrainingError(f"Optimization failed: {e}")
    
    def _create_objective(self, model_type: str, X: pd.DataFrame, y: pd.Series, cv_folds: int):
        """Create optimization objective function."""
        
        def objective(trial):
            try:
                # Get model-specific hyperparameter suggestions
                if model_type == 'xgboost':
                    params = self._suggest_xgboost_params(trial)
                    model = XGBoostPredictor(random_state=self.random_state)
                elif model_type == 'lightgbm':
                    params = self._suggest_lightgbm_params(trial)
                    model = LightGBMPredictor(random_state=self.random_state)
                elif model_type == 'randomforest':
                    params = self._suggest_randomforest_params(trial)
                    model = RandomForestPredictor(random_state=self.random_state)
                elif model_type == 'ensemble':
                    params = self._suggest_ensemble_params(trial)
                    model = EnsemblePredictor(random_state=self.random_state)
                else:
                    raise ValidationError(f"Unknown model type: {model_type}")
                
                # Set hyperparameters
                for param, value in params.items():
                    setattr(model, param, value)
                
                # Override best_params for immediate use
                model.best_params = params
                
                # Evaluate using cross-validation
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
                
                # Train model for CV evaluation
                model.train(X, y)
                
                if hasattr(model, 'model') and model.model is not None:
                    scores = cross_val_score(model.model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
                    return scores.mean()
                else:
                    # Fallback evaluation
                    return 0.33  # Random chance for 3-class problem
                
            except Exception as e:
                self.logger.warning(f"Trial failed: {e}")
                return 0.0  # Return poor score for failed trials
        
        return objective
    
    def _suggest_xgboost_params(self, trial) -> Dict[str, Any]:
        """Suggest XGBoost hyperparameters."""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        }
    
    def _suggest_lightgbm_params(self, trial) -> Dict[str, Any]:
        """Suggest LightGBM hyperparameters."""
        return {
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
    
    def _suggest_randomforest_params(self, trial) -> Dict[str, Any]:
        """Suggest Random Forest hyperparameters."""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        }
    
    def _suggest_ensemble_params(self, trial) -> Dict[str, Any]:
        """Suggest Ensemble hyperparameters."""
        return {
            'ensemble_method': trial.suggest_categorical('ensemble_method', ['stacking', 'voting', 'dynamic'])
        }


class ModelTrainer(IModelTrainer):
    """
    Comprehensive model training system.
    
    Orchestrates the complete training workflow including data preparation,
    hyperparameter optimization, model training, and evaluation.
    """
    
    def __init__(self, 
                 use_mlflow: bool = True,
                 experiment_name: str = "soccer_prediction",
                 random_state: int = 42):
        self.use_mlflow = use_mlflow
        self.experiment_name = experiment_name
        self.random_state = random_state
        self.logger = get_logger("ModelTrainer")
        
        # Initialize components
        self.cross_validator = CrossValidator(random_state)
        self.optimizer = HyperparameterOptimizer(random_state=random_state)
        
        # Setup MLflow if enabled
        if self.use_mlflow:
            self._setup_mlflow()
        
        # Track training session
        self.training_session = {
            'start_time': None,
            'models_trained': [],
            'best_model': None,
            'session_id': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
    
    def train_all_models(self, 
                        features: pd.DataFrame,
                        targets: pd.Series,
                        optimize_hyperparams: bool = True,
                        cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train all available models with comprehensive evaluation.
        
        Args:
            features: Feature matrix
            targets: Target labels
            optimize_hyperparams: Whether to optimize hyperparameters
            cv_folds: Number of CV folds
            
        Returns:
            Dict with training results and model comparison
        """
        try:
            self.training_session['start_time'] = datetime.now()
            self.logger.info("Starting comprehensive model training session")
            
            # Validate data
            self._validate_training_data(features, targets)
            
            # Define models to train
            model_configs = {
                'xgboost': XGBoostPredictor,
                'lightgbm': LightGBMPredictor,
                'randomforest': RandomForestPredictor
            }
            
            # Train individual models
            trained_models = {}
            training_results = {}
            
            for model_name, model_class in model_configs.items():
                try:
                    self.logger.info(f"Training {model_name}...")
                    
                    # Initialize model
                    model = model_class(random_state=self.random_state)
                    
                    # Optimize hyperparameters if requested
                    if optimize_hyperparams:
                        self.logger.info(f"Optimizing hyperparameters for {model_name}...")
                        opt_results = self.optimizer.optimize_model(
                            model_name, features, targets, cv_folds
                        )
                        
                        # Apply optimized parameters
                        for param, value in opt_results['best_params'].items():
                            setattr(model, param, value)
                        model.best_params = opt_results['best_params']
                    
                    # Train model
                    model.train(features, targets)
                    
                    # Evaluate model
                    metrics = self.cross_validator.evaluate_model(
                        model, features, targets, cv_strategy='temporal', cv_folds=cv_folds
                    )
                    
                    # Store results
                    trained_models[model_name] = model
                    training_results[model_name] = {
                        'metrics': metrics,
                        'optimization_results': opt_results if optimize_hyperparams else None,
                        'training_time': time.time() - self.training_session['start_time'].timestamp()
                    }
                    
                    self.training_session['models_trained'].append(model_name)
                    
                    # Log to MLflow
                    if self.use_mlflow:
                        self._log_model_to_mlflow(model_name, model, metrics, 
                                               opt_results if optimize_hyperparams else None)
                    
                    self.logger.info(f"✅ {model_name} trained successfully: {metrics.accuracy:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to train {model_name}: {e}")
                    continue
            
            # Train ensemble model
            if len(trained_models) >= 2:
                try:
                    self.logger.info("Training ensemble model...")
                    
                    ensemble = EnsemblePredictor(
                        base_models=list(trained_models.values()),
                        ensemble_method='stacking',
                        random_state=self.random_state
                    )
                    
                    ensemble.train(features, targets)
                    ensemble_metrics = self.cross_validator.evaluate_model(
                        ensemble, features, targets, cv_strategy='temporal', cv_folds=cv_folds
                    )
                    
                    trained_models['ensemble'] = ensemble
                    training_results['ensemble'] = {
                        'metrics': ensemble_metrics,
                        'optimization_results': None,
                        'training_time': time.time() - self.training_session['start_time'].timestamp()
                    }
                    
                    self.training_session['models_trained'].append('ensemble')
                    
                    if self.use_mlflow:
                        self._log_model_to_mlflow('ensemble', ensemble, ensemble_metrics)
                    
                    self.logger.info(f"✅ Ensemble trained successfully: {ensemble_metrics.accuracy:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to train ensemble: {e}")
            
            # Determine best model
            best_model_name, best_model = self._select_best_model(trained_models, training_results)
            self.training_session['best_model'] = best_model_name
            
            # Create comprehensive results
            session_results = {
                'session_info': self.training_session,
                'trained_models': trained_models,
                'training_results': training_results,
                'best_model': {
                    'name': best_model_name,
                    'model': best_model,
                    'metrics': training_results[best_model_name]['metrics']
                },
                'model_comparison': self._create_model_comparison(training_results),
                'recommendations': self._generate_recommendations(training_results)
            }
            
            # Save training session
            self._save_training_session(session_results)
            
            # Log session summary
            self._log_training_summary(session_results)
            
            return session_results
            
        except Exception as e:
            self.logger.error(f"Training session failed: {e}")
            raise ModelTrainingError(f"Training session failed: {e}")
    
    def _validate_training_data(self, features: pd.DataFrame, targets: pd.Series) -> None:
        """Validate training data quality."""
        if features.empty:
            raise ValidationError("Features dataset is empty")
        
        if targets.empty:
            raise ValidationError("Targets dataset is empty")
        
        if len(features) != len(targets):
            raise ValidationError("Features and targets have different lengths")
        
        if len(targets.unique()) < 2:
            raise ValidationError("Targets must have at least 2 classes")
        
        # Check for excessive missing values
        missing_ratio = features.isnull().sum().sum() / (features.shape[0] * features.shape[1])
        if missing_ratio > 0.5:
            self.logger.warning(f"High missing value ratio: {missing_ratio:.2%}")
    
    def _setup_mlflow(self) -> None:
        """Setup MLflow experiment tracking."""
        try:
            mlflow.set_experiment(self.experiment_name)
            self.logger.info(f"MLflow experiment set: {self.experiment_name}")
        except Exception as e:
            self.logger.warning(f"Failed to setup MLflow: {e}")
            self.use_mlflow = False
    
    def _log_model_to_mlflow(self, model_name: str, model, metrics: ModelMetrics, 
                           optimization_results: Dict = None) -> None:
        """Log model and metrics to MLflow."""
        if not self.use_mlflow:
            return
        
        try:
            with mlflow.start_run(run_name=f"{model_name}_{self.training_session['session_id']}"):
                # Log parameters
                if hasattr(model, 'best_params') and model.best_params:
                    mlflow.log_params(model.best_params)
                
                # Log metrics
                mlflow.log_metric("accuracy", metrics.accuracy)
                mlflow.log_metric("precision", metrics.precision)
                mlflow.log_metric("recall", metrics.recall)
                mlflow.log_metric("f1_score", metrics.f1_score)
                
                if metrics.cv_scores:
                    for metric, value in metrics.cv_scores.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"cv_{metric}", value)
                
                # Log model
                if hasattr(model, 'model') and model.model is not None:
                    mlflow.sklearn.log_model(model.model, f"model_{model_name}")
                
                # Log optimization results
                if optimization_results:
                    mlflow.log_param("n_trials", optimization_results['n_trials'])
                    mlflow.log_metric("best_cv_score", optimization_results['best_value'])
                
        except Exception as e:
            self.logger.warning(f"Failed to log {model_name} to MLflow: {e}")
    
    def _select_best_model(self, models: Dict, results: Dict) -> Tuple[str, Any]:
        """Select best model based on cross-validation performance."""
        best_model_name = None
        best_score = -1
        
        for model_name, result in results.items():
            score = result['metrics'].accuracy
            if score > best_score:
                best_score = score
                best_model_name = model_name
        
        return best_model_name, models[best_model_name] if best_model_name else None
    
    def _create_model_comparison(self, results: Dict) -> pd.DataFrame:
        """Create model comparison DataFrame."""
        comparison_data = []
        
        for model_name, result in results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics.accuracy,
                'Precision': metrics.precision,
                'Recall': metrics.recall,
                'F1_Score': metrics.f1_score,
                'CV_Std': metrics.cv_scores.get('accuracy_std', 0.0) if metrics.cv_scores else 0.0,
                'Training_Time': result['training_time']
            })
        
        return pd.DataFrame(comparison_data).sort_values('Accuracy', ascending=False)
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate training recommendations based on results."""
        recommendations = []
        
        # Performance recommendations
        accuracies = [result['metrics'].accuracy for result in results.values()]
        best_accuracy = max(accuracies)
        
        if best_accuracy < 0.5:
            recommendations.append("⚠️ Low accuracy detected. Consider feature engineering improvements.")
        elif best_accuracy > 0.7:
            recommendations.append("🎉 Excellent accuracy achieved. Model is ready for production.")
        
        # Model-specific recommendations
        if 'ensemble' in results:
            ensemble_acc = results['ensemble']['metrics'].accuracy
            best_individual = max([r['metrics'].accuracy for k, r in results.items() if k != 'ensemble'])
            
            if ensemble_acc > best_individual:
                recommendations.append("✅ Ensemble outperforms individual models. Use ensemble for production.")
            else:
                recommendations.append("⚠️ Ensemble doesn't improve performance. Consider individual models.")
        
        # Variance recommendations
        cv_stds = [result['metrics'].cv_scores.get('accuracy_std', 0.0) 
                  for result in results.values() if result['metrics'].cv_scores]
        
        if cv_stds and max(cv_stds) > 0.05:
            recommendations.append("⚠️ High variance detected. Consider more regularization or more data.")
        
        return recommendations
    
    def _save_training_session(self, results: Dict) -> None:
        """Save training session results."""
        try:
            # Create models directory
            models_dir = Path("data/models")
            models_dir.mkdir(exist_ok=True)
            
            session_id = self.training_session['session_id']
            
            # Save individual models
            for model_name, model in results['trained_models'].items():
                model_path = models_dir / f"{model_name}_{session_id}.pkl"
                model.save_model(str(model_path))
            
            # Save session metadata
            session_metadata = {
                'session_info': results['session_info'],
                'model_comparison': results['model_comparison'].to_dict('records'),
                'recommendations': results['recommendations'],
                'best_model': {
                    'name': results['best_model']['name'],
                    'metrics': results['best_model']['metrics'].__dict__
                }
            }
            
            metadata_path = models_dir / f"training_session_{session_id}.json"
            with open(metadata_path, 'w') as f:
                json.dump(session_metadata, f, indent=2, default=str)
            
            self.logger.info(f"Training session saved: {models_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save training session: {e}")
    
    def _log_training_summary(self, results: Dict) -> None:
        """Log comprehensive training summary."""
        self.logger.info("\n" + "="*60)
        self.logger.info("🎯 MODEL TRAINING SESSION COMPLETED")
        self.logger.info("="*60)
        
        session = results['session_info']
        self.logger.info(f"📊 SESSION SUMMARY:")
        self.logger.info(f"   Session ID: {session['session_id']}")
        self.logger.info(f"   Models trained: {len(session['models_trained'])}")
        self.logger.info(f"   Best model: {session['best_model']}")
        
        # Model comparison
        comparison = results['model_comparison']
        self.logger.info(f"\n🏆 MODEL PERFORMANCE RANKING:")
        for _, row in comparison.iterrows():
            self.logger.info(f"   {row['Model']}: {row['Accuracy']:.4f} (±{row['CV_Std']:.4f})")
        
        # Recommendations
        self.logger.info(f"\n💡 RECOMMENDATIONS:")
        for rec in results['recommendations']:
            self.logger.info(f"   {rec}")
        
        self.logger.info("\n" + "="*60)
    
    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        """Train model (required by interface)."""
        # This is mainly for individual model training
        results = self.train_all_models(features, targets)
        self.best_model = results['best_model']['model']
    
    def save_model(self, path: str) -> None:
        """Save best model."""
        if hasattr(self, 'best_model') and self.best_model:
            self.best_model.save_model(path)
        else:
            raise ModelTrainingError("No trained model to save")
    
    def load_model(self, path: str) -> None:
        """Load trained model."""
        # This would be implemented based on specific model type
        raise NotImplementedError("Use specific model classes to load models")