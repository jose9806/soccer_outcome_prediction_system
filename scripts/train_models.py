"""
Main training script for all ML models.

Single entry point for training XGBoost, LightGBM, RandomForest and Ensemble models.
Follows SOLID principles with clean separation of concerns.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Optional
import argparse

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.config.logging_config import get_logger
from src.ml.models.predictors import XGBoostPredictor, LightGBMPredictor, RandomForestPredictor
from src.ml.models.ensemble import EnsemblePredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def create_synthetic_training_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create comprehensive synthetic training data."""
    logger = get_logger("DataGenerator")
    logger.info(f"Creating synthetic training data with {n_samples} samples")
    
    np.random.seed(42)
    
    # Comprehensive feature set
    data = {
        # Basic match statistics
        'home_goals': np.random.poisson(1.3, n_samples),
        'away_goals': np.random.poisson(1.1, n_samples),
        'home_shots': np.random.poisson(12, n_samples),
        'away_shots': np.random.poisson(10, n_samples),
        'home_shots_on_target': np.random.poisson(5, n_samples),
        'away_shots_on_target': np.random.poisson(4, n_samples),
        'home_possession': np.random.normal(50, 15, n_samples),
        'away_possession': np.random.normal(50, 15, n_samples),
        
        # Performance metrics
        'home_xg': np.random.normal(1.4, 0.4, n_samples),
        'away_xg': np.random.normal(1.2, 0.4, n_samples),
        'goal_difference': np.random.normal(0, 1.5, n_samples),
        'shots_ratio': np.random.lognormal(0, 0.3, n_samples),
        'home_advantage': np.random.normal(0.1, 0.05, n_samples),
        
        # Form indicators
        'home_form_5': np.random.normal(1.5, 0.5, n_samples),
        'away_form_5': np.random.normal(1.5, 0.5, n_samples),
        'home_momentum': np.random.normal(0, 1, n_samples),
        'away_momentum': np.random.normal(0, 1, n_samples),
    }
    
    # Add more features for realistic dataset
    for i in range(25):  # Additional synthetic features
        data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
    
    df = pd.DataFrame(data)
    
    # Ensure possession adds to 100
    total_poss = df['home_possession'] + df['away_possession']
    df['home_possession'] = (df['home_possession'] / total_poss) * 100
    df['away_possession'] = 100 - df['home_possession']
    
    # Create realistic results based on features
    results = []
    for i in range(n_samples):
        home_strength = (
            df.loc[i, 'home_advantage'] +
            0.1 * df.loc[i, 'goal_difference'] +
            0.05 * df.loc[i, 'home_momentum'] +
            0.05 * df.loc[i, 'home_form_5']
        )
        
        # Convert to probabilities
        home_prob = 0.45 + home_strength
        draw_prob = 0.25 - 0.05 * abs(home_strength)
        away_prob = 1 - home_prob - draw_prob
        
        # Normalize probabilities
        probs = np.array([max(0.05, home_prob), max(0.05, draw_prob), max(0.05, away_prob)])
        probs = probs / probs.sum()
        
        result = np.random.choice(['H', 'D', 'A'], p=probs)
        results.append(result)
    
    df['result'] = results
    
    logger.info(f"Generated data with {len(df.columns)} features")
    logger.info(f"Result distribution: {pd.Series(results).value_counts().to_dict()}")
    
    return df

def train_individual_models(X_train: pd.DataFrame, y_train: pd.Series, 
                          X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    """Train individual ML models."""
    logger = get_logger("IndividualTraining")
    logger.info("Training individual models")
    
    models = [
        ("XGBoost", XGBoostPredictor(random_state=42)),
        ("LightGBM", LightGBMPredictor(random_state=42)),
        ("RandomForest", RandomForestPredictor(random_state=42))
    ]
    
    results = {}
    trained_models = []
    
    for model_name, model in models:
        logger.info(f"Training {model_name}...")
        
        try:
            # Train model
            model.train(X_train, y_train)
            
            # Test predictions
            predictions = []
            for i in range(len(X_test)):
                pred = model.predict(X_test.iloc[i])
                predictions.append(pred.predicted_outcome)
            
            accuracy = accuracy_score(y_test, predictions)
            
            # Save model
            model_path = f"models/{model_name.lower()}_model.pkl"
            model.save_model(model_path)
            
            results[model_name] = {
                'accuracy': accuracy,
                'model_path': model_path,
                'training_metrics': model.training_metrics
            }
            
            trained_models.append(model)
            logger.info(f"✅ {model_name}: {accuracy:.4f} accuracy")
            
        except Exception as e:
            logger.error(f"❌ {model_name} training failed: {e}")
            results[model_name] = {'error': str(e)}
    
    return results, trained_models

def train_ensemble_models(X_train: pd.DataFrame, y_train: pd.Series,
                         X_test: pd.DataFrame, y_test: pd.Series,
                         base_models: List) -> Dict:
    """Train ensemble models."""
    logger = get_logger("EnsembleTraining")
    logger.info("Training ensemble models")
    
    ensemble_methods = ['voting', 'stacking', 'dynamic']
    results = {}
    
    for method in ensemble_methods:
        logger.info(f"Training {method} ensemble...")
        
        try:
            # Create ensemble with trained base models
            ensemble = EnsemblePredictor(
                base_models=[model for model in base_models if model.is_trained],
                ensemble_method=method,
                random_state=42
            )
            
            # Train ensemble
            ensemble.train(X_train, y_train)
            
            # Test predictions  
            predictions = []
            for i in range(min(100, len(X_test))):  # Limit for performance
                pred = ensemble.predict(X_test.iloc[i])
                predictions.append(pred.predicted_outcome)
            
            accuracy = accuracy_score(y_test[:len(predictions)], predictions)
            
            # Save model
            model_path = f"models/ensemble_{method}_model.pkl"
            ensemble.save_model(model_path)
            
            results[method] = {
                'accuracy': accuracy,
                'model_path': model_path,
                'n_base_models': len(ensemble.base_models)
            }
            
            logger.info(f"✅ {method} ensemble: {accuracy:.4f} accuracy")
            
        except Exception as e:
            logger.error(f"❌ {method} ensemble training failed: {e}")
            results[method] = {'error': str(e)}
    
    return results

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train ML models for soccer prediction')
    parser.add_argument('--models', choices=['individual', 'ensemble', 'all'], 
                       default='all', help='Which models to train')
    parser.add_argument('--samples', type=int, default=1000, 
                       help='Number of training samples')
    
    args = parser.parse_args()
    
    logger = get_logger("ModelTraining")
    logger.info("🚀 Starting ML model training")
    logger.info(f"Training configuration: {args.models} models with {args.samples} samples")
    
    # Create training data
    logger.info("Generating training data...")
    data = create_synthetic_training_data(args.samples)
    
    # Prepare data
    X = data.drop(['result'], axis=1)
    y = data['result']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    all_results = {}
    trained_models = []
    
    # Train individual models
    if args.models in ['individual', 'all']:
        logger.info("\n" + "="*60)
        logger.info("TRAINING INDIVIDUAL MODELS")
        logger.info("="*60)
        
        individual_results, trained_models = train_individual_models(
            X_train, y_train, X_test, y_test
        )
        all_results['individual'] = individual_results
    
    # Train ensemble models
    if args.models in ['ensemble', 'all'] and trained_models:
        logger.info("\n" + "="*60)
        logger.info("TRAINING ENSEMBLE MODELS")
        logger.info("="*60)
        
        ensemble_results = train_ensemble_models(
            X_train, y_train, X_test, y_test, trained_models
        )
        all_results['ensemble'] = ensemble_results
    
    # Generate final report
    logger.info("\n" + "="*80)
    logger.info("🎯 TRAINING RESULTS SUMMARY")
    logger.info("="*80)
    
    best_accuracy = 0
    best_model = None
    
    for category, results in all_results.items():
        logger.info(f"\n{category.upper()} MODELS:")
        for model_name, result in results.items():
            if 'error' in result:
                logger.error(f"  ❌ {model_name}: {result['error']}")
            else:
                accuracy = result['accuracy']
                logger.info(f"  ✅ {model_name}: {accuracy:.4f}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = f"{category}_{model_name}"
    
    if best_model:
        logger.info(f"\n🏆 Best model: {best_model} ({best_accuracy:.4f})")
    
    # Save comprehensive report
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'configuration': {
            'models_trained': args.models,
            'training_samples': args.samples,
            'features': X_train.shape[1]
        },
        'results': all_results,
        'best_model': best_model,
        'best_accuracy': best_accuracy
    }
    
    report_path = "models/training_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"\n📊 Complete report saved to: {report_path}")
    logger.info("🎉 Training completed successfully!")

if __name__ == "__main__":
    # Create models directory
    Path("models").mkdir(exist_ok=True)
    main()