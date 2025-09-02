"""
Train ensemble model with synthetic data for testing and validation.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.config.logging_config import get_logger
from src.ml.models.ensemble import EnsemblePredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def create_synthetic_training_data() -> pd.DataFrame:
    """Create synthetic training data with diverse features."""
    logger = get_logger("SyntheticTrainingData")
    logger.info("Creating synthetic training data for ensemble")
    
    np.random.seed(42)
    n_samples = 1000  # More samples for better training
    
    # Create realistic soccer match features
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
        'home_corners': np.random.poisson(5, n_samples),
        'away_corners': np.random.poisson(5, n_samples),
        'home_fouls': np.random.poisson(12, n_samples),
        'away_fouls': np.random.poisson(12, n_samples),
        'home_yellow_cards': np.random.poisson(2, n_samples),
        'away_yellow_cards': np.random.poisson(2, n_samples),
        
        # Derived features
        'goal_difference': np.random.normal(0, 1.5, n_samples),
        'shots_ratio': np.random.lognormal(0, 0.3, n_samples),
        'possession_advantage': np.random.normal(0, 10, n_samples),
        'home_advantage': np.random.normal(0.1, 0.05, n_samples),
        
        # Rolling averages (simulated)
        'home_goals_avg_3': np.random.normal(1.3, 0.3, n_samples),
        'away_goals_avg_3': np.random.normal(1.1, 0.3, n_samples),
        'home_goals_avg_5': np.random.normal(1.3, 0.25, n_samples),
        'away_goals_avg_5': np.random.normal(1.1, 0.25, n_samples),
        'home_form_3': np.random.normal(1.5, 0.5, n_samples),
        'away_form_3': np.random.normal(1.5, 0.5, n_samples),
        'home_form_5': np.random.normal(1.5, 0.4, n_samples),
        'away_form_5': np.random.normal(1.5, 0.4, n_samples),
        
        # Expected goals features
        'home_xg': np.random.normal(1.4, 0.4, n_samples),
        'away_xg': np.random.normal(1.2, 0.4, n_samples),
        'xg_difference': np.random.normal(0.2, 0.3, n_samples),
        'home_xg_avg_3': np.random.normal(1.4, 0.3, n_samples),
        'away_xg_avg_3': np.random.normal(1.2, 0.3, n_samples),
        
        # Efficiency metrics
        'home_shot_efficiency': np.random.beta(2, 5, n_samples),
        'away_shot_efficiency': np.random.beta(2, 5, n_samples),
        'home_defense_efficiency': np.random.beta(3, 2, n_samples),
        'away_defense_efficiency': np.random.beta(3, 2, n_samples),
        
        # Momentum and streaks
        'home_momentum': np.random.normal(0, 1, n_samples),
        'away_momentum': np.random.normal(0, 1, n_samples),
        'home_win_streak': np.random.poisson(1, n_samples),
        'away_win_streak': np.random.poisson(1, n_samples),
        
        # Contextual features
        'is_derby': np.random.binomial(1, 0.1, n_samples),
        'travel_distance': np.random.exponential(100, n_samples),
        'days_since_last_match': np.random.poisson(7, n_samples),
        'season_progress': np.random.uniform(0, 1, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Ensure possession adds to 100
    total_poss = df['home_possession'] + df['away_possession']
    df['home_possession'] = (df['home_possession'] / total_poss) * 100
    df['away_possession'] = 100 - df['home_possession']
    
    # Create realistic result distribution
    results = []
    for i in range(n_samples):
        # Use multiple factors to determine result
        home_advantage = df.loc[i, 'home_advantage']
        goal_diff = df.loc[i, 'goal_difference']
        momentum_diff = df.loc[i, 'home_momentum'] - df.loc[i, 'away_momentum']
        xg_diff = df.loc[i, 'xg_difference']
        
        # Combine factors with some randomness
        home_prob = 0.45 + home_advantage + 0.1 * goal_diff + 0.05 * momentum_diff + 0.05 * xg_diff
        draw_prob = 0.25 - 0.05 * abs(goal_diff)  # Less likely with big goal difference
        away_prob = 1 - home_prob - draw_prob
        
        # Ensure probabilities are valid
        probs = np.array([home_prob, draw_prob, away_prob])
        probs = np.maximum(probs, 0.05)  # Minimum probability
        probs = probs / probs.sum()  # Normalize
        
        result = np.random.choice(['H', 'D', 'A'], p=probs)
        results.append(result)
    
    df['result'] = results
    
    logger.info(f"Created synthetic training data with {len(df)} samples and {len(df.columns)} features")
    logger.info(f"Result distribution: {pd.Series(results).value_counts().to_dict()}")
    
    return df

def train_ensemble_models():
    """Train and test ensemble models."""
    logger = get_logger("EnsembleTrainer")
    logger.info("Starting ensemble model training")
    
    # Create training data
    logger.info("Generating synthetic training data...")
    data = create_synthetic_training_data()
    
    # Prepare features and target
    X = data.drop(['result'], axis=1)
    y = data['result']
    
    logger.info(f"Training data shape: {X.shape}")
    logger.info(f"Features: {list(X.columns)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # Test different ensemble methods
    ensemble_methods = ['voting', 'stacking', 'dynamic']
    results = {}
    
    for method in ensemble_methods:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {method.upper()} ensemble")
        logger.info(f"{'='*60}")
        
        try:
            # Create ensemble
            ensemble = EnsemblePredictor(
                ensemble_method=method,
                random_state=42
            )
            
            # Train ensemble
            logger.info("Training ensemble (this may take a few minutes)...")
            ensemble.train(X_train, y_train)
            
            # Make predictions
            logger.info("Making predictions on test set...")
            test_predictions = []
            prediction_times = []
            
            import time
            for i in range(len(X_test)):
                start_time = time.time()
                pred = ensemble.predict(X_test.iloc[i])
                prediction_times.append(time.time() - start_time)
                test_predictions.append(pred.predicted_outcome)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, test_predictions)
            avg_prediction_time = np.mean(prediction_times)
            
            # Get confidence scores
            confidence_scores = []
            for i in range(min(20, len(X_test))):  # Sample of predictions
                pred = ensemble.predict(X_test.iloc[i])
                confidence_scores.append(pred.confidence)
            
            avg_confidence = np.mean(confidence_scores)
            
            results[method] = {
                'accuracy': accuracy,
                'avg_prediction_time': avg_prediction_time,
                'avg_confidence': avg_confidence,
                'model_info': ensemble.get_model_info(),
                'n_base_models_trained': len([m for m in ensemble.base_models if m.is_trained])
            }
            
            # Save model
            model_path = f"models/ensemble_{method}_model.pkl"
            ensemble.save_model(model_path)
            results[method]['model_path'] = model_path
            
            logger.info(f"✅ {method.upper()} ensemble completed:")
            logger.info(f"   Accuracy: {accuracy:.4f}")
            logger.info(f"   Avg confidence: {avg_confidence:.4f}")
            logger.info(f"   Avg prediction time: {avg_prediction_time*1000:.2f}ms")
            logger.info(f"   Base models trained: {results[method]['n_base_models_trained']}")
            logger.info(f"   Model saved to: {model_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to train {method} ensemble: {e}")
            results[method] = {'error': str(e)}
    
    # Generate final report
    logger.info(f"\n{'='*80}")
    logger.info("ENSEMBLE TRAINING RESULTS SUMMARY")
    logger.info(f"{'='*80}")
    
    successful_models = []
    for method, result in results.items():
        if 'error' not in result:
            successful_models.append((method, result['accuracy']))
            logger.info(f"{method.upper():15} - Accuracy: {result['accuracy']:.4f}")
        else:
            logger.error(f"{method.upper():15} - FAILED: {result['error']}")
    
    if successful_models:
        best_method, best_accuracy = max(successful_models, key=lambda x: x[1])
        logger.info(f"\n🏆 Best performing ensemble: {best_method.upper()} ({best_accuracy:.4f})")
    
    # Save comprehensive report
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'training_data_shape': list(X.shape),
        'test_data_shape': list(X_test.shape),
        'ensemble_results': results,
        'best_model': best_method if successful_models else None,
        'best_accuracy': best_accuracy if successful_models else None
    }
    
    report_path = "models/ensemble_training_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"\n📊 Comprehensive report saved to: {report_path}")
    logger.info("Ensemble training completed!")

if __name__ == "__main__":
    train_ensemble_models()