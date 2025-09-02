"""
Test script for trained ML models validation and performance evaluation.

Tests XGBoost, LightGBM, RandomForest, and ensemble predictions
with real data and comprehensive metrics analysis.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from typing import Dict, List, Any
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.config.logging_config import get_logger
from src.ml.models.predictors import XGBoostPredictor, LightGBMPredictor, RandomForestPredictor
from src.ml.models.ensemble import EnsemblePredictor
from src.ml.features.engine import FeatureEngineeringEngine
from src.ml.data.loaders import DataLoaderFactory
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def load_test_data() -> pd.DataFrame:
    """Load processed match data for testing."""
    logger = get_logger("TestDataLoader")
    
    try:
        # Use direct JSON loading since DataLoader interface may have issues
        data_files = []
        data_dir = Path("data/raw")
        
        if data_dir.exists():
            for season_dir in sorted(data_dir.glob("*")):
                if season_dir.is_dir():
                    season_files = list(season_dir.glob("*.json"))
                    data_files.extend(season_files[-50:])  # Last 50 files per season
                    
        if not data_files:
            logger.warning("No data files found, creating synthetic data")
            return create_synthetic_data()
            
        # Load and combine data using direct JSON loading
        all_matches = []
        for file_path in data_files[:200]:  # Limit to avoid memory issues
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_matches.extend(data)
                    elif isinstance(data, dict):
                        all_matches.append(data)
            except Exception as e:
                logger.warning(f"Could not load {file_path}: {e}")
                continue
                
        if not all_matches:
            logger.warning("No valid matches found, creating synthetic data")
            return create_synthetic_data()
            
        logger.info(f"Loaded {len(all_matches)} matches for testing")
        return pd.DataFrame(all_matches)
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return create_synthetic_data()

def create_synthetic_data() -> pd.DataFrame:
    """Create synthetic match data for testing."""
    logger = get_logger("SyntheticData")
    logger.info("Creating synthetic test data")
    
    np.random.seed(42)
    n_matches = 500
    
    # Team names
    teams = [f"Team_{i}" for i in range(1, 21)]
    
    matches = []
    for i in range(n_matches):
        home_team = np.random.choice(teams)
        away_team = np.random.choice([t for t in teams if t != home_team])
        
        match = {
            'home_team': home_team,
            'away_team': away_team,
            'home_goals': np.random.poisson(1.3),
            'away_goals': np.random.poisson(1.1),
            'home_shots': np.random.poisson(12),
            'away_shots': np.random.poisson(10),
            'home_shots_on_target': np.random.poisson(5),
            'away_shots_on_target': np.random.poisson(4),
            'home_possession': np.random.normal(50, 15),
            'away_possession': None,  # Will be calculated as 100 - home
            'home_corners': np.random.poisson(5),
            'away_corners': np.random.poisson(5),
            'home_fouls': np.random.poisson(12),
            'away_fouls': np.random.poisson(12),
            'home_yellow_cards': np.random.poisson(2),
            'away_yellow_cards': np.random.poisson(2),
            'home_red_cards': np.random.poisson(0.1),
            'away_red_cards': np.random.poisson(0.1),
            'season': '2024-2025',
            'date': f"2024-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}"
        }
        
        # Calculate away possession
        match['away_possession'] = 100 - match['home_possession']
        
        # Calculate match result
        if match['home_goals'] > match['away_goals']:
            match['result'] = 'H'
        elif match['home_goals'] < match['away_goals']:
            match['result'] = 'A'
        else:
            match['result'] = 'D'
            
        matches.append(match)
    
    return pd.DataFrame(matches)

def test_individual_models(X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict]:
    """Test individual ML models."""
    logger = get_logger("ModelTester")
    results = {}
    
    # Test models
    models = [
        ("XGBoost", XGBoostPredictor()),
        ("LightGBM", LightGBMPredictor()),
        ("RandomForest", RandomForestPredictor())
    ]
    
    for model_name, model in models:
        logger.info(f"Testing {model_name} model...")
        
        try:
            # Check for saved model
            model_path = f"models/{model_name.lower()}_model.pkl"
            if Path(model_path).exists():
                logger.info(f"Loading saved {model_name} model")
                model.load_model(model_path)
            else:
                # Train model if not saved
                logger.info(f"Training {model_name} model")
                X_train, X_val, y_train, y_val = train_test_split(
                    X_test, y_test, test_size=0.3, random_state=42, stratify=y_test
                )
                model.train(X_train, y_train)
                model.save_model(model_path)
                
            # Test predictions
            predictions = model.predict_batch([row for _, row in X_test.iterrows()])
            predicted_outcomes = [pred.predicted_outcome for pred in predictions]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, predicted_outcomes)
            
            # Confidence scores
            confidences = [pred.confidence for pred in predictions]
            avg_confidence = np.mean(confidences)
            
            results[model_name] = {
                'accuracy': accuracy,
                'avg_confidence': avg_confidence,
                'predictions': len(predictions),
                'model_info': model.get_model_info(),
                'training_metrics': model.training_metrics
            }
            
            logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, Confidence: {avg_confidence:.4f}")
            
        except Exception as e:
            logger.error(f"Error testing {model_name}: {e}")
            results[model_name] = {'error': str(e)}
            
    return results

def test_simple_model() -> Dict[str, Any]:
    """Test the simple RF model if it exists."""
    logger = get_logger("SimpleModelTester")
    
    simple_model_path = "models/simple_rf_model.pkl"
    if not Path(simple_model_path).exists():
        logger.warning("Simple RF model not found")
        return {'error': 'Model file not found'}
    
    try:
        # Load the simple model
        with open(simple_model_path, 'rb') as f:
            simple_model_data = pickle.load(f)
            
        logger.info("Simple RF model loaded successfully")
        
        # Extract information about the model
        if isinstance(simple_model_data, dict):
            model_info = {
                'type': 'Dictionary containing model data',
                'keys': list(simple_model_data.keys()),
                'file_size_mb': Path(simple_model_path).stat().st_size / (1024 * 1024)
            }
        else:
            model_info = {
                'type': str(type(simple_model_data)),
                'file_size_mb': Path(simple_model_path).stat().st_size / (1024 * 1024)
            }
            
        return {
            'status': 'loaded_successfully',
            'info': model_info
        }
        
    except Exception as e:
        logger.error(f"Error loading simple model: {e}")
        return {'error': str(e)}

def generate_test_report(results: Dict[str, Any]) -> None:
    """Generate comprehensive test report."""
    logger = get_logger("ReportGenerator")
    
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'test_results': results,
        'summary': {
            'models_tested': len([k for k in results.keys() if 'error' not in results[k]]),
            'models_failed': len([k for k in results.keys() if 'error' in results[k]]),
        }
    }
    
    # Calculate best model
    successful_models = {k: v for k, v in results.items() if 'error' not in v and 'accuracy' in v}
    if successful_models:
        best_model = max(successful_models.items(), key=lambda x: x[1]['accuracy'])
        report['summary']['best_model'] = best_model[0]
        report['summary']['best_accuracy'] = best_model[1]['accuracy']
    
    # Save report
    report_path = "models/test_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Test report saved to {report_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("🧪 ML MODELS TEST RESULTS")
    print("="*80)
    
    for model_name, model_results in results.items():
        print(f"\n📊 {model_name}:")
        if 'error' in model_results:
            print(f"   ❌ Error: {model_results['error']}")
        elif 'accuracy' in model_results:
            print(f"   ✅ Accuracy: {model_results['accuracy']:.4f}")
            print(f"   🎯 Confidence: {model_results['avg_confidence']:.4f}")
            print(f"   📝 Predictions: {model_results['predictions']}")
            
            # Training metrics if available
            if 'training_metrics' in model_results and model_results['training_metrics']:
                metrics = model_results['training_metrics']
                if 'cv_accuracy_mean' in metrics:
                    print(f"   📈 CV Accuracy: {metrics['cv_accuracy_mean']:.4f} ± {metrics.get('cv_accuracy_std', 0):.4f}")
        else:
            print(f"   ℹ️ Status: {model_results.get('status', 'Unknown')}")
    
    if successful_models:
        print(f"\n🏆 Best Model: {report['summary']['best_model']} ({report['summary']['best_accuracy']:.4f})")
    
    print("\n" + "="*80)

def main():
    """Main testing function."""
    logger = get_logger("MLModelsTester")
    logger.info("Starting ML models testing")
    
    # Load test data
    logger.info("Loading test data...")
    raw_data = load_test_data()
    
    if raw_data.empty:
        logger.error("No data available for testing")
        return
    
    # Engineer features
    logger.info("Engineering features...")
    
    try:
        feature_engine = FeatureEngineeringEngine()
        processed_data = feature_engine.process(raw_data)
        
        # Prepare features and targets
        if 'result' not in processed_data.columns:
            logger.error("No 'result' column found in data")
            return
            
        # Split features and target
        X = processed_data.drop(['result'], axis=1, errors='ignore')
        y = processed_data['result']
        
        logger.info(f"Test data shape: {X.shape}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Test individual models
        logger.info("Testing individual models...")
        results = test_individual_models(X, y)
        
        # Test simple model
        logger.info("Testing simple model...")
        simple_results = test_simple_model()
        results['SimpleRF'] = simple_results
        
        # Generate report
        generate_test_report(results)
        
        logger.info("ML models testing completed successfully")
        
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        logger.info("Testing simple model only...")
        
        # Test just the simple model
        simple_results = test_simple_model()
        results = {'SimpleRF': simple_results}
        generate_test_report(results)

if __name__ == "__main__":
    main()