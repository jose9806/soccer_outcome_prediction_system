"""
Simple test script for ML models without feature engineering dependencies.
Tests the trained models with synthetic data.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from typing import Dict, List, Any
import json
from sklearn.metrics import accuracy_score, classification_report

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.config.logging_config import get_logger

def create_synthetic_test_data() -> pd.DataFrame:
    """Create synthetic test data with features matching model expectations."""
    logger = get_logger("SyntheticData")
    logger.info("Creating synthetic test data for models")
    
    np.random.seed(42)
    n_samples = 100
    
    # Create basic features that should be present after feature engineering
    data = {
        # Basic match stats
        'home_goals': np.random.poisson(1.3, n_samples),
        'away_goals': np.random.poisson(1.1, n_samples),
        'home_shots': np.random.poisson(12, n_samples),
        'away_shots': np.random.poisson(10, n_samples),
        'home_shots_on_target': np.random.poisson(5, n_samples),
        'away_shots_on_target': np.random.poisson(4, n_samples),
        'home_possession': np.random.normal(50, 15, n_samples),
        'away_possession': np.random.normal(50, 15, n_samples),
        
        # Additional synthetic features that might be created by feature engineering
        'goal_difference': np.random.normal(0, 1.5, n_samples),
        'shots_ratio': np.random.normal(1.2, 0.3, n_samples),
        'possession_advantage': np.random.normal(0, 10, n_samples),
        'home_advantage': np.random.normal(0.1, 0.05, n_samples),
        
        # Rolling averages (synthetic)
        'home_goals_avg_5': np.random.normal(1.3, 0.3, n_samples),
        'away_goals_avg_5': np.random.normal(1.1, 0.3, n_samples),
        'home_form_5': np.random.normal(1.5, 0.5, n_samples),
        'away_form_5': np.random.normal(1.5, 0.5, n_samples),
        
        # Expected goals features
        'home_xg': np.random.normal(1.4, 0.4, n_samples),
        'away_xg': np.random.normal(1.2, 0.4, n_samples),
        'xg_difference': np.random.normal(0.2, 0.3, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create result column
    results = []
    for i in range(n_samples):
        home_goals = df.loc[i, 'home_goals']
        away_goals = df.loc[i, 'away_goals']
        
        if home_goals > away_goals:
            results.append('H')
        elif home_goals < away_goals:
            results.append('A')
        else:
            results.append('D')
    
    df['result'] = results
    
    logger.info(f"Created synthetic data with {len(df)} samples and {len(df.columns)} features")
    logger.info(f"Result distribution: {pd.Series(results).value_counts().to_dict()}")
    
    return df

def test_simple_model() -> Dict[str, Any]:
    """Test the simple RF model."""
    logger = get_logger("SimpleModelTester")
    
    simple_model_path = "models/simple_rf_model.pkl"
    if not Path(simple_model_path).exists():
        logger.error("Simple RF model not found")
        return {'error': 'Model file not found'}
    
    try:
        # Load the simple model
        with open(simple_model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        logger.info("Simple RF model loaded successfully")
        
        # Create test data
        test_data = create_synthetic_test_data()
        
        # Try to make predictions if it's a trained model
        if hasattr(model_data, 'predict'):
            # It's a trained model
            X_test = test_data.drop(['result'], axis=1, errors='ignore')
            y_test = test_data['result']
            
            try:
                # Try to predict
                predictions = model_data.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                
                return {
                    'status': 'predictions_successful',
                    'accuracy': accuracy,
                    'predictions_count': len(predictions),
                    'test_samples': len(X_test),
                    'features_used': X_test.shape[1],
                    'model_type': str(type(model_data).__name__)
                }
                
            except Exception as e:
                logger.warning(f"Could not make predictions: {e}")
                return {
                    'status': 'loaded_but_prediction_failed',
                    'error': str(e),
                    'model_type': str(type(model_data).__name__),
                    'file_size_mb': Path(simple_model_path).stat().st_size / (1024 * 1024)
                }
        else:
            # It's model data dictionary
            return {
                'status': 'loaded_successfully',
                'type': 'Dictionary containing model data' if isinstance(model_data, dict) else str(type(model_data)),
                'keys': list(model_data.keys()) if isinstance(model_data, dict) else 'N/A',
                'file_size_mb': Path(simple_model_path).stat().st_size / (1024 * 1024)
            }
        
    except Exception as e:
        logger.error(f"Error loading simple model: {e}")
        return {'error': str(e)}

def test_advanced_models() -> Dict[str, Any]:
    """Test advanced ML models if they exist."""
    logger = get_logger("AdvancedModelTester")
    results = {}
    
    # Look for model files
    models_dir = Path("models")
    model_files = {
        'XGBoost': ['xgboost_model.pkl', 'xgb_model.pkl'],
        'LightGBM': ['lightgbm_model.pkl', 'lgb_model.pkl'],
        'RandomForest': ['randomforest_model.pkl', 'rf_model.pkl']
    }
    
    for model_name, possible_files in model_files.items():
        model_found = False
        for file_name in possible_files:
            model_path = models_dir / file_name
            if model_path.exists():
                logger.info(f"Found {model_name} model at {model_path}")
                
                try:
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    results[model_name] = {
                        'status': 'loaded_successfully',
                        'file_path': str(model_path),
                        'file_size_mb': model_path.stat().st_size / (1024 * 1024),
                        'model_type': str(type(model_data).__name__)
                    }
                    
                    # Try to make predictions if possible
                    if hasattr(model_data, 'predict'):
                        test_data = create_synthetic_test_data()
                        X_test = test_data.drop(['result'], axis=1, errors='ignore')
                        y_test = test_data['result']
                        
                        try:
                            predictions = model_data.predict(X_test)
                            accuracy = accuracy_score(y_test, predictions)
                            results[model_name].update({
                                'predictions_successful': True,
                                'accuracy': accuracy,
                                'test_samples': len(X_test)
                            })
                        except Exception as e:
                            results[model_name]['prediction_error'] = str(e)
                    
                    model_found = True
                    break
                    
                except Exception as e:
                    results[model_name] = {
                        'status': 'load_error',
                        'error': str(e),
                        'file_path': str(model_path)
                    }
                    model_found = True
                    break
        
        if not model_found:
            results[model_name] = {'status': 'not_found'}
    
    return results

def generate_simple_report(simple_results: Dict, advanced_results: Dict) -> None:
    """Generate simple test report."""
    logger = get_logger("ReportGenerator")
    
    # Combine results
    all_results = {
        'SimpleRF': simple_results,
        **advanced_results
    }
    
    # Create report
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'test_type': 'simple_model_validation',
        'results': all_results,
        'summary': {
            'total_models_found': len([k for k, v in all_results.items() if 'error' not in v and 'not_found' not in v.get('status', '')]),
            'successful_predictions': len([k for k, v in all_results.items() if v.get('predictions_successful', False)]),
        }
    }
    
    # Save report
    report_path = "models/simple_test_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Test report saved to {report_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("🧪 SIMPLE ML MODELS TEST RESULTS")
    print("="*80)
    
    for model_name, results in all_results.items():
        print(f"\n📊 {model_name}:")
        
        status = results.get('status', 'unknown')
        
        if 'error' in results:
            print(f"   ❌ Error: {results['error']}")
        elif status == 'not_found':
            print(f"   ⚠️ Model file not found")
        elif status == 'loaded_successfully':
            print(f"   ✅ Model loaded successfully")
            if 'file_size_mb' in results:
                print(f"   📦 Size: {results['file_size_mb']:.2f} MB")
            if 'model_type' in results:
                print(f"   🏷️ Type: {results['model_type']}")
        elif status == 'predictions_successful':
            print(f"   ✅ Predictions successful")
            print(f"   🎯 Accuracy: {results['accuracy']:.4f}")
            print(f"   📝 Test samples: {results['test_samples']}")
            print(f"   🔢 Features: {results.get('features_used', 'unknown')}")
        
        if results.get('predictions_successful') and 'accuracy' in results:
            accuracy = results['accuracy']
            if accuracy > 0.6:
                print(f"   🏆 Excellent performance!")
            elif accuracy > 0.5:
                print(f"   👍 Good performance!")
            else:
                print(f"   ⚠️ Performance needs improvement")
    
    print(f"\n📊 Summary:")
    print(f"   Models found: {report['summary']['total_models_found']}")
    print(f"   Successful predictions: {report['summary']['successful_predictions']}")
    
    print("\n" + "="*80)

def main():
    """Main testing function."""
    logger = get_logger("SimpleMLTester")
    logger.info("Starting simple ML models testing")
    
    # Test simple model
    logger.info("Testing simple model...")
    simple_results = test_simple_model()
    
    # Test advanced models
    logger.info("Testing advanced models...")
    advanced_results = test_advanced_models()
    
    # Generate report
    generate_simple_report(simple_results, advanced_results)
    
    logger.info("Simple ML testing completed")

if __name__ == "__main__":
    main()