"""
Main prediction script for soccer match outcomes.

Unified interface for making predictions using all available models.
Supports individual models, ensemble methods, and batch predictions.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
from typing import Dict, List, Optional, Union
from datetime import datetime
import pickle

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.config.logging_config import get_logger
from src.ml.models.predictors import XGBoostPredictor, LightGBMPredictor, RandomForestPredictor
from src.ml.models.ensemble import EnsemblePredictor

class PredictionEngine:
    """Main prediction engine that manages all available models."""
    
    def __init__(self):
        self.logger = get_logger("PredictionEngine")
        self.models = {}
        self.ensemble_models = {}
        self.available_models = []
        self._load_available_models()
    
    def _load_available_models(self):
        """Load all available trained models."""
        models_dir = Path("models")
        if not models_dir.exists():
            self.logger.warning("Models directory not found")
            return
        
        # Individual models
        individual_models = {
            'xgboost': XGBoostPredictor(),
            'lightgbm': LightGBMPredictor(), 
            'randomforest': RandomForestPredictor()
        }
        
        for model_name, model_instance in individual_models.items():
            model_path = models_dir / f"{model_name}_model.pkl"
            if model_path.exists():
                try:
                    model_instance.load_model(str(model_path))
                    self.models[model_name] = model_instance
                    self.available_models.append(model_name)
                    self.logger.info(f"✅ Loaded {model_name} model")
                except Exception as e:
                    self.logger.error(f"❌ Failed to load {model_name}: {e}")
        
        # Ensemble models
        ensemble_methods = ['voting', 'stacking', 'dynamic']
        for method in ensemble_methods:
            model_path = models_dir / f"ensemble_{method}_model.pkl"
            if model_path.exists():
                try:
                    ensemble = EnsemblePredictor(ensemble_method=method)
                    ensemble.load_model(str(model_path))
                    self.ensemble_models[method] = ensemble
                    self.available_models.append(f"ensemble_{method}")
                    self.logger.info(f"✅ Loaded ensemble {method} model")
                except Exception as e:
                    self.logger.error(f"❌ Failed to load ensemble {method}: {e}")
        
        # Simple models (legacy)
        simple_model_path = models_dir / "simple_rf_model.pkl"
        if simple_model_path.exists():
            try:
                with open(simple_model_path, 'rb') as f:
                    simple_model = pickle.load(f)
                self.models['simple_rf'] = simple_model
                self.available_models.append('simple_rf')
                self.logger.info("✅ Loaded simple RF model")
            except Exception as e:
                self.logger.error(f"❌ Failed to load simple RF: {e}")
        
        if not self.available_models:
            self.logger.error("No trained models found!")
        else:
            self.logger.info(f"Available models: {', '.join(self.available_models)}")
    
    def predict_single(self, features: Dict, model_name: str = None) -> Dict:
        """Make prediction for a single match."""
        if not self.available_models:
            raise ValueError("No trained models available")
        
        # Use best available model if none specified
        if model_name is None:
            model_name = self._get_best_model()
        
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not available. Available: {self.available_models}")
        
        try:
            # Convert features to DataFrame
            features_df = pd.DataFrame([features])
            
            # Make prediction based on model type
            if model_name.startswith('ensemble_'):
                ensemble_method = model_name.split('_')[1]
                model = self.ensemble_models[ensemble_method]
                prediction = model.predict(features_df)
            elif model_name in self.models:
                model = self.models[model_name]
                if hasattr(model, 'predict'):  # Our custom predictors
                    prediction = model.predict(features_df)
                else:  # Simple sklearn model
                    pred_class = model.predict(features_df)[0]
                    pred_proba = model.predict_proba(features_df)[0] if hasattr(model, 'predict_proba') else [0.33, 0.33, 0.33]
                    
                    # Create prediction object
                    from src.ml.core.types import Prediction
                    prediction = Prediction(
                        home_win_prob=float(pred_proba[self._get_class_index('H', model.classes_)]),
                        draw_prob=float(pred_proba[self._get_class_index('D', model.classes_)]),
                        away_win_prob=float(pred_proba[self._get_class_index('A', model.classes_)]),
                        predicted_outcome=pred_class,
                        confidence=float(max(pred_proba)),
                        model_name=model_name
                    )
            
            return {
                'model_used': model_name,
                'predicted_outcome': prediction.predicted_outcome,
                'home_win_probability': prediction.home_win_prob,
                'draw_probability': prediction.draw_prob,
                'away_win_probability': prediction.away_win_prob,
                'confidence': prediction.confidence,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Prediction failed with {model_name}: {e}")
            raise
    
    def predict_batch(self, matches: List[Dict], model_name: str = None) -> List[Dict]:
        """Make predictions for multiple matches."""
        results = []
        for i, match in enumerate(matches):
            try:
                prediction = self.predict_single(match, model_name)
                prediction['match_index'] = i
                results.append(prediction)
            except Exception as e:
                self.logger.error(f"Failed to predict match {i}: {e}")
                results.append({
                    'match_index': i,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        return results
    
    def compare_models(self, features: Dict) -> Dict:
        """Compare predictions from all available models."""
        results = {}
        
        for model_name in self.available_models:
            try:
                prediction = self.predict_single(features, model_name)
                results[model_name] = prediction
            except Exception as e:
                results[model_name] = {'error': str(e)}
                self.logger.warning(f"Model {model_name} failed: {e}")
        
        return results
    
    def _get_best_model(self) -> str:
        """Get the best available model based on preference order."""
        preference_order = [
            'ensemble_stacking',
            'ensemble_voting', 
            'ensemble_dynamic',
            'xgboost',
            'lightgbm',
            'randomforest',
            'simple_rf'
        ]
        
        for model in preference_order:
            if model in self.available_models:
                return model
        
        return self.available_models[0] if self.available_models else None
    
    def _get_class_index(self, class_label: str, classes) -> int:
        """Get index of class label."""
        try:
            return list(classes).index(class_label)
        except (ValueError, AttributeError):
            return 0

def create_sample_match() -> Dict:
    """Create a sample match for demonstration."""
    np.random.seed(42)
    
    return {
        'home_goals': 2,
        'away_goals': 1,
        'home_shots': 15,
        'away_shots': 8,
        'home_shots_on_target': 6,
        'away_shots_on_target': 3,
        'home_possession': 65.0,
        'away_possession': 35.0,
        'home_xg': 1.8,
        'away_xg': 0.9,
        'goal_difference': 1.0,
        'shots_ratio': 1.875,
        'home_advantage': 0.15,
        'home_form_5': 2.1,
        'away_form_5': 1.2,
        'home_momentum': 0.8,
        'away_momentum': -0.3,
        **{f'feature_{i}': np.random.normal(0, 1) for i in range(25)}
    }

def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='Predict soccer match outcomes')
    parser.add_argument('--model', type=str, help='Specific model to use')
    parser.add_argument('--compare', action='store_true', help='Compare all models')
    parser.add_argument('--input', type=str, help='JSON file with match data')
    parser.add_argument('--output', type=str, help='Output file for predictions')
    parser.add_argument('--sample', action='store_true', help='Use sample match data')
    
    args = parser.parse_args()
    
    logger = get_logger("PredictionMain")
    logger.info("🎯 Starting soccer prediction system")
    
    # Initialize prediction engine
    engine = PredictionEngine()
    
    if not engine.available_models:
        logger.error("No models available. Please train models first.")
        return
    
    # Prepare match data
    if args.input:
        # Load from file
        with open(args.input, 'r') as f:
            match_data = json.load(f)
        
        if isinstance(match_data, list):
            matches = match_data
        else:
            matches = [match_data]
    elif args.sample:
        # Use sample data
        matches = [create_sample_match()]
        logger.info("Using sample match data")
    else:
        # Interactive input
        logger.info("No input specified. Using sample match data.")
        matches = [create_sample_match()]
    
    # Make predictions
    results = []
    
    for i, match in enumerate(matches):
        logger.info(f"\nPredicting match {i+1}/{len(matches)}")
        
        if args.compare:
            # Compare all models
            comparison = engine.compare_models(match)
            results.append({
                'match_index': i,
                'match_data': match,
                'model_comparison': comparison
            })
            
            # Print comparison
            print(f"\n📊 Match {i+1} - Model Comparison:")
            for model_name, prediction in comparison.items():
                if 'error' in prediction:
                    print(f"  ❌ {model_name}: ERROR - {prediction['error']}")
                else:
                    outcome = prediction['predicted_outcome']
                    confidence = prediction['confidence']
                    print(f"  ✅ {model_name}: {outcome} ({confidence:.3f})")
        else:
            # Single model prediction
            prediction = engine.predict_single(match, args.model)
            results.append({
                'match_index': i,
                'match_data': match,
                'prediction': prediction
            })
            
            # Print prediction
            print(f"\n🎯 Match {i+1} Prediction:")
            print(f"  Model: {prediction['model_used']}")
            print(f"  Outcome: {prediction['predicted_outcome']}")
            print(f"  Confidence: {prediction['confidence']:.3f}")
            print(f"  Probabilities:")
            print(f"    Home Win: {prediction['home_win_probability']:.3f}")
            print(f"    Draw: {prediction['draw_probability']:.3f}")
            print(f"    Away Win: {prediction['away_win_probability']:.3f}")
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {args.output}")
    
    logger.info("🏁 Prediction completed!")

if __name__ == "__main__":
    main()