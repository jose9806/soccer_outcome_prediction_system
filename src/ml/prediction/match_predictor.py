"""
Real-time Match Predictor

Predicts soccer match outcomes using only team names as input.
Handles feature generation, model loading, and ensemble predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
import glob
import json

from src.config.logging_config import get_logger
from ..core.types import MatchFeatures, Prediction, MatchOutcome
from ..core.exceptions import ModelPredictionError, DataProcessingError
from ..features.engine import FeatureEngineeringEngine
from ..models.trainer import ModelTrainer


class MatchPredictor:
    """
    High-level interface for match outcome prediction.
    
    Takes team names as input and handles all the complexity of:
    - Loading historical data
    - Generating features in real-time
    - Loading trained models
    - Making ensemble predictions
    - Providing betting recommendations
    """
    
    def __init__(self, models_dir: str = "data/models", data_dir: str = "data/raw"):
        """Initialize the match predictor."""
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir) 
        self.logger = get_logger("MatchPredictor")
        
        # Initialize components
        self.feature_engine = None
        self.models = {}
        self.team_mappings = {}
        self.historical_data = None
        
        # Load system components
        self._load_components()
    
    def _load_components(self):
        """Load feature engine, models, and historical data."""
        try:
            self.logger.info("Loading prediction system components...")
            
            # Load feature engineering engine
            self.feature_engine = FeatureEngineeringEngine()
            
            # Load historical data for feature generation
            self._load_historical_data()
            
            # Load trained models
            self._load_models()
            
            self.logger.info("✅ All components loaded successfully")
            
        except Exception as e:
            raise ModelPredictionError(f"Failed to load predictor components: {e}")
    
    def _load_historical_data(self):
        """Load and index historical match data."""
        try:
            self.logger.info("Loading historical match data...")
            
            # Load all JSON files from data directory
            all_matches = []
            json_files = list(self.data_dir.glob("**/*.json"))
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        match_data = json.load(f)
                        all_matches.append(match_data)
                except Exception as e:
                    self.logger.warning(f"Failed to load {json_file}: {e}")
            
            if not all_matches:
                raise DataProcessingError("No historical match data found")
            
            # Convert to DataFrame
            self.historical_data = pd.DataFrame(all_matches)
            
            # Create team mappings (normalize team names)
            self._create_team_mappings()
            
            self.logger.info(f"Loaded {len(all_matches)} historical matches")
            
        except Exception as e:
            raise DataProcessingError(f"Failed to load historical data: {e}")
    
    def _create_team_mappings(self):
        """Create team name mappings for flexible matching."""
        teams = set()
        
        # Extract all team names
        for _, match in self.historical_data.iterrows():
            home_team = match.get('home_team', '')
            away_team = match.get('away_team', '')
            
            # Only add valid team names
            if home_team and not pd.isna(home_team) and isinstance(home_team, str) and home_team.strip():
                teams.add(home_team.strip())
            if away_team and not pd.isna(away_team) and isinstance(away_team, str) and away_team.strip():
                teams.add(away_team.strip())
        
        # Create normalized mappings
        for team in teams:
            normalized = self._normalize_team_name(team)
            if normalized:  # Only create mappings for valid normalized names
                self.team_mappings[normalized] = team
                self.team_mappings[team.lower()] = team
                self.team_mappings[team] = team
        
        self.logger.info(f"Created mappings for {len(teams)} teams")
    
    def _normalize_team_name(self, team_name: str) -> str:
        """Normalize team name for flexible matching."""
        if not team_name or pd.isna(team_name):
            return ""
        if not isinstance(team_name, str):
            team_name = str(team_name)
        return team_name.lower().strip().replace(' ', '')
    
    def _load_models(self):
        """Load trained ML models."""
        try:
            self.logger.info("Loading trained models...")
            
            # Look for the most recent training session
            model_files = list(self.models_dir.glob("*_model_*.pkl"))
            
            if not model_files:
                self.logger.warning("No trained models found. Run training first.")
                return
            
            # Group by model type
            model_types = {}
            for model_file in model_files:
                parts = model_file.stem.split('_')
                if len(parts) >= 3:
                    model_type = parts[0]  # e.g., 'xgboost', 'lightgbm'
                    if model_type not in model_types:
                        model_types[model_type] = []
                    model_types[model_type].append(model_file)
            
            # Load the most recent model of each type
            for model_type, files in model_types.items():
                latest_file = max(files, key=lambda x: x.stat().st_mtime)
                
                try:
                    # Import the appropriate predictor class
                    if model_type == 'xgboost':
                        from ..models.predictors import XGBoostPredictor
                        model = XGBoostPredictor()
                    elif model_type == 'lightgbm':
                        from ..models.predictors import LightGBMPredictor
                        model = LightGBMPredictor()
                    elif model_type == 'randomforest':
                        from ..models.predictors import RandomForestPredictor
                        model = RandomForestPredictor()
                    else:
                        continue
                    
                    # Load the model
                    model.load_model(str(latest_file))
                    self.models[model_type] = model
                    
                    self.logger.info(f"Loaded {model_type} model from {latest_file.name}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load {model_type} model: {e}")
            
            if not self.models:
                raise ModelPredictionError("No models could be loaded")
                
        except Exception as e:
            raise ModelPredictionError(f"Failed to load models: {e}")
    
    def predict_match(
        self, 
        home_team: str, 
        away_team: str, 
        match_date: Optional[datetime] = None,
        competition: str = ""
    ) -> Prediction:
        """
        Predict match outcome using team names.
        
        Args:
            home_team: Home team name
            away_team: Away team name  
            match_date: Match date (defaults to today)
            competition: Competition name
            
        Returns:
            Prediction object with probabilities and recommendations
        """
        try:
            # Normalize team names
            home_team_key = self._find_team_match(home_team)
            away_team_key = self._find_team_match(away_team)
            
            if not home_team_key:
                raise ModelPredictionError(f"Home team '{home_team}' not found in historical data")
            if not away_team_key:
                raise ModelPredictionError(f"Away team '{away_team}' not found in historical data")
            
            # Use today's date if not specified
            if match_date is None:
                match_date = datetime.now()
            
            self.logger.info(f"Generating features for {home_team_key} vs {away_team_key}")
            
            # Generate features for this match
            features = self._generate_match_features(
                home_team_key, away_team_key, match_date, competition
            )
            
            # Make prediction using best available model
            prediction = self._make_ensemble_prediction(features, match_date)
            
            self.logger.info(f"✅ Prediction completed: {prediction.predicted_outcome} ({prediction.confidence:.1%})")
            
            return prediction
            
        except Exception as e:
            raise ModelPredictionError(f"Match prediction failed: {e}")
    
    def _find_team_match(self, team_name: str) -> Optional[str]:
        """Find matching team name in historical data."""
        # Direct match
        if team_name in self.team_mappings:
            return self.team_mappings[team_name]
        
        # Normalized match
        normalized = self._normalize_team_name(team_name)
        if normalized in self.team_mappings:
            return self.team_mappings[normalized]
        
        # Partial match
        for key, mapped_name in self.team_mappings.items():
            if normalized in self._normalize_team_name(key):
                return mapped_name
            if self._normalize_team_name(key) in normalized:
                return mapped_name
        
        return None
    
    def _generate_match_features(
        self, 
        home_team: str, 
        away_team: str, 
        match_date: datetime,
        competition: str
    ) -> pd.DataFrame:
        """Generate features for the specific match."""
        try:
            # Filter historical data up to match date
            cutoff_date = match_date.strftime('%Y-%m-%d')
            historical_cutoff = self.historical_data[
                pd.to_datetime(self.historical_data['date']).dt.strftime('%Y-%m-%d') < cutoff_date
            ]
            
            if historical_cutoff.empty:
                raise DataProcessingError("No historical data available before match date")
            
            # Create a synthetic match record for feature generation
            match_record = {
                'match_id': f"PRED_{home_team}_{away_team}_{match_date.strftime('%Y%m%d')}",
                'date': match_date.strftime('%Y-%m-%dT%H:%M:%S'),
                'home_team': home_team,
                'away_team': away_team,
                'competition': competition,
                'season': str(match_date.year),
                'stage': 'Regular Season',
                'status': 'SCHEDULED'
            }
            
            # Add to historical data temporarily
            temp_data = pd.concat([
                historical_cutoff,
                pd.DataFrame([match_record])
            ], ignore_index=True)
            
            # Generate features using the feature engine
            processed_data = self.feature_engine.create_features(temp_data)
            
            # Debug: check available columns
            self.logger.debug(f"Available columns after feature engineering: {list(processed_data.columns)}")
            
            # Extract features for our prediction match
            # Use iloc to get the last row (our synthetic prediction match)
            match_features = processed_data.iloc[[-1]]  # Last row is our prediction match
            
            if match_features.empty:
                raise DataProcessingError("Failed to generate features for match")
            
            # Prepare feature columns (same as training)
            exclude_columns = [
                'match_id', 'date', 'home_team', 'away_team', 'home_score', 'away_score',
                'season', 'competition', 'stage', 'status', 'file_path', 'year', 'result'
            ]
            
            feature_columns = [col for col in match_features.columns if col not in exclude_columns]
            features_df = match_features[feature_columns].copy()
            
            # Handle missing values
            numeric_columns = features_df.select_dtypes(include=[np.number]).columns
            categorical_columns = features_df.select_dtypes(exclude=[np.number]).columns
            
            # Fill numeric missing values with median from historical data
            for col in numeric_columns:
                if col in processed_data.columns:
                    median_val = processed_data[col].median()
                    features_df[col] = features_df[col].fillna(median_val)
                else:
                    features_df[col] = features_df[col].fillna(0)
            
            # Fill categorical missing values
            for col in categorical_columns:
                features_df[col] = features_df[col].fillna('unknown')
            
            self.logger.info(f"Generated {features_df.shape[1]} features for prediction")
            
            return features_df
            
        except Exception as e:
            raise DataProcessingError(f"Feature generation failed: {e}")
    
    def _make_ensemble_prediction(self, features: pd.DataFrame, match_date: datetime) -> Prediction:
        """Make ensemble prediction using available models."""
        try:
            if not self.models:
                raise ModelPredictionError("No trained models available")
            
            predictions = {}
            all_probabilities = []
            
            # Get predictions from all available models
            for model_name, model in self.models.items():
                try:
                    # Convert to MatchFeatures format if needed
                    feature_row = features.iloc[0]
                    
                    # Create MatchFeatures object (simplified)
                    match_features = MatchFeatures(
                        match_id=f"pred_{int(match_date.timestamp())}",
                        home_strength=feature_row.get('home_strength', 0.5),
                        away_strength=feature_row.get('away_strength', 0.5),
                        strength_diff=feature_row.get('strength_diff', 0.0),
                        home_form=feature_row.get('home_form', 0.5),
                        away_form=feature_row.get('away_form', 0.5),
                        h2h_home_wins=int(feature_row.get('h2h_home_wins', 0)),
                        h2h_draws=int(feature_row.get('h2h_draws', 0)),
                        h2h_away_wins=int(feature_row.get('h2h_away_wins', 0)),
                        h2h_avg_goals=feature_row.get('h2h_avg_goals', 2.5),
                        home_advantage=feature_row.get('home_advantage', 0.1),
                        rest_days_home=int(feature_row.get('rest_days_home', 7)),
                        rest_days_away=int(feature_row.get('rest_days_away', 7)),
                        features={col: val for col, val in feature_row.items() 
                                if col not in ['home_strength', 'away_strength', 'strength_diff',
                                             'home_form', 'away_form', 'h2h_home_wins', 'h2h_draws',
                                             'h2h_away_wins', 'h2h_avg_goals', 'home_advantage',
                                             'rest_days_home', 'rest_days_away']}
                    )
                    
                    # Make prediction
                    pred = model.predict(match_features)
                    predictions[model_name] = pred
                    
                    # Collect probabilities for ensemble
                    prob_values = list(pred.probabilities.values())
                    all_probabilities.append(prob_values)
                    
                    self.logger.debug(f"{model_name} prediction: {pred.predicted_outcome} ({pred.confidence:.3f})")
                    
                except Exception as e:
                    self.logger.warning(f"Model {model_name} prediction failed: {e}")
            
            if not predictions:
                raise ModelPredictionError("All model predictions failed")
            
            # Create ensemble prediction
            if len(predictions) == 1:
                # Single model
                return list(predictions.values())[0]
            else:
                # Average probabilities across models
                avg_probs = np.mean(all_probabilities, axis=0)
                
                # Map to outcomes
                outcomes = [MatchOutcome.HOME_WIN, MatchOutcome.DRAW, MatchOutcome.AWAY_WIN]
                ensemble_probs = {outcomes[i]: float(avg_probs[i]) for i in range(len(outcomes))}
                
                # Get best prediction
                best_outcome = max(ensemble_probs, key=ensemble_probs.get)
                confidence = float(max(avg_probs))
                
                # Use the best individual model's additional predictions
                best_pred = max(predictions.values(), key=lambda p: p.confidence)
                
                return Prediction(
                    match_id=f"ensemble_pred_{int(match_date.timestamp())}",
                    model_name=f"Ensemble({len(predictions)} models)",
                    timestamp=datetime.now(),
                    probabilities=ensemble_probs,
                    predicted_outcome=best_outcome,
                    confidence=confidence,
                    expected_goals_home=best_pred.expected_goals_home,
                    expected_goals_away=best_pred.expected_goals_away,
                    over_under_2_5=best_pred.over_under_2_5,
                    both_teams_score=best_pred.both_teams_score,
                    model_version="ensemble_v1.0",
                    features_used=[f"ensemble_of_{len(predictions)}_models"]
                )
                
        except Exception as e:
            raise ModelPredictionError(f"Ensemble prediction failed: {e}")
    
    def get_available_teams(self) -> List[str]:
        """Get list of all available teams."""
        return sorted(set(self.team_mappings.values()))
    
    def get_team_suggestions(self, partial_name: str) -> List[str]:
        """Get team name suggestions for partial input."""
        partial_norm = self._normalize_team_name(partial_name)
        suggestions = []
        
        for team in self.get_available_teams():
            team_norm = self._normalize_team_name(team)
            if partial_norm in team_norm:
                suggestions.append(team)
        
        return sorted(suggestions)[:10]  # Top 10 suggestions