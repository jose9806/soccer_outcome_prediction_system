"""
Unified predictor for soccer match predictions.

This module provides a unified interface for making predictions using various models.
It can handle match outcome predictions, expected goals, and various in-game events.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Union, Optional
import logging

from src.ai_models import (
    MatchOutcomeModel,
    ExpectedGoalsModel,
    EventPredictionModel,
    BTTSModel,
)

try:
    from src.ai_models import DeepMatchOutcomeModel, DeepExpectedGoalsModel

    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False


class UnifiedPredictor:
    """Unified interface for making predictions using various models."""

    def __init__(self, models_dir: str = "models", deep_learning: bool = False):
        """
        Initialize the unified predictor.

        Args:
            models_dir: Directory containing trained models
            deep_learning: Whether to use deep learning models when available
        """
        self.models_dir = models_dir
        self.use_deep_learning = deep_learning and DEEP_LEARNING_AVAILABLE
        self.models = {}
        self.logger = logging.getLogger("UnifiedPredictor")
        self.logger.setLevel(logging.INFO)

        # Configure logging
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info(f"Initialized unified predictor with models from {models_dir}")
        self.logger.info(
            f"Deep learning: {'Available' if DEEP_LEARNING_AVAILABLE else 'Not available'}, "
            f"Using: {self.use_deep_learning}"
        )

    def load_models(self, model_types: Optional[List[str]] = None) -> None:
        """
        Load all required models.

        Args:
            model_types: List of model types to load (None for all available models)
        """
        if model_types is None:
            model_types = [
                "match_outcome",
                "total_goals",
                "home_goals",
                "away_goals",
                "yellow_cards",
                "corner_kicks",
                "fouls",
                "both_teams_scored",
            ]

        self.logger.info(f"Loading {len(model_types)} model types")

        # Load match outcome model
        if "match_outcome" in model_types:
            if self.use_deep_learning:
                try:
                    model = DeepMatchOutcomeModel()
                    model.load_model(self.models_dir)
                    self.models["match_outcome"] = model
                    self.logger.info("Loaded deep learning match outcome model")
                except Exception as e:
                    self.logger.warning(
                        f"Failed to load deep learning match outcome model: {e}"
                    )
                    self._load_ml_model("match_outcome", MatchOutcomeModel)
            else:
                self._load_ml_model("match_outcome", MatchOutcomeModel)

        # Load expected goals models
        for goal_type in ["total_goals", "home_goals", "away_goals"]:
            if goal_type in model_types:
                if self.use_deep_learning:
                    try:
                        model = DeepExpectedGoalsModel(target_type=goal_type)
                        model.load_model(self.models_dir)
                        self.models[goal_type] = model
                        self.logger.info(f"Loaded deep learning {goal_type} model")
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to load deep learning {goal_type} model: {e}"
                        )
                        self._load_ml_model(
                            goal_type, ExpectedGoalsModel, target_type=goal_type
                        )
                else:
                    self._load_ml_model(
                        goal_type, ExpectedGoalsModel, target_type=goal_type
                    )

        # Load event prediction models
        for event_type in ["yellow_cards", "corner_kicks", "fouls"]:
            if event_type in model_types:
                # Load count prediction model
                self._load_ml_model(
                    f"{event_type}_count",
                    EventPredictionModel,
                    event_type=event_type,
                    prediction_type="count",
                )

                # Load over/under models for various thresholds
                if event_type == "yellow_cards":
                    thresholds = [1.5, 2.5, 3.5, 4.5, 5.5]
                elif event_type == "corner_kicks":
                    thresholds = [7.5, 8.5, 9.5, 10.5, 11.5]
                elif event_type == "fouls":
                    thresholds = [15.5, 20.5, 25.5]
                else:
                    thresholds = []

                for threshold in thresholds:
                    threshold_str = str(threshold).replace(".", "_")
                    model_name = f"{event_type}_over_{threshold_str}"
                    self._load_ml_model(
                        model_name,
                        EventPredictionModel,
                        event_type=event_type,
                        prediction_type="over_under",
                        threshold=threshold,
                    )

        # Load BTTS model
        if "both_teams_scored" in model_types:
            self._load_ml_model("both_teams_scored", BTTSModel)

        self.logger.info(f"Successfully loaded {len(self.models)} models")

    def _load_ml_model(self, model_name: str, model_class, **kwargs) -> None:
        """
        Load a specific ML model.

        Args:
            model_name: Name of the model
            model_class: Model class to instantiate
            **kwargs: Additional arguments for the model class
        """
        try:
            model = model_class(**kwargs)
            model.load_model(self.models_dir)
            self.models[model_name] = model
            self.logger.info(f"Loaded {model_name} model")
        except Exception as e:
            self.logger.warning(f"Failed to load {model_name} model: {e}")

    def predict_match(
        self, home_team: str, away_team: str, match_date: str
    ) -> Dict[str, Any]:
        """
        Make comprehensive predictions for a match.

        Args:
            home_team: Home team name
            away_team: Away team name
            match_date: Match date (string in format YYYY-MM-DD)

        Returns:
            Dictionary with all predictions
        """
        # Convert date string to datetime
        try:
            match_date_dt = datetime.strptime(match_date, "%Y-%m-%d")
        except ValueError:
            match_date_dt = datetime.now()  # Use current date as fallback
            self.logger.warning(
                f"Invalid date format: {match_date}. Using current date."
            )

        # Initialize prediction results
        predictions = {
            "match_info": {
                "home_team": home_team,
                "away_team": away_team,
                "match_date": match_date,
            },
            "predictions": {},
        }

        # Make match outcome prediction
        if "match_outcome" in self.models:
            try:
                outcome_probs = self.models["match_outcome"].predict_proba(
                    home_team, away_team, match_date_dt
                )
                outcome = self.models["match_outcome"].predict(
                    home_team, away_team, match_date_dt
                )

                predictions["predictions"]["match_outcome"] = {
                    "prediction": outcome,
                    "probabilities": outcome_probs,
                }
            except Exception as e:
                self.logger.error(f"Error predicting match outcome: {e}")

        # Make expected goals predictions
        for goal_type in ["total_goals", "home_goals", "away_goals"]:
            if goal_type in self.models:
                try:
                    goals = self.models[goal_type].predict(
                        home_team, away_team, match_date_dt
                    )

                    if "expected_goals" not in predictions["predictions"]:
                        predictions["predictions"]["expected_goals"] = {}

                    predictions["predictions"]["expected_goals"][goal_type] = goals

                    # Add implied scoreline after home and away goals are predicted
                    if all(
                        k in predictions["predictions"]["expected_goals"]
                        for k in ["home_goals", "away_goals"]
                    ):
                        home_goals = predictions["predictions"]["expected_goals"][
                            "home_goals"
                        ]
                        away_goals = predictions["predictions"]["expected_goals"][
                            "away_goals"
                        ]

                        winner = (
                            "home_win"
                            if home_goals > away_goals
                            else ("away_win" if away_goals > home_goals else "draw")
                        )

                        predictions["predictions"]["expected_goals"][
                            "implied_outcome"
                        ] = winner
                        predictions["predictions"]["expected_goals"][
                            "implied_scoreline"
                        ] = f"{home_goals}-{away_goals}"

                except Exception as e:
                    self.logger.error(f"Error predicting {goal_type}: {e}")

        # Make event predictions
        for event_type in ["yellow_cards", "corner_kicks", "fouls"]:
            count_model_name = f"{event_type}_count"

            if count_model_name in self.models:
                try:
                    count = self.models[count_model_name].predict(
                        home_team, away_team, match_date_dt
                    )

                    if "events" not in predictions["predictions"]:
                        predictions["predictions"]["events"] = {}

                    if event_type not in predictions["predictions"]["events"]:
                        predictions["predictions"]["events"][event_type] = {}

                    predictions["predictions"]["events"][event_type]["count"] = count

                    # Add over/under predictions for various thresholds
                    thresholds = []
                    if event_type == "yellow_cards":
                        thresholds = [1.5, 2.5, 3.5, 4.5, 5.5]
                    elif event_type == "corner_kicks":
                        thresholds = [7.5, 8.5, 9.5, 10.5, 11.5]
                    elif event_type == "fouls":
                        thresholds = [15.5, 20.5, 25.5]

                    for threshold in thresholds:
                        threshold_str = str(threshold).replace(".", "_")
                        model_name = f"{event_type}_over_{threshold_str}"

                        if model_name in self.models:
                            try:
                                is_over = self.models[model_name].predict(
                                    home_team, away_team, match_date_dt
                                )

                                over_probs = self.models[model_name].predict_proba(
                                    home_team, away_team, match_date_dt
                                )

                                if (
                                    "thresholds"
                                    not in predictions["predictions"]["events"][
                                        event_type
                                    ]
                                ):
                                    predictions["predictions"]["events"][event_type][
                                        "thresholds"
                                    ] = {}

                                predictions["predictions"]["events"][event_type][
                                    "thresholds"
                                ][str(threshold)] = {
                                    "is_over": is_over,
                                    "probabilities": over_probs,
                                }
                            except Exception as e:
                                self.logger.error(f"Error predicting {model_name}: {e}")

                except Exception as e:
                    self.logger.error(f"Error predicting {count_model_name}: {e}")

        # Make BTTS prediction
        if "both_teams_scored" in self.models:
            try:
                btts = self.models["both_teams_scored"].predict(
                    home_team, away_team, match_date_dt
                )

                btts_probs = self.models["both_teams_scored"].predict_proba(
                    home_team, away_team, match_date_dt
                )

                if "events" not in predictions["predictions"]:
                    predictions["predictions"]["events"] = {}

                predictions["predictions"]["events"]["both_teams_to_score"] = {
                    "prediction": btts,
                    "probabilities": btts_probs,
                }
            except Exception as e:
                self.logger.error(f"Error predicting BTTS: {e}")

        return predictions

    def save_prediction(
        self, prediction: Dict[str, Any], output_dir: str = "predictions"
    ) -> str:
        """
        Save a prediction to disk.

        Args:
            prediction: Prediction dictionary
            output_dir: Directory to save the prediction

        Returns:
            Path to the saved prediction file
        """
        os.makedirs(output_dir, exist_ok=True)

        # Create filename from match details
        home_team = prediction["match_info"]["home_team"].replace(" ", "_")
        away_team = prediction["match_info"]["away_team"].replace(" ", "_")
        match_date = prediction["match_info"]["match_date"].replace("-", "")

        filename = f"{match_date}_{home_team}_vs_{away_team}.json"
        filepath = os.path.join(output_dir, filename)

        # Add timestamp to prediction
        prediction["timestamp"] = datetime.now().isoformat()

        # Save prediction
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(prediction, f, indent=2)

        self.logger.info(f"Saved prediction to {filepath}")
        return filepath
