"""
Base model for soccer match prediction.

This module provides a base class for all prediction models with common functionality
for loading data, feature extraction, and model evaluation.
"""

import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
import logging

from src.features.feature_engineering import FeatureEngineer


class BaseModel:
    """Base class for soccer prediction models."""

    def __init__(self, model_name: str, data_dir: str = "data/datasets"):
        """
        Initialize the base model.

        Args:
            model_name: Name of the model
            data_dir: Directory containing dataset files
        """
        self.model_name = model_name
        self.data_dir = data_dir
        self.model = None
        self.feature_engineer = None
        self.feature_columns = []
        self.categorical_columns = []
        self.numerical_columns = []
        self.target_column = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.match_data_cache = None

        # Configure logging
        self.logger = logging.getLogger(f"{model_name}_model")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def load_data(self, dataset_path: str) -> pd.DataFrame:
        """
        Load dataset from a CSV file.

        Args:
            dataset_path: Path to the dataset file

        Returns:
            Loaded DataFrame
        """
        self.logger.info(f"Loading dataset from {dataset_path}")
        df = pd.read_csv(dataset_path)

        # Convert date column to datetime if present
        if "match_date" in df.columns:
            df["match_date"] = pd.to_datetime(df["match_date"])

        return df

    def load_data_split(
        self, dataset_name: str, split_type: str = None
    ) -> pd.DataFrame:
        """
        Load a specific dataset split (train, val, test).

        Args:
            dataset_name: Name of the dataset (e.g., 'match_outcome', 'yellow_cards')
            split_type: Type of split to load ('train', 'val', 'test', or None for full dataset)

        Returns:
            Loaded DataFrame
        """
        if split_type is not None:
            path = os.path.join(
                self.data_dir, dataset_name, "splits", f"{split_type}.csv"
            )
        else:
            # Try to find the main dataset file
            if dataset_name == "match_outcome":
                path = os.path.join(
                    self.data_dir, dataset_name, "match_outcome_3way.csv"
                )
            elif dataset_name.startswith("goals_over_") or dataset_name.startswith(
                "goals_under_"
            ):
                path = os.path.join(self.data_dir, "events", f"{dataset_name}.csv")
            elif dataset_name in ["home_goals", "away_goals", "total_goals"]:
                path = os.path.join(
                    self.data_dir, "expected_goals", f"{dataset_name}.csv"
                )
            else:
                path = os.path.join(self.data_dir, "events", f"{dataset_name}.csv")

        if not os.path.exists(path):
            self.logger.error(f"Dataset file not found: {path}")
            return pd.DataFrame()

        return self.load_data(path)

    def load_match_data(self, data_path: str = "data/raw") -> pd.DataFrame:
        """
        Load raw match data for feature engineering.

        Args:
            data_path: Path to raw match data

        Returns:
            DataFrame with match data
        """
        if self.match_data_cache is not None:
            return self.match_data_cache

        import json
        from tqdm import tqdm

        self.logger.info(f"Loading raw match data from {data_path}")

        all_matches = []

        # Walk through all files in the data path
        for root, _, files in os.walk(data_path):
            # Filter JSON files
            json_files = [f for f in files if f.endswith(".json")]

            if not json_files:
                continue

            # Process each JSON file
            for file in tqdm(
                json_files, desc=f"Loading files from {os.path.basename(root)}"
            ):
                try:
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        match_data = json.load(f)

                        # Check if match has the required data
                        if not all(
                            k in match_data for k in ["home_team", "away_team", "date"]
                        ):
                            continue

                        all_matches.append(match_data)
                except Exception as e:
                    self.logger.warning(f"Error processing {file}: {str(e)}")

        # Convert to DataFrame
        matches_df = pd.DataFrame(all_matches)

        # Convert dates to datetime
        if "date" in matches_df.columns:
            matches_df["date"] = pd.to_datetime(matches_df["date"])

        # Sort by date
        if "date" in matches_df.columns:
            matches_df = matches_df.sort_values("date")

        self.match_data_cache = matches_df
        self.logger.info(f"Loaded {len(matches_df)} matches")

        return matches_df

    def initialize_feature_engineer(self, matches_df: pd.DataFrame = None) -> None:
        """
        Initialize the feature engineer with match data.

        Args:
            matches_df: DataFrame with match data (if None, will load from default path)
        """
        if matches_df is None:
            matches_df = self.load_match_data()

        self.feature_engineer = FeatureEngineer(matches_df)
        self.logger.info("Initialized feature engineer")

    def prepare_prediction_features(
        self,
        home_team: str,
        away_team: str,
        match_date: datetime,
        season: str = None,
        stage: str = "Regular Season",
    ) -> Dict[str, Any]:
        """
        Extract features for making predictions on new matches.

        Args:
            home_team: Home team name
            away_team: Away team name
            match_date: Date of the match
            season: Season identifier (optional, will be derived from date if None)
            stage: Tournament stage

        Returns:
            Dictionary with extracted features
        """
        # Initialize feature engineer if not already done
        if self.feature_engineer is None:
            self.initialize_feature_engineer()

        # Determine season from date if not provided
        if season is None:
            season = str(match_date.year)

        # Extract features using the feature engineer
        if self.model_name in [
            "match_outcome",
            "home_goals",
            "away_goals",
            "total_goals",
        ]:
            features = self.feature_engineer.extract_match_features(
                home_team=home_team,
                away_team=away_team,
                match_date=match_date,
                season=season,
                stage=stage,
            )
        else:
            # For event prediction models
            features = self.feature_engineer.extract_match_events_features(
                home_team=home_team,
                away_team=away_team,
                match_date=match_date,
                season=season,
                stage=stage,
            )

        return features

    def preprocess_features(self, features: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess features for model input.

        Args:
            features: Dictionary with extracted features

        Returns:
            DataFrame with preprocessed features
        """
        # Convert features dict to DataFrame
        df = pd.DataFrame([features])

        # Convert date to string to avoid serialization issues
        if "match_date" in df.columns:
            df["match_date"] = df["match_date"].astype(str)

        # Select only the feature columns used by the model
        if self.feature_columns:
            # Keep only the columns that exist in the DataFrame
            valid_columns = [col for col in self.feature_columns if col in df.columns]
            df = df[valid_columns]

        return df

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model on validation or test data.

        Args:
            X: Feature DataFrame
            y: Target Series

        Returns:
            Dictionary with evaluation metrics
        """
        # This is a base method to be implemented by subclasses
        raise NotImplementedError("Subclasses must implement evaluate() method")

    def save_model(self, output_dir: str = "models") -> str:
        """
        Save the trained model to disk.

        Args:
            output_dir: Directory to save the model

        Returns:
            Path to the saved model
        """
        os.makedirs(output_dir, exist_ok=True)

        model_path = os.path.join(output_dir, f"{self.model_name}_model.pkl")

        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        # Save feature columns for future reference
        feature_columns_path = os.path.join(
            output_dir, f"{self.model_name}_features.pkl"
        )
        with open(feature_columns_path, "wb") as f:
            pickle.dump(
                {
                    "feature_columns": self.feature_columns,
                    "categorical_columns": self.categorical_columns,
                    "numerical_columns": self.numerical_columns,
                    "target_column": self.target_column,
                },
                f,
            )

        self.logger.info(f"Model saved to {model_path}")
        return model_path

    def load_model(self, input_dir: str = "models") -> None:
        """
        Load a trained model from disk.

        Args:
            input_dir: Directory containing the saved model
        """
        model_path = os.path.join(input_dir, f"{self.model_name}_model.pkl")
        feature_columns_path = os.path.join(
            input_dir, f"{self.model_name}_features.pkl"
        )

        if not os.path.exists(model_path):
            self.logger.error(f"Model file not found: {model_path}")
            return

        if not os.path.exists(feature_columns_path):
            self.logger.warning(
                f"Feature columns file not found: {feature_columns_path}"
            )

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        # Load feature columns if available
        if os.path.exists(feature_columns_path):
            with open(feature_columns_path, "rb") as f:
                feature_data = pickle.load(f)
                self.feature_columns = feature_data.get("feature_columns", [])
                self.categorical_columns = feature_data.get("categorical_columns", [])
                self.numerical_columns = feature_data.get("numerical_columns", [])
                self.target_column = feature_data.get("target_column")

        self.logger.info(f"Model loaded from {model_path}")

    def predict_proba(
        self, home_team: str, away_team: str, match_date: Union[str, datetime]
    ) -> Dict[str, float]:
        """
        Make a probabilistic prediction for a new match.

        Args:
            home_team: Home team name
            away_team: Away team name
            match_date: Date of the match (string or datetime)

        Returns:
            Dictionary with prediction probabilities
        """
        # Convert date string to datetime if needed
        if isinstance(match_date, str):
            match_date = pd.to_datetime(match_date)

        # Extract features
        features = self.prepare_prediction_features(home_team, away_team, match_date)

        # Preprocess features
        X = self.preprocess_features(features)

        # This is a base method to be implemented by subclasses
        raise NotImplementedError("Subclasses must implement predict_proba() method")

    def predict(
        self, home_team: str, away_team: str, match_date: Union[str, datetime]
    ) -> Any:
        """
        Make a prediction for a new match.

        Args:
            home_team: Home team name
            away_team: Away team name
            match_date: Date of the match (string or datetime)

        Returns:
            Prediction result (type depends on the specific model)
        """
        # Convert date string to datetime if needed
        if isinstance(match_date, str):
            match_date = pd.to_datetime(match_date)

        # Extract features
        features = self.prepare_prediction_features(home_team, away_team, match_date)

        # Preprocess features
        X = self.preprocess_features(features)

        # This is a base method to be implemented by subclasses
        raise NotImplementedError("Subclasses must implement predict() method")
