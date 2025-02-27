"""
Pipeline for extracting, combining, and selecting features for match prediction.

This module orchestrates the entire feature engineering process:
1. Loading processed match data
2. Extracting various feature sets
3. Combining features into a unified dataset
4. Selecting the most predictive features
5. Saving the engineered features
"""

import logging
import json
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Set, Tuple
from pathlib import Path
import datetime

from src.feature_engineering.extractors.team_performance_features import (
    TeamPerformanceFeatures,
)
from src.feature_engineering.extractors.match_context_features import (
    MatchContextFeatures,
)
from src.feature_engineering.extractors.advanced_metrics import AdvancedMetrics
from src.feature_engineering.extractors.temporal_features import TemporalFeatures
from src.feature_engineering.selectors.feature_selector import FeatureSelector

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Pipeline for feature engineering of soccer match data."""

    def __init__(
        self,
        processed_data_dir: str = "data/processed",
        features_data_dir: str = "data/features",
        derby_file: Optional[str] = "data/features/derbies.json",
    ):
        """
        Initialize the feature engineering pipeline.

        Args:
            processed_data_dir: Directory containing processed match data
            features_data_dir: Directory to save engineered features
            derby_file: Path to a JSON file containing derby pairs
        """
        self.processed_data_dir = Path(processed_data_dir)
        self.features_data_dir = Path(features_data_dir)
        self.derby_file = derby_file

        # Create feature extractors
        self.team_performance = TeamPerformanceFeatures()

        # Load derby pairs if file exists
        derby_pairs = []
        if derby_file and os.path.exists(derby_file):
            try:
                with open(derby_file, "r", encoding="utf-8") as f:
                    derby_pairs = json.load(f)
                    logger.info(
                        f"Loaded {len(derby_pairs)} derby pairs from {derby_file}"
                    )
            except Exception as e:
                logger.error(f"Error loading derby pairs from {derby_file}: {e}")

        self.match_context = MatchContextFeatures(derby_pairs=derby_pairs)
        self.advanced_metrics = AdvancedMetrics()
        self.temporal_features = TemporalFeatures()

        # Create feature selector
        self.feature_selector = FeatureSelector()

        logger.info("Initialized FeaturePipeline")

    def load_processed_data(
        self, seasons: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Load processed match data from specified seasons.

        Args:
            seasons: List of seasons to load, or None to load all

        Returns:
            List of match data dictionaries
        """
        if not self.processed_data_dir.exists():
            logger.error(
                f"Processed data directory does not exist: {self.processed_data_dir}"
            )
            return []

        # Get available seasons
        available_seasons = [
            d.name for d in self.processed_data_dir.iterdir() if d.is_dir()
        ]

        if not available_seasons:
            logger.warning(f"No seasons found in {self.processed_data_dir}")
            return []

        # Determine which seasons to load
        if seasons:
            seasons_to_load = [s for s in seasons if s in available_seasons]
            if not seasons_to_load:
                logger.warning(
                    f"None of the specified seasons {seasons} found in {self.processed_data_dir}"
                )
                return []
        else:
            seasons_to_load = available_seasons

        logger.info(
            f"Loading processed data for {len(seasons_to_load)} seasons: {seasons_to_load}"
        )

        # Load match data from each season
        all_matches = []

        for season in seasons_to_load:
            season_dir = self.processed_data_dir / season
            match_files = list(season_dir.glob("*.json"))

            logger.info(f"Found {len(match_files)} match files for season {season}")

            for file_path in match_files:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        match_data = json.load(f)
                        all_matches.append(match_data)
                except Exception as e:
                    logger.error(f"Error loading match file {file_path}: {e}")
                    continue

        logger.info(
            f"Loaded {len(all_matches)} matches from {len(seasons_to_load)} seasons"
        )
        return all_matches

    def detect_derbies(
        self, matches: List[Dict[str, Any]], save: bool = True
    ) -> List[Tuple[str, str]]:
        """
        Detect derby matches from historical data.

        Args:
            matches: List of match data dictionaries
            save: Whether to save detected derbies to file

        Returns:
            List of detected derby pairs
        """
        logger.info("Detecting derby matches from historical data")

        # Use match context feature extractor to detect derbies
        derby_pairs = self.match_context.detect_derbies(matches)

        # Save derby pairs if requested
        if save and self.derby_file:
            derby_dir = os.path.dirname(self.derby_file)
            os.makedirs(derby_dir, exist_ok=True)

            with open(self.derby_file, "w", encoding="utf-8") as f:
                json.dump(derby_pairs, f, indent=2)

            logger.info(f"Saved {len(derby_pairs)} derby pairs to {self.derby_file}")

        return derby_pairs

    def extract_features(
        self, matches: List[Dict[str, Any]]
    ) -> Dict[str, pd.DataFrame]:
        """
        Extract all feature sets from match data.

        Args:
            matches: List of match data dictionaries

        Returns:
            Dictionary mapping feature set names to DataFrames
        """
        if not matches:
            logger.warning("No matches provided for feature extraction")
            return {}

        logger.info(f"Extracting features from {len(matches)} matches")

        # Extract team performance features
        team_performance_df = self.team_performance.extract_features(matches)
        logger.info(
            f"Extracted team performance features: {len(team_performance_df)} rows"
        )

        # Create match features from team performance
        match_features_df = self.team_performance.create_match_features(
            team_performance_df, matches
        )
        logger.info(
            f"Created match features from team performance: {len(match_features_df)} matches"
        )

        # Extract match context features
        match_context_df = self.match_context.extract_features(matches)
        logger.info(
            f"Extracted match context features: {len(match_context_df)} matches"
        )

        # Extract advanced metrics
        advanced_metrics_df = self.advanced_metrics.extract_features(matches)
        logger.info(f"Extracted advanced metrics: {len(advanced_metrics_df)} matches")

        # Extract temporal features
        temporal_features_df = self.temporal_features.extract_features(matches)
        logger.info(f"Extracted temporal features: {len(temporal_features_df)} matches")

        # Calculate team strength metrics
        team_strength_df = self.advanced_metrics.get_team_strength_metrics(matches)
        logger.info(f"Calculated team strength metrics: {len(team_strength_df)} teams")

        # Return all feature sets
        feature_sets = {
            "team_performance": team_performance_df,
            "match_features": match_features_df,
            "match_context": match_context_df,
            "advanced_metrics": advanced_metrics_df,
            "temporal_features": temporal_features_df,
            "team_strength": team_strength_df,
        }

        return feature_sets

    def combine_features(self, feature_sets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine feature sets into a unified dataset.

        Args:
            feature_sets: Dictionary mapping feature set names to DataFrames

        Returns:
            DataFrame with combined features
        """
        if not feature_sets:
            logger.warning("No feature sets provided for combination")
            return pd.DataFrame()

        logger.info(f"Combining {len(feature_sets)} feature sets")

        # Start with match features as the base
        if (
            "match_features" in feature_sets
            and not feature_sets["match_features"].empty
        ):
            combined_df = feature_sets["match_features"].copy()
        else:
            logger.warning("Match features not available for combination")
            return pd.DataFrame()

        # Ensure the index is unique
        if combined_df.index.duplicated().any():
            logger.warning(
                "Match features DataFrame has duplicate indices. Resetting index."
            )
            combined_df = combined_df.reset_index(drop=True)

        # Combine with match context features
        if "match_context" in feature_sets and not feature_sets["match_context"].empty:
            match_context_df = feature_sets["match_context"]

            # Merge on match_id
            combined_df = pd.merge(
                combined_df,
                match_context_df,
                on="match_id",
                how="left",
                suffixes=("", "_context"),
            )

            # Remove duplicate columns
            duplicate_cols = [
                col for col in combined_df.columns if col.endswith("_context")
            ]
            combined_df = combined_df.drop(columns=duplicate_cols)

        # Combine with advanced metrics
        if (
            "advanced_metrics" in feature_sets
            and not feature_sets["advanced_metrics"].empty
        ):
            advanced_metrics_df = feature_sets["advanced_metrics"]

            # Merge on match_id
            combined_df = pd.merge(
                combined_df,
                advanced_metrics_df,
                on="match_id",
                how="left",
                suffixes=("", "_metrics"),
            )

            # Remove duplicate columns
            duplicate_cols = [
                col for col in combined_df.columns if col.endswith("_metrics")
            ]
            combined_df = combined_df.drop(columns=duplicate_cols)

        # Combine with temporal features
        if (
            "temporal_features" in feature_sets
            and not feature_sets["temporal_features"].empty
        ):
            temporal_features_df = feature_sets["temporal_features"]

            # Merge on match_id
            combined_df = pd.merge(
                combined_df,
                temporal_features_df,
                on="match_id",
                how="left",
                suffixes=("", "_temporal"),
            )

            # Remove duplicate columns
            duplicate_cols = [
                col for col in combined_df.columns if col.endswith("_temporal")
            ]
            combined_df = combined_df.drop(columns=duplicate_cols)

        # Add team strength metrics
        if "team_strength" in feature_sets and not feature_sets["team_strength"].empty:
            team_strength_df = feature_sets["team_strength"]

            # Merge home team strength
            combined_df = pd.merge(
                combined_df,
                team_strength_df,
                left_on="home_team",
                right_on="team",
                how="left",
                suffixes=("", "_home_strength"),
            )

            # Rename columns to indicate home team
            home_strength_cols = [
                col for col in combined_df.columns if col.endswith("_home_strength")
            ]
            for col in home_strength_cols:
                new_col = f"home_{col.replace('_home_strength', '')}"
                combined_df.rename(columns={col: new_col}, inplace=True)

            # Drop team column from merge
            if "team" in combined_df.columns:
                combined_df = combined_df.drop(columns=["team"])

            # Merge away team strength
            combined_df = pd.merge(
                combined_df,
                team_strength_df,
                left_on="away_team",
                right_on="team",
                how="left",
                suffixes=("", "_away_strength"),
            )

            # Rename columns to indicate away team
            away_strength_cols = [
                col for col in combined_df.columns if col.endswith("_away_strength")
            ]
            for col in away_strength_cols:
                new_col = f"away_{col.replace('_away_strength', '')}"
                combined_df.rename(columns={col: new_col}, inplace=True)

            # Drop team column from merge
            if "team" in combined_df.columns:
                combined_df = combined_df.drop(columns=["team"])

        # Check for duplicate indices again after all merges
        if combined_df.index.duplicated().any():
            logger.warning(
                "Combined DataFrame has duplicate indices after merges. Resetting index."
            )
            combined_df = combined_df.reset_index(drop=True)

        # Calculate team strength differentials more efficiently
        numeric_cols = combined_df.select_dtypes(include=[np.number]).columns

        # Prepare a dictionary to hold all differential features
        diff_features = {}

        # Find matching home/away columns
        home_cols = [
            col
            for col in numeric_cols
            if col.startswith("home_") and not col.startswith("home_team")
        ]
        away_cols = [
            col
            for col in numeric_cols
            if col.startswith("away_") and not col.startswith("away_team")
        ]

        # Create all differential features at once
        for home_col in home_cols:
            away_col = home_col.replace("home_", "away_")

            if away_col in away_cols and away_col in combined_df.columns:
                # Create differential feature name
                diff_col = home_col.replace("home_", "diff_")

                try:
                    # Verify both columns have compatible shapes and are numeric
                    home_values = combined_df[home_col].values
                    away_values = combined_df[away_col].values

                    # Check if they are 1D arrays with the same length
                    if (
                        home_values.ndim == 1
                        and away_values.ndim == 1
                        and len(home_values) == len(away_values)
                    ):
                        # Calculate the differential
                        diff_features[diff_col] = home_values - away_values
                    else:
                        logger.warning(
                            f"Skipping differential calculation for {diff_col}: Incompatible shapes ({home_values.shape} vs {away_values.shape})"
                        )
                except Exception as e:
                    logger.warning(
                        f"Error calculating differential for {diff_col}: {e}"
                    )

        # Add all differential features at once to avoid fragmentation
        if diff_features:
            # Create a new DataFrame with the differential features
            diff_df = pd.DataFrame(diff_features, index=combined_df.index)

            # Concatenate horizontally with the combined DataFrame
            combined_df = pd.concat([combined_df, diff_df], axis=1)
        logger.info(
            f"Combined features: {len(combined_df)} rows with {len(combined_df.columns)} columns"
        )
        return combined_df

    def select_features(
        self,
        combined_df: pd.DataFrame,
        target: str = "result",
        max_features: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select the most predictive features for the target variable.

        Args:
            combined_df: DataFrame with combined features
            target: Target variable to predict
            max_features: Maximum number of features to select

        Returns:
            Tuple of (DataFrame with selected features, List of selected feature names)
        """
        if combined_df.empty:
            logger.warning("Empty DataFrame provided for feature selection")
            return pd.DataFrame(), []

        logger.info(f"Selecting features for target '{target}'")

        # Use feature selector to select best features
        selected_df, selected_features = self.feature_selector.select_features(
            combined_df, target, max_features
        )

        logger.info(f"Selected {len(selected_features)} features")
        return selected_df, selected_features

    def save_features(
        self, feature_sets: Dict[str, pd.DataFrame], selected_df: pd.DataFrame
    ) -> None:
        """
        Save engineered features to files.

        Args:
            feature_sets: Dictionary mapping feature set names to DataFrames
            selected_df: DataFrame with selected features
        """
        if not feature_sets:
            logger.warning("No feature sets provided for saving")
            return

        # Create features directory
        os.makedirs(self.features_data_dir, exist_ok=True)

        # Save each feature set
        for name, df in feature_sets.items():
            if df.empty:
                logger.warning(f"Empty feature set '{name}', skipping save")
                continue

            file_path = self.features_data_dir / f"{name}.csv"

            try:
                df.to_csv(file_path, index=False)
                logger.info(f"Saved {len(df)} rows of '{name}' features to {file_path}")
            except Exception as e:
                logger.error(f"Error saving feature set '{name}' to {file_path}: {e}")

        # Save selected features if provided
        if selected_df is not None and not selected_df.empty:
            file_path = self.features_data_dir / "selected_features.csv"

            try:
                selected_df.to_csv(file_path, index=False)
                logger.info(
                    f"Saved {len(selected_df)} rows of selected features to {file_path}"
                )
            except Exception as e:
                logger.error(f"Error saving selected features to {file_path}: {e}")

    def run(
        self,
        seasons: Optional[List[str]] = None,
        target: str = "result",
        max_features: Optional[int] = None,
        save: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the feature engineering pipeline.

        Args:
            seasons: List of seasons to process, or None to process all
            target: Target variable to predict
            max_features: Maximum number of features to select
            save: Whether to save the engineered features

        Returns:
            Dictionary with pipeline results
        """
        start_time = datetime.datetime.now()

        logger.info(f"Running feature engineering pipeline for target '{target}'")

        # Load processed match data
        matches = self.load_processed_data(seasons)

        if not matches:
            logger.error("No match data loaded, aborting pipeline")
            return {"success": False, "error": "No match data loaded"}

        # Extract feature sets
        feature_sets = self.extract_features(matches)

        if not feature_sets:
            logger.error("Feature extraction failed, aborting pipeline")
            return {"success": False, "error": "Feature extraction failed"}

        # Combine feature sets
        combined_df = self.combine_features(feature_sets)

        if combined_df.empty:
            logger.error("Feature combination failed, aborting pipeline")
            return {"success": False, "error": "Feature combination failed"}

        # Select features
        selected_df, selected_features = self.select_features(
            combined_df, target, max_features
        )

        # Save features if requested
        if save:
            self.save_features(feature_sets, selected_df)

            # Save selected feature names
            if selected_features:
                feature_file = self.features_data_dir / "selected_feature_names.json"

                try:
                    with open(feature_file, "w", encoding="utf-8") as f:
                        json.dump(selected_features, f, indent=2)

                    logger.info(
                        f"Saved {len(selected_features)} selected feature names to {feature_file}"
                    )
                except Exception as e:
                    logger.error(
                        f"Error saving selected feature names to {feature_file}: {e}"
                    )

        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Return pipeline results
        results = {
            "success": True,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "matches_processed": len(matches),
            "feature_sets": {name: len(df) for name, df in feature_sets.items()},
            "combined_features": len(combined_df.columns),
            "selected_features": len(selected_features),
            "selected_feature_names": selected_features,
        }

        logger.info(f"Feature engineering pipeline completed in {duration:.2f} seconds")
        return results
