"""
Temporal feature extraction module.

This module extracts features related to time, tournament stage, and seasonality.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any

from src.features.config import SEASON_PERIODS


class TemporalFeatureExtractor:
    """
    Extract time-related features from match data.

    Features include:
    - Day of week
    - Time of day
    - Season stage (early, mid, late season)
    - Days since last match for each team
    - Match significance based on tournament stage
    """

    def __init__(self, matches_df: pd.DataFrame):
        """Initialize temporal feature extractor.

        Args:
            matches_df: DataFrame with all matches (optional)
        """
        self.matches_df = matches_df

    def set_matches_df(self, matches_df: pd.DataFrame):
        """Set matches DataFrame.

        Args:
            matches_df: DataFrame with all matches
        """
        self.matches_df = matches_df

    def get_days_since_last_match(self, team: str, match_date: datetime) -> int:
        """Calculate days since team's last match.

        Args:
            team: Team name
            match_date: Date of the current match

        Returns:
            Number of days since last match (or 7 if no previous match found)
        """
        if self.matches_df is None:
            return 7  # Default if no match data available

        # Find all matches for the team before this match
        team_matches = self.matches_df[
            (
                (self.matches_df["home_team"] == team)
                | (self.matches_df["away_team"] == team)
            )
            & (self.matches_df["date"] < match_date)
        ]

        if team_matches.empty:
            return 7  # Default if no previous matches found

        # Get date of last match
        last_match_date = team_matches["date"].max()

        # Calculate days difference
        days_diff = (match_date - last_match_date).days

        return days_diff

    def get_season_progress(self, match_date: datetime, season: str) -> float:
        """Calculate normalized season progress (0.0 to 1.0).

        Args:
            match_date: Date of the match
            season: Season identifier

        Returns:
            Season progress (0.0 = start, 1.0 = end)
        """
        if self.matches_df is None:
            # Default to mid-season if no match data available
            return 0.5

        # Filter matches for this season
        season_matches = self.matches_df[self.matches_df["season"] == season]

        if season_matches.empty:
            # Default to mid-season if no season data available
            return 0.5

        # Get season boundaries
        season_start = season_matches["date"].min()
        season_end = season_matches["date"].max()

        # Calculate season length in days
        season_length = (season_end - season_start).days
        if season_length <= 0:
            return 0.5  # Fallback

        # Calculate days elapsed
        days_elapsed = (match_date - season_start).days

        # Normalize to 0.0-1.0
        progress = max(0.0, min(1.0, days_elapsed / season_length))

        return progress

    def get_season_stage(self, match_date: datetime, season: str) -> str:
        """Determine the stage of the season (early, mid, late).

        Args:
            match_date: Date of the match
            season: Season identifier

        Returns:
            Season stage as string
        """
        progress = self.get_season_progress(match_date, season)

        if progress < 0.33:
            return "early_season"
        elif progress < 0.66:
            return "mid_season"
        else:
            return "late_season"

    def is_important_match(
        self, match_date: datetime, season: str, stage: str
    ) -> float:
        """Determine match importance based on season progress and stage.

        Args:
            match_date: Date of the match
            season: Season identifier
            stage: Tournament stage

        Returns:
            Importance score (0.0 to 1.0)
        """
        progress = self.get_season_progress(match_date, season)

        # Higher importance for late season matches
        importance = progress * 0.5  # Base importance increases with season progress

        # Add importance for specific stages
        if "final" in stage.lower():
            importance += 0.5
        elif "semi" in stage.lower():
            importance += 0.4
        elif "quarter" in stage.lower():
            importance += 0.3
        elif "knockout" in stage.lower() or "playoff" in stage.lower():
            importance += 0.25

        # Cap at 1.0
        return min(1.0, importance)

    def extract_temporal_features(
        self,
        match_date: datetime,
        home_team: str,
        away_team: str,
        season: str,
        stage: str,
    ) -> Dict[str, Any]:
        """Extract all temporal features for a match.

        Args:
            match_date: Date of the match
            home_team: Home team name
            away_team: Away team name
            season: Season identifier
            stage: Tournament stage

        Returns:
            Dictionary with temporal features
        """
        # Convert to datetime if string
        if isinstance(match_date, str):
            match_date = pd.to_datetime(match_date)

        # Basic date features
        features = {
            "dayofweek": match_date.weekday(),  # 0=Monday, 6=Sunday
            "weekend": 1 if match_date.weekday() >= 5 else 0,
            "month": match_date.month,
            "day": match_date.day,
        }

        # One-hot encode day of week
        for i in range(7):
            features[f"day_{i}"] = 1 if match_date.weekday() == i else 0

        # One-hot encode month
        for i in range(1, 13):
            features[f"month_{i}"] = 1 if match_date.month == i else 0

        # Season period features
        for period, months in SEASON_PERIODS.items():
            features[period] = 1 if match_date.month in months else 0

        # Season progress
        progress = self.get_season_progress(match_date, season)
        features["season_progress"] = int(progress)
        features["season_progress_squared"] = int(progress**2)  # Non-linear term

        # Season stage
        stage_name = self.get_season_stage(match_date, season)
        for s in ["early_season", "mid_season", "late_season"]:
            features[s] = 1 if stage_name == s else 0

        # Days since last match
        features["home_days_rest"] = self.get_days_since_last_match(
            home_team, match_date
        )
        features["away_days_rest"] = self.get_days_since_last_match(
            away_team, match_date
        )
        features["rest_advantage"] = (
            features["home_days_rest"] - features["away_days_rest"]
        )

        # Match importance
        features["match_importance"] = int(self.is_important_match(
            match_date, season, stage
        ))

        return features
