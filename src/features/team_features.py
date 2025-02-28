"""
Team feature extraction module.

This module extracts team-specific features from historical match data.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

from src.features.config import (
    TIME_WINDOWS,
    MATCH_STATS,
    ADVANCED_STATS,
    MIN_MATCHES,
)


class TeamFeatureExtractor:
    """
    Extract team-specific features from historical match data.

    Features include:
    - Recent form (wins/draws/losses in last N matches)
    - Average statistics over different time windows
    - Home vs away performance differences
    - Team strength indicators
    """

    def __init__(self, data_path: str = "data/processed"):
        """Initialize team feature extractor.

        Args:
            data_path: Path to processed match data
        """
        self.data_path = data_path
        self.team_matches = {}  # Cache for team matches
        self.matches_df = pd.DataFrame()
        self.load_all_matches()

    def load_all_matches(self):
        """Load all match data into a DataFrame."""
        all_matches = []

        for root, _, files in os.walk(self.data_path):
            for file in files:
                if file.endswith(".json"):
                    with open(os.path.join(root, file), "r") as f:
                        try:
                            match_data = json.load(f)
                            all_matches.append(match_data)
                        except json.JSONDecodeError:
                            print(f"Error decoding {os.path.join(root, file)}")

        # Convert to DataFrame
        self.matches_df = pd.DataFrame(all_matches)

        # Convert dates to datetime
        self.matches_df["date"] = pd.to_datetime(self.matches_df["date"])

        # Sort by date
        self.matches_df.sort_values("date", inplace=True)

    def get_team_matches(self, team_name: str) -> pd.DataFrame:
        """Get all matches for a specific team.

        Args:
            team_name: Name of the team

        Returns:
            DataFrame with all matches for the team
        """
        if team_name in self.team_matches:
            return self.team_matches[team_name]

        # Find all matches where team played
        team_matches = self.matches_df[
            (self.matches_df["home_team"] == team_name)
            | (self.matches_df["away_team"] == team_name)
        ].copy()

        # Add columns for easier analysis
        team_matches["is_home"] = team_matches["home_team"] == team_name

        team_matches["opponent"] = np.where(
            team_matches["is_home"],
            team_matches["away_team"],
            team_matches["home_team"],
        )

        team_matches["team_score"] = np.where(
            team_matches["is_home"],
            team_matches["home_score"],
            team_matches["away_score"],
        )

        team_matches["opponent_score"] = np.where(
            team_matches["is_home"],
            team_matches["away_score"],
            team_matches["home_score"],
        )

        # Add result from team's perspective
        team_matches["team_result"] = np.where(
            team_matches["team_score"] > team_matches["opponent_score"],
            "W",
            np.where(
                team_matches["team_score"] < team_matches["opponent_score"], "L", "D"
            ),
        )

        # Cache the result
        self.team_matches[team_name] = team_matches

        return team_matches

    def get_recent_form(
        self, team: str, match_date: datetime, window_days: int = 90
    ) -> Dict[str, Any]:
        """Get a team's recent form before a specific match date.

        Args:
            team: Team name
            match_date: Date of the match
            window_days: Number of days to look back

        Returns:
            Dictionary with form statistics
        """
        # Get all matches for the team
        team_matches = self.get_team_matches(team)

        # Filter matches before match_date and within window
        start_date = match_date - timedelta(days=window_days)
        recent_matches = team_matches[
            (team_matches["date"] < match_date) & (team_matches["date"] >= start_date)
        ]

        # If not enough matches, return default values
        if len(recent_matches) < MIN_MATCHES:
            return {
                "matches_played": len(recent_matches),
                "win_rate": 0.5,  # Default values
                "draw_rate": 0.25,
                "loss_rate": 0.25,
                "goals_scored_avg": 1.0,
                "goals_conceded_avg": 1.0,
            }

        # Calculate form statistics
        home_matches = recent_matches[recent_matches["is_home"]]
        away_matches = recent_matches[~recent_matches["is_home"]]

        stats = {
            "matches_played": len(recent_matches),
            "win_rate": (recent_matches["team_result"] == "W").mean(),
            "draw_rate": (recent_matches["team_result"] == "D").mean(),
            "loss_rate": (recent_matches["team_result"] == "L").mean(),
            "goals_scored_avg": recent_matches["team_score"].mean(),
            "goals_conceded_avg": recent_matches["opponent_score"].mean(),
        }

        # Add home/away specific stats if available
        if len(home_matches) >= MIN_MATCHES:
            stats["home_win_rate"] = (home_matches["team_result"] == "W").mean()
            stats["home_goals_scored_avg"] = home_matches["team_score"].mean()
            stats["home_goals_conceded_avg"] = home_matches["opponent_score"].mean()

        if len(away_matches) >= MIN_MATCHES:
            stats["away_win_rate"] = (away_matches["team_result"] == "W").mean()
            stats["away_goals_scored_avg"] = away_matches["team_score"].mean()
            stats["away_goals_conceded_avg"] = away_matches["opponent_score"].mean()

        # Calculate form score (weighted recent results)
        # Last 5 matches with exponential decay
        last_matches = recent_matches.sort_values("date", ascending=False).head(5)

        if len(last_matches) > 0:
            # Convert results to points (W=3, D=1, L=0)
            points = [
                3 if r == "W" else (1 if r == "D" else 0)
                for r in last_matches["team_result"]
            ]

            # Calculate weighted average (more recent matches have higher weight)
            weights = [0.4, 0.25, 0.15, 0.1, 0.1][: len(points)]
            weights = [w / sum(weights) for w in weights]  # Normalize

            form_score = (
                sum(p * w for p, w in zip(points, weights)) / 3.0
            )  # Scale to 0-1
            stats["form_score"] = form_score
        else:
            stats["form_score"] = 0.5  # Neutral

        return stats

    def extract_match_stats(
        self, team_matches: pd.DataFrame, is_home: bool
    ) -> Dict[str, float]:
        """Extract statistics from match data.

        Args:
            team_matches: DataFrame with team matches
            is_home: Whether to extract home or away stats

        Returns:
            Dictionary with statistics
        """
        stats = {}
        prefix = "home" if is_home else "away"
        opposite = "away" if is_home else "home"

        # Extract full-time statistics
        for stat in MATCH_STATS:
            # Standard statistics (shots, possession, etc.)
            stat_col = f"full_time_stats.{stat}.{prefix}"
            if stat_col in team_matches.columns:
                stats[f"{stat}_avg"] = team_matches[stat_col].mean()

            # Calculate differential (team's stat - opponent's stat)
            opp_stat_col = f"full_time_stats.{stat}.{opposite}"
            if (
                stat_col in team_matches.columns
                and opp_stat_col in team_matches.columns
            ):
                diff = team_matches[stat_col] - team_matches[opp_stat_col]
                stats[f"{stat}_diff_avg"] = diff.mean()

        # Extract advanced stats if available
        for stat in ADVANCED_STATS:
            stat_col = f"full_time_stats.{stat}.{prefix}"
            if stat_col in team_matches.columns:
                stats[f"{stat}_avg"] = team_matches[stat_col].mean()

        return stats

    def get_team_features(
        self, team: str, match_date: datetime, is_home: bool
    ) -> Dict[str, Any]:
        """Get comprehensive features for a team for a specific match.

        Args:
            team: Team name
            match_date: Date of the match
            is_home: Whether the team is playing at home

        Returns:
            Dictionary with team features
        """
        features = {
            "team": team,
            "is_home": is_home,
        }

        # Add form features for different time windows
        for window in TIME_WINDOWS:
            form = self.get_recent_form(team, match_date, window_days=window)
            for k, v in form.items():
                features[f"{k}_{window}d"] = v

        # Add home/away specific performance
        team_matches = self.get_team_matches(team)
        past_matches = team_matches[team_matches["date"] < match_date]

        if len(past_matches) >= MIN_MATCHES:
            # Get home/away specific stats
            home_matches = past_matches[past_matches["is_home"]]
            away_matches = past_matches[~past_matches["is_home"]]

            if len(home_matches) >= MIN_MATCHES:
                home_stats = self.extract_match_stats(home_matches, True)
                for k, v in home_stats.items():
                    features[f"home_{k}"] = v

            if len(away_matches) >= MIN_MATCHES:
                away_stats = self.extract_match_stats(away_matches, False)
                for k, v in away_stats.items():
                    features[f"away_{k}"] = v

        return features
