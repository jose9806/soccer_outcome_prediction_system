# src/feature_engineering/extractors/advanced_metrics.py
"""
Extract advanced metrics from match data.

This module provides functionality for creating advanced metrics such as:
- Expected Goals (xG) derivatives
- Defensive and offensive efficiency
- Set piece effectiveness
- Possession quality metrics
- Team strength indicators
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class AdvancedMetrics:
    """Extract and create advanced soccer metrics from match data."""

    def __init__(self, rolling_windows: List[int] = [3, 5, 10]):
        """
        Initialize the advanced metrics extractor.

        Args:
            rolling_windows: List of window sizes for rolling metrics (default: [3, 5, 10])
        """
        self.rolling_windows = rolling_windows
        logger.info(
            f"Initialized AdvancedMetrics with rolling windows: {rolling_windows}"
        )

    def prepare_match_dataframe(self, matches: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Prepare a DataFrame from match data for metrics calculation.

        Args:
            matches: List of match data dictionaries

        Returns:
            DataFrame with relevant match data
        """
        if not matches:
            logger.warning("No matches provided for DataFrame preparation")
            return pd.DataFrame()

        # Extract relevant fields from match data
        match_data = []

        for match in matches:
            try:
                match_dict = {
                    "match_id": match.get("match_id"),
                    "date": pd.to_datetime(match.get("date")),
                    "competition": match.get("competition"),
                    "season": match.get("season"),
                    "home_team": match.get("home_team"),
                    "away_team": match.get("away_team"),
                    "home_score": match.get("home_score"),
                    "away_score": match.get("away_score"),
                    "result": match.get("result"),
                }

                # Extract all stats from full_time_stats
                full_time_stats = match.get("full_time_stats", {})
                if full_time_stats:
                    for stat_name, stat_values in full_time_stats.items():
                        if (
                            isinstance(stat_values, dict)
                            and "home" in stat_values
                            and "away" in stat_values
                        ):
                            match_dict[f"home_{stat_name}"] = stat_values.get("home")
                            match_dict[f"away_{stat_name}"] = stat_values.get("away")

                match_data.append(match_dict)
            except Exception as e:
                logger.error(
                    f"Error processing match {match.get('match_id', 'unknown')}: {e}"
                )
                continue

        df = pd.DataFrame(match_data)

        # Sort by date
        if "date" in df.columns:
            df = df.sort_values("date")

        logger.info(
            f"Prepared DataFrame with {len(df)} matches and {len(df.columns)} columns"
        )
        return df

    def calculate_shot_efficiency(self, match_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate shot efficiency metrics.

        Args:
            match_df: DataFrame with match data

        Returns:
            DataFrame with shot efficiency metrics added
        """
        if match_df.empty:
            logger.warning("Empty DataFrame provided for shot efficiency calculation")
            return pd.DataFrame()

        # Create a copy to avoid modifying the original
        df = match_df.copy()

        # Calculate shot efficiency metrics for home team
        if "home_shots_on_goal" in df.columns and "home_goal_attempts" in df.columns:
            # Shot accuracy: percentage of shots that are on target
            df["home_shot_accuracy"] = np.where(
                df["home_goal_attempts"] > 0,
                df["home_shots_on_goal"] / df["home_goal_attempts"] * 100,
                0,
            )

            # Shot conversion: percentage of shots on target that result in goals
            df["home_shot_conversion"] = np.where(
                df["home_shots_on_goal"] > 0,
                df["home_score"] / df["home_shots_on_goal"] * 100,
                0,
            )

            # Overall shot efficiency: percentage of all shots that result in goals
            df["home_shot_efficiency"] = np.where(
                df["home_goal_attempts"] > 0,
                df["home_score"] / df["home_goal_attempts"] * 100,
                0,
            )

        # Calculate shot efficiency metrics for away team
        if "away_shots_on_goal" in df.columns and "away_goal_attempts" in df.columns:
            # Shot accuracy: percentage of shots that are on target
            df["away_shot_accuracy"] = np.where(
                df["away_goal_attempts"] > 0,
                df["away_shots_on_goal"] / df["away_goal_attempts"] * 100,
                0,
            )

            # Shot conversion: percentage of shots on target that result in goals
            df["away_shot_conversion"] = np.where(
                df["away_shots_on_goal"] > 0,
                df["away_score"] / df["away_shots_on_goal"] * 100,
                0,
            )

            # Overall shot efficiency: percentage of all shots that result in goals
            df["away_shot_efficiency"] = np.where(
                df["away_goal_attempts"] > 0,
                df["away_score"] / df["away_goal_attempts"] * 100,
                0,
            )

        logger.info("Added shot efficiency metrics")
        return df

    def calculate_defensive_metrics(self, match_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate defensive metrics.

        Args:
            match_df: DataFrame with match data

        Returns:
            DataFrame with defensive metrics added
        """
        if match_df.empty:
            logger.warning("Empty DataFrame provided for defensive metrics calculation")
            return pd.DataFrame()

        # Create a copy to avoid modifying the original
        df = match_df.copy()

        # Calculate defensive metrics for home team
        if "home_goalkeeper_saves" in df.columns and "away_shots_on_goal" in df.columns:
            # Save percentage: percentage of shots on target that are saved
            df["home_save_percentage"] = np.where(
                df["away_shots_on_goal"] > 0,
                df["home_goalkeeper_saves"] / df["away_shots_on_goal"] * 100,
                0,
            )

            # Goals conceded per shot on target
            df["home_goals_per_shot_on_target_against"] = np.where(
                df["away_shots_on_goal"] > 0,
                df["away_score"] / df["away_shots_on_goal"],
                0,
            )

        # Calculate defensive metrics for away team
        if "away_goalkeeper_saves" in df.columns and "home_shots_on_goal" in df.columns:
            # Save percentage: percentage of shots on target that are saved
            df["away_save_percentage"] = np.where(
                df["home_shots_on_goal"] > 0,
                df["away_goalkeeper_saves"] / df["home_shots_on_goal"] * 100,
                0,
            )

            # Goals conceded per shot on target
            df["away_goals_per_shot_on_target_against"] = np.where(
                df["home_shots_on_goal"] > 0,
                df["home_score"] / df["home_shots_on_goal"],
                0,
            )

        # Add defensive pressure metrics if data is available
        if "home_fouls" in df.columns and "away_fouls" in df.columns:
            # Calculate defensive pressure (fouls per opposing possession %)
            df["home_defensive_pressure"] = np.where(
                df["away_possession"] > 0,
                df["home_fouls"] / (df["away_possession"] / 100),
                0,
            )

            df["away_defensive_pressure"] = np.where(
                df["home_possession"] > 0,
                df["away_fouls"] / (df["home_possession"] / 100),
                0,
            )

        logger.info("Added defensive metrics")
        return df

    def calculate_possession_metrics(self, match_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate possession quality metrics.

        Args:
            match_df: DataFrame with match data

        Returns:
            DataFrame with possession quality metrics added
        """
        if match_df.empty:
            logger.warning(
                "Empty DataFrame provided for possession metrics calculation"
            )
            return pd.DataFrame()

        # Create a copy to avoid modifying the original
        df = match_df.copy()

        # Calculate possession efficiency for home team
        if "home_possession" in df.columns:
            # Possession efficiency: goals per % of possession
            df["home_possession_efficiency"] = np.where(
                df["home_possession"] > 0,
                df["home_score"] / (df["home_possession"] / 100),
                0,
            )

            # Shot generation: shots per % of possession
            if "home_goal_attempts" in df.columns:
                df["home_shot_generation"] = np.where(
                    df["home_possession"] > 0,
                    df["home_goal_attempts"] / (df["home_possession"] / 100),
                    0,
                )

            # Chances per possession: big chances per % of possession
            if "home_big_chances" in df.columns:
                df["home_chances_per_possession"] = np.where(
                    df["home_possession"] > 0,
                    df["home_big_chances"] / (df["home_possession"] / 100),
                    0,
                )

        # Calculate possession efficiency for away team
        if "away_possession" in df.columns:
            # Possession efficiency: goals per % of possession
            df["away_possession_efficiency"] = np.where(
                df["away_possession"] > 0,
                df["away_score"] / (df["away_possession"] / 100),
                0,
            )

            # Shot generation: shots per % of possession
            if "away_goal_attempts" in df.columns:
                df["away_shot_generation"] = np.where(
                    df["away_possession"] > 0,
                    df["away_goal_attempts"] / (df["away_possession"] / 100),
                    0,
                )

            # Chances per possession: big chances per % of possession
            if "away_big_chances" in df.columns:
                df["away_chances_per_possession"] = np.where(
                    df["away_possession"] > 0,
                    df["away_big_chances"] / (df["away_possession"] / 100),
                    0,
                )

        logger.info("Added possession efficiency metrics")
        return df

    def calculate_set_piece_metrics(self, match_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate set piece effectiveness metrics.

        Args:
            match_df: DataFrame with match data

        Returns:
            DataFrame with set piece metrics added
        """
        if match_df.empty:
            logger.warning("Empty DataFrame provided for set piece metrics calculation")
            return pd.DataFrame()

        # Create a copy to avoid modifying the original
        df = match_df.copy()

        # Set piece effectiveness requires additional data not present in the standard match stats
        # For example, goals from corners or free kicks
        # As a proxy, we can use corner kicks to estimate set piece threat

        if "home_corner_kicks" in df.columns and "away_corner_kicks" in df.columns:
            # Calculate corner threat (corners per goal)
            df["home_corner_per_goal"] = np.where(
                df["home_score"] > 0,
                df["home_corner_kicks"] / df["home_score"],
                df[
                    "home_corner_kicks"
                ],  # If no goals, just return the number of corners
            )

            df["away_corner_per_goal"] = np.where(
                df["away_score"] > 0,
                df["away_corner_kicks"] / df["away_score"],
                df[
                    "away_corner_kicks"
                ],  # If no goals, just return the number of corners
            )

            # Corner efficiency (inverse of corners per goal)
            df["home_corner_efficiency"] = np.where(
                df["home_corner_kicks"] > 0,
                df["home_score"] / df["home_corner_kicks"] * 100,
                0,
            )

            df["away_corner_efficiency"] = np.where(
                df["away_corner_kicks"] > 0,
                df["away_score"] / df["away_corner_kicks"] * 100,
                0,
            )

        logger.info("Added set piece metrics")
        return df

    def calculate_expected_goals_metrics(self, match_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate metrics derived from Expected Goals (xG).

        Args:
            match_df: DataFrame with match data

        Returns:
            DataFrame with xG-derived metrics added
        """
        if match_df.empty:
            logger.warning("Empty DataFrame provided for xG metrics calculation")
            return pd.DataFrame()

        # Create a copy to avoid modifying the original
        df = match_df.copy()

        # Check if xG data is available
        if "home_expected_goals" in df.columns and "away_expected_goals" in df.columns:
            # xG difference (home - away)
            df["xg_difference"] = df["home_expected_goals"] - df["away_expected_goals"]

            # xG ratio (home / total)
            df["home_xg_ratio"] = np.where(
                df["home_expected_goals"] + df["away_expected_goals"] > 0,
                df["home_expected_goals"]
                / (df["home_expected_goals"] + df["away_expected_goals"]),
                0.5,  # Default to 0.5 if total xG is 0
            )

            # xG efficiency: actual goals / expected goals
            df["home_xg_efficiency"] = np.where(
                df["home_expected_goals"] > 0,
                df["home_score"] / df["home_expected_goals"],
                0,
            )

            df["away_xg_efficiency"] = np.where(
                df["away_expected_goals"] > 0,
                df["away_score"] / df["away_expected_goals"],
                0,
            )

            # xG overperformance: actual goals - expected goals
            df["home_xg_overperformance"] = df["home_score"] - df["home_expected_goals"]
            df["away_xg_overperformance"] = df["away_score"] - df["away_expected_goals"]

            # Expected points based on xG (using a probabilistic model)
            # This is a simplified version; a more sophisticated model would use a Poisson distribution

            # Calculate win probability for home team
            # Assuming goals follow a Poisson-like distribution
            home_win_prob = np.where(
                df["home_expected_goals"] + df["away_expected_goals"] > 0,
                df["home_expected_goals"]
                / (df["home_expected_goals"] + df["away_expected_goals"]),
                0.5,
            )

            # Adjust for home advantage
            home_win_prob = home_win_prob * 1.1  # Simple adjustment factor
            home_win_prob = np.clip(home_win_prob, 0, 1)

            # Calculate draw probability (simplified)
            draw_prob = np.where(
                (df["home_expected_goals"] > 0) & (df["away_expected_goals"] > 0),
                (
                    1
                    - np.abs(df["home_expected_goals"] - df["away_expected_goals"])
                    / (df["home_expected_goals"] + df["away_expected_goals"])
                )
                * 0.3,  # Scale factor for draw probability
                0.2,  # Default draw probability
            )

            draw_prob = np.clip(draw_prob, 0, 0.5)  # Cap draw probability

            # Away win probability
            away_win_prob = 1 - home_win_prob - draw_prob
            away_win_prob = np.clip(away_win_prob, 0, 1)

            # Normalize probabilities to sum to 1
            total_prob = home_win_prob + draw_prob + away_win_prob
            home_win_prob = home_win_prob / total_prob
            draw_prob = draw_prob / total_prob
            away_win_prob = away_win_prob / total_prob

            # Calculate expected points
            df["home_expected_points"] = home_win_prob * 3 + draw_prob * 1
            df["away_expected_points"] = away_win_prob * 3 + draw_prob * 1

            # Points overperformance: actual points - expected points
            df["home_actual_points"] = np.where(
                df["result"] == "W", 3, np.where(df["result"] == "D", 1, 0)
            )

            df["away_actual_points"] = np.where(
                df["result"] == "L", 3, np.where(df["result"] == "D", 1, 0)
            )

            df["home_points_overperformance"] = (
                df["home_actual_points"] - df["home_expected_points"]
            )
            df["away_points_overperformance"] = (
                df["away_actual_points"] - df["away_expected_points"]
            )

        logger.info("Added xG-derived metrics")
        return df

    def calculate_team_metrics(self, match_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate team-level advanced metrics by aggregating match data.

        Args:
            match_df: DataFrame with match data and calculated metrics

        Returns:
            DataFrame with team metrics
        """
        if match_df.empty:
            logger.warning("Empty DataFrame provided for team metrics calculation")
            return pd.DataFrame()

        # Create a copy of the dataframe with processed metrics
        df = match_df.copy()

        # Initialize a list to store team metrics
        team_metrics = []

        # Get all unique teams
        all_teams = pd.unique(df[["home_team", "away_team"]].values.ravel("K"))

        # Calculate metrics for each team
        for team in all_teams:
            # Get home and away matches for this team
            home_matches = df[df["home_team"] == team]
            away_matches = df[df["away_team"] == team]

            # Combine metrics from home and away matches
            team_metric = {
                "team": team,
                "matches_played": len(home_matches) + len(away_matches),
                "home_matches": len(home_matches),
                "away_matches": len(away_matches),
            }

            if team_metric["matches_played"] == 0:
                continue

            # Offensive metrics
            if len(home_matches) > 0:
                team_metric["home_goals_per_game"] = home_matches["home_score"].mean()

                if "home_shot_efficiency" in home_matches.columns:
                    team_metric["home_shot_efficiency"] = home_matches[
                        "home_shot_efficiency"
                    ].mean()

                if "home_possession_efficiency" in home_matches.columns:
                    team_metric["home_possession_efficiency"] = home_matches[
                        "home_possession_efficiency"
                    ].mean()

                if "home_xg_efficiency" in home_matches.columns:
                    team_metric["home_xg_efficiency"] = home_matches[
                        "home_xg_efficiency"
                    ].mean()

            if len(away_matches) > 0:
                team_metric["away_goals_per_game"] = away_matches["away_score"].mean()

                if "away_shot_efficiency" in away_matches.columns:
                    team_metric["away_shot_efficiency"] = away_matches[
                        "away_shot_efficiency"
                    ].mean()

                if "away_possession_efficiency" in away_matches.columns:
                    team_metric["away_possession_efficiency"] = away_matches[
                        "away_possession_efficiency"
                    ].mean()

                if "away_xg_efficiency" in away_matches.columns:
                    team_metric["away_xg_efficiency"] = away_matches[
                        "away_xg_efficiency"
                    ].mean()

            # Defensive metrics
            if len(home_matches) > 0:
                team_metric["home_goals_conceded_per_game"] = home_matches[
                    "away_score"
                ].mean()

                if "home_save_percentage" in home_matches.columns:
                    team_metric["home_save_percentage"] = home_matches[
                        "home_save_percentage"
                    ].mean()

            if len(away_matches) > 0:
                team_metric["away_goals_conceded_per_game"] = away_matches[
                    "home_score"
                ].mean()

                if "away_save_percentage" in away_matches.columns:
                    team_metric["away_save_percentage"] = away_matches[
                        "away_save_percentage"
                    ].mean()

            # Overall metrics
            team_metric["goals_per_game"] = (
                sum(home_matches["home_score"]) + sum(away_matches["away_score"])
            ) / team_metric["matches_played"]

            team_metric["goals_conceded_per_game"] = (
                sum(home_matches["away_score"]) + sum(away_matches["home_score"])
            ) / team_metric["matches_played"]

            team_metric["goal_difference_per_game"] = (
                team_metric["goals_per_game"] - team_metric["goals_conceded_per_game"]
            )

            # Points and performance
            home_points = sum(
                [
                    3 if r == "W" else (1 if r == "D" else 0)
                    for r in home_matches["result"]
                ]
            )
            away_points = sum(
                [
                    3 if r == "L" else (1 if r == "D" else 0)
                    for r in away_matches["result"]
                ]
            )

            team_metric["total_points"] = home_points + away_points
            team_metric["points_per_game"] = (
                team_metric["total_points"] / team_metric["matches_played"]
            )

            # xG metrics if available
            if (
                "home_expected_goals" in home_matches.columns
                and "away_expected_goals" in away_matches.columns
            ):
                team_metric["total_xg"] = sum(
                    home_matches["home_expected_goals"]
                ) + sum(away_matches["away_expected_goals"])
                team_metric["total_xg_against"] = sum(
                    home_matches["away_expected_goals"]
                ) + sum(away_matches["home_expected_goals"])

                team_metric["xg_per_game"] = (
                    team_metric["total_xg"] / team_metric["matches_played"]
                )
                team_metric["xg_against_per_game"] = (
                    team_metric["total_xg_against"] / team_metric["matches_played"]
                )
                team_metric["xg_difference_per_game"] = (
                    team_metric["xg_per_game"] - team_metric["xg_against_per_game"]
                )

                # xG overperformance
                team_metric["goals_vs_xg_ratio"] = (
                    (sum(home_matches["home_score"]) + sum(away_matches["away_score"]))
                    / team_metric["total_xg"]
                    if team_metric["total_xg"] > 0
                    else 1
                )

                # Points overperformance if available
                if (
                    "home_expected_points" in home_matches.columns
                    and "away_expected_points" in away_matches.columns
                ):
                    team_metric["total_expected_points"] = sum(
                        home_matches["home_expected_points"]
                    ) + sum(away_matches["away_expected_points"])
                    team_metric["expected_points_per_game"] = (
                        team_metric["total_expected_points"]
                        / team_metric["matches_played"]
                    )
                    team_metric["points_overperformance"] = (
                        team_metric["total_points"]
                        - team_metric["total_expected_points"]
                    )
                    team_metric["points_overperformance_per_game"] = (
                        team_metric["points_overperformance"]
                        / team_metric["matches_played"]
                    )

            team_metrics.append(team_metric)

        # Convert to DataFrame
        team_metrics_df = pd.DataFrame(team_metrics)

        logger.info(f"Calculated advanced metrics for {len(team_metrics_df)} teams")
        return team_metrics_df

    def calculate_rolling_team_metrics(self, match_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rolling team metrics for each match.

        Args:
            match_df: DataFrame with match data and calculated metrics

        Returns:
            DataFrame with rolling team metrics for each match
        """
        if match_df.empty:
            logger.warning(
                "Empty DataFrame provided for rolling team metrics calculation"
            )
            return pd.DataFrame()

        # Create a copy of the dataframe with processed metrics
        df = match_df.copy()

        # Sort by date
        df = df.sort_values("date")

        # Get all teams
        all_teams = pd.unique(df[["home_team", "away_team"]].values.ravel("K"))

        # Create a dictionary to store team metrics by match
        match_team_metrics = {}

        # Calculate rolling metrics for each team
        for team in all_teams:
            # Get home and away matches for this team
            team_matches = pd.concat(
                [
                    df[df["home_team"] == team].assign(is_home=True),
                    df[df["away_team"] == team].assign(is_home=False),
                ]
            ).sort_values("date")

            if len(team_matches) == 0:
                continue

            # Calculate rolling metrics for each window size
            for window in self.rolling_windows:
                if len(team_matches) < window:
                    continue

                # Apply rolling window to calculate metrics
                team_matches[f"rolling_{window}_goals_for"] = np.where(
                    team_matches["is_home"],
                    team_matches["home_score"].rolling(window, min_periods=1).mean(),
                    team_matches["away_score"].rolling(window, min_periods=1).mean(),
                )

                team_matches[f"rolling_{window}_goals_against"] = np.where(
                    team_matches["is_home"],
                    team_matches["away_score"].rolling(window, min_periods=1).mean(),
                    team_matches["home_score"].rolling(window, min_periods=1).mean(),
                )

                # Calculate goal difference
                team_matches[f"rolling_{window}_goal_diff"] = (
                    team_matches[f"rolling_{window}_goals_for"]
                    - team_matches[f"rolling_{window}_goals_against"]
                )

                # Calculate points (assuming W=3, D=1, L=0)
                team_matches[f"rolling_{window}_points"] = (
                    team_matches.apply(
                        lambda row: (
                            3
                            if (row["is_home"] and row["result"] == "W")
                            or (not row["is_home"] and row["result"] == "L")
                            else 1 if row["result"] == "D" else 0
                        ),
                        axis=1,
                    )
                    .rolling(window, min_periods=1)
                    .mean()
                )

                # Add xG metrics if available
                if (
                    "home_expected_goals" in team_matches.columns
                    and "away_expected_goals" in team_matches.columns
                ):

                    team_matches[f"rolling_{window}_xg_for"] = np.where(
                        team_matches["is_home"],
                        team_matches["home_expected_goals"]
                        .rolling(window, min_periods=1)
                        .mean(),
                        team_matches["away_expected_goals"]
                        .rolling(window, min_periods=1)
                        .mean(),
                    )

                    team_matches[f"rolling_{window}_xg_against"] = np.where(
                        team_matches["is_home"],
                        team_matches["away_expected_goals"]
                        .rolling(window, min_periods=1)
                        .mean(),
                        team_matches["home_expected_goals"]
                        .rolling(window, min_periods=1)
                        .mean(),
                    )

                    team_matches[f"rolling_{window}_xg_diff"] = (
                        team_matches[f"rolling_{window}_xg_for"]
                        - team_matches[f"rolling_{window}_xg_against"]
                    )

                # Add shot efficiency metrics if available
                if all(
                    col in team_matches.columns
                    for col in ["home_shot_efficiency", "away_shot_efficiency"]
                ):

                    team_matches[f"rolling_{window}_shot_efficiency"] = np.where(
                        team_matches["is_home"],
                        team_matches["home_shot_efficiency"]
                        .rolling(window, min_periods=1)
                        .mean(),
                        team_matches["away_shot_efficiency"]
                        .rolling(window, min_periods=1)
                        .mean(),
                    )

                # Add possession efficiency metrics if available
                if all(
                    col in team_matches.columns
                    for col in [
                        "home_possession_efficiency",
                        "away_possession_efficiency",
                    ]
                ):

                    team_matches[f"rolling_{window}_possession_efficiency"] = np.where(
                        team_matches["is_home"],
                        team_matches["home_possession_efficiency"]
                        .rolling(window, min_periods=1)
                        .mean(),
                        team_matches["away_possession_efficiency"]
                        .rolling(window, min_periods=1)
                        .mean(),
                    )

            # Store the calculated metrics by match ID
            for _, row in team_matches.iterrows():
                match_id = row["match_id"]

                if match_id not in match_team_metrics:
                    match_team_metrics[match_id] = {}

                team_key = "home_team" if row["is_home"] else "away_team"

                # Store all calculated rolling metrics
                for col in row.index:
                    if col.startswith("rolling_"):
                        metric_key = f"{team_key}_{col}"
                        match_team_metrics[match_id][metric_key] = row[col]

        # Add the rolling metrics to the original dataframe
        for match_id, metrics in match_team_metrics.items():
            match_idx = df[df["match_id"] == match_id].index

            if len(match_idx) > 0:
                for metric_key, value in metrics.items():
                    df.loc[match_idx, metric_key] = value

        logger.info("Added rolling team metrics to match data")
        return df

    def extract_features(self, matches: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Extract all advanced metrics features from match data.

        Args:
            matches: List of match data dictionaries

        Returns:
            DataFrame with advanced metrics features
        """
        if not matches:
            logger.warning("No matches provided for feature extraction")
            return pd.DataFrame()

        logger.info(f"Extracting advanced metrics from {len(matches)} matches")

        # Prepare match dataframe
        match_df = self.prepare_match_dataframe(matches)

        # Calculate shot efficiency metrics
        match_df = self.calculate_shot_efficiency(match_df)

        # Calculate defensive metrics
        match_df = self.calculate_defensive_metrics(match_df)

        # Calculate possession metrics
        match_df = self.calculate_possession_metrics(match_df)

        # Calculate set piece metrics
        match_df = self.calculate_set_piece_metrics(match_df)

        # Calculate expected goals metrics
        match_df = self.calculate_expected_goals_metrics(match_df)

        # Calculate rolling team metrics
        match_df = self.calculate_rolling_team_metrics(match_df)

        logger.info(
            f"Extracted advanced metrics: {len(match_df)} matches with {len(match_df.columns)} metrics"
        )
        return match_df

    def get_team_strength_metrics(self, matches: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Calculate team strength metrics based on advanced statistics.

        Args:
            matches: List of match data dictionaries

        Returns:
            DataFrame with team strength metrics
        """
        # Extract advanced metrics
        match_df = self.extract_features(matches)

        # Calculate team-level metrics
        team_metrics = self.calculate_team_metrics(match_df)

        logger.info(f"Calculated team strength metrics for {len(team_metrics)} teams")
        return team_metrics
