"""
Extract team performance features from match data.

This module provides functionality for creating features related to
team performance, such as:
- Recent form (win/loss streaks)
- Rolling averages for various statistics
- Home/away performance metrics
- Team strength indicators
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TeamPerformanceFeatures:
    """Extract and create team performance features from match data."""

    def __init__(self, rolling_windows: List[int] = [3, 5, 10]):
        """
        Initialize the team performance feature extractor.

        Args:
            rolling_windows: List of window sizes for rolling averages (default: [3, 5, 10])
        """
        self.rolling_windows = rolling_windows
        logger.info(
            f"Initialized TeamPerformanceFeatures with rolling windows: {rolling_windows}"
        )

    def prepare_match_dataframe(self, matches: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Prepare a DataFrame from match data for feature extraction.

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
                date_val = match.get("date")
                match_dict = {
                    "match_id": match.get("match_id"),
                    "date": (
                        pd.to_datetime(date_val) if date_val is not None else pd.NaT
                    ),
                    "competition": match.get("competition"),
                    "season": match.get("season"),
                    "home_team": match.get("home_team"),
                    "away_team": match.get("away_team"),
                    "home_score": match.get("home_score"),
                    "away_score": match.get("away_score"),
                    "result": match.get("result"),
                }

                # Extract full time stats
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

    def calculate_team_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic result metrics for each team.

        Args:
            df: DataFrame with match data

        Returns:
            DataFrame with result metrics for each team
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for team results calculation")
            return pd.DataFrame()

        # Create a new DataFrame to hold team results
        all_teams = pd.unique(df[["home_team", "away_team"]].values.ravel("K"))
        logger.info(f"Calculating results for {len(all_teams)} teams")

        team_results = []

        for team in all_teams:
            # Home matches
            home_matches = df[df["home_team"] == team].copy()
            home_matches["team"] = team
            home_matches["opponent"] = home_matches["away_team"]
            home_matches["goals_scored"] = home_matches["home_score"]
            home_matches["goals_conceded"] = home_matches["away_score"]
            home_matches["is_home"] = True

            # Determine match result from team's perspective
            home_matches["team_result"] = "D"  # Draw by default
            home_matches.loc[
                home_matches["home_score"] > home_matches["away_score"], "team_result"
            ] = "W"
            home_matches.loc[
                home_matches["home_score"] < home_matches["away_score"], "team_result"
            ] = "L"

            # Away matches
            away_matches = df[df["away_team"] == team].copy()
            away_matches["team"] = team
            away_matches["opponent"] = away_matches["home_team"]
            away_matches["goals_scored"] = away_matches["away_score"]
            away_matches["goals_conceded"] = away_matches["home_score"]
            away_matches["is_home"] = False

            # Determine match result from team's perspective (opposite of home result)
            away_matches["team_result"] = "D"  # Draw by default
            away_matches.loc[
                away_matches["home_score"] < away_matches["away_score"], "team_result"
            ] = "W"
            away_matches.loc[
                away_matches["home_score"] > away_matches["away_score"], "team_result"
            ] = "L"

            # Combine home and away matches
            team_matches = pd.concat([home_matches, away_matches])

            # Add points
            team_matches["points"] = 1  # 1 point for draw by default
            team_matches.loc[team_matches["team_result"] == "W", "points"] = 3
            team_matches.loc[team_matches["team_result"] == "L", "points"] = 0

            # Sort by date
            team_matches = team_matches.sort_values("date")

            team_results.append(team_matches)

        # Combine all team results
        all_team_results = pd.concat(team_results)

        logger.info(
            f"Calculated results for teams, resulting in {len(all_team_results)} team-match entries"
        )
        return all_team_results

    def add_rolling_performance_metrics(
        self, team_results: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add rolling performance metrics for each team.

        Args:
            team_results: DataFrame with team results

        Returns:
            DataFrame with additional rolling performance metrics
        """
        if team_results.empty:
            logger.warning("Empty DataFrame provided for rolling performance metrics")
            return pd.DataFrame()

        # Create a copy to avoid modifying the original
        results = team_results.copy()

        # Ensure results are sorted by date within each team
        results = results.sort_values(["team", "date"])

        # Calculate metrics by team
        for team in results["team"].unique():
            team_data = results[results["team"] == team].copy()

            if len(team_data) == 0:
                continue

            # Calculate streaks
            team_data["current_streak"] = (
                team_data["team_result"].map({"W": 1, "D": 0, "L": -1}).cumsum()
            )
            results.loc[team_data.index, "current_streak"] = team_data["current_streak"]

            # Calculate rolling metrics for different window sizes
            for window in self.rolling_windows:
                # Skip if we don't have enough matches
                if len(team_data) < window:
                    logger.debug(
                        f"Not enough matches for team {team} to calculate {window}-match rolling metrics"
                    )
                    continue

                # Rolling points
                points_rolling = (
                    team_data["points"].rolling(window=window, min_periods=1).sum()
                )
                results.loc[team_data.index, f"points_last_{window}"] = points_rolling

                # Rolling win percentage
                win_rolling = (team_data["team_result"] == "W").astype(float).rolling(
                    window=window, min_periods=1
                ).mean() * 100
                results.loc[team_data.index, f"win_rate_last_{window}"] = win_rolling

                # Rolling goal metrics - calculate and assign separately
                goals_scored_rolling = (
                    team_data["goals_scored"]
                    .rolling(window=window, min_periods=1)
                    .sum()
                )
                results.loc[team_data.index, f"goals_scored_last_{window}"] = (
                    goals_scored_rolling
                )

                goals_conceded_rolling = (
                    team_data["goals_conceded"]
                    .rolling(window=window, min_periods=1)
                    .sum()
                )
                results.loc[team_data.index, f"goals_conceded_last_{window}"] = (
                    goals_conceded_rolling
                )

                # Calculate goal difference directly from the rolling sums
                results.loc[team_data.index, f"goal_diff_last_{window}"] = (
                    goals_scored_rolling - goals_conceded_rolling
                )

                # Rolling average goals
                avg_goals_scored = (
                    team_data["goals_scored"]
                    .rolling(window=window, min_periods=1)
                    .mean()
                )
                results.loc[team_data.index, f"avg_goals_scored_last_{window}"] = (
                    avg_goals_scored
                )

                avg_goals_conceded = (
                    team_data["goals_conceded"]
                    .rolling(window=window, min_periods=1)
                    .mean()
                )
                results.loc[team_data.index, f"avg_goals_conceded_last_{window}"] = (
                    avg_goals_conceded
                )

                # Add other relevant statistics if available
                for stat in [
                    "possession",
                    "shots_on_goal",
                    "shots_off_goal",
                    "corner_kicks",
                    "fouls",
                ]:
                    home_stat = f"home_{stat}"
                    away_stat = f"away_{stat}"

                    if home_stat in results.columns and away_stat in results.columns:
                        # Create temporary series for the team's stat regardless of home/away
                        team_stat = pd.Series(index=team_data.index, dtype=float)

                        # Fill with appropriate values
                        home_matches = team_data["is_home"] == True
                        away_matches = team_data["is_home"] == False

                        if any(home_matches):
                            team_stat[home_matches] = team_data.loc[
                                home_matches, home_stat
                            ].values

                        if any(away_matches):
                            team_stat[away_matches] = team_data.loc[
                                away_matches, away_stat
                            ].values

                        # Calculate rolling average
                        avg_stat = team_stat.rolling(
                            window=window, min_periods=1
                        ).mean()
                        results.loc[team_data.index, f"avg_{stat}_last_{window}"] = (
                            avg_stat
                        )

        # Fill NaN values with 0
        columns_to_fill = [
            col for col in results.columns if "last_" in col or col == "current_streak"
        ]
        results[columns_to_fill] = results[columns_to_fill].fillna(0)

        logger.info(
            f"Added rolling performance metrics with windows {self.rolling_windows}"
        )
        return results

    def add_home_away_performance(self, team_results: pd.DataFrame) -> pd.DataFrame:
        """
        Add separate home and away performance metrics.

        Args:
            team_results: DataFrame with team results

        Returns:
            DataFrame with additional home/away performance metrics
        """
        if team_results.empty:
            logger.warning("Empty DataFrame provided for home/away performance metrics")
            return pd.DataFrame()

        # Create a copy to avoid modifying the original
        results = team_results.copy()

        # Calculate home/away metrics by team
        for team in results["team"].unique():
            team_mask = results["team"] == team

            # Sort by date for this team
            team_data = results[team_mask].sort_values("date")

            # Calculate home metrics
            home_data = team_data[team_data["is_home"] == True]
            if len(home_data) > 0:
                # Create cumulative counts
                home_matches = np.arange(1, len(home_data) + 1)
                home_wins = (
                    (home_data["team_result"] == "W").astype(int).cumsum().values
                )
                home_draws = (
                    (home_data["team_result"] == "D").astype(int).cumsum().values
                )
                home_losses = (
                    (home_data["team_result"] == "L").astype(int).cumsum().values
                )
                home_points = home_data["points"].cumsum().values

                # Assign back to the team's data
                for i, idx in enumerate(home_data.index):
                    results.loc[idx, "home_matches_played"] = home_matches[i]
                    results.loc[idx, "home_wins"] = home_wins[i]
                    results.loc[idx, "home_draws"] = home_draws[i]
                    results.loc[idx, "home_losses"] = home_losses[i]
                    results.loc[idx, "home_points"] = home_points[i]
                    results.loc[idx, "home_win_rate"] = (
                        home_wins[i] / home_matches[i]
                    ) * 100

                # Goals
                home_goals_scored = home_data["goals_scored"].cumsum().values
                home_goals_conceded = home_data["goals_conceded"].cumsum().values

                for i, idx in enumerate(home_data.index):
                    results.loc[idx, "home_goals_scored"] = home_goals_scored[i]
                    results.loc[idx, "home_goals_conceded"] = home_goals_conceded[i]
                    results.loc[idx, "home_goal_diff"] = (
                        home_goals_scored[i] - home_goals_conceded[i]
                    )

            # Calculate away metrics
            away_data = team_data[team_data["is_home"] == False]
            if len(away_data) > 0:
                # Create cumulative counts
                away_matches = np.arange(1, len(away_data) + 1)
                away_wins = (
                    (away_data["team_result"] == "W").astype(int).cumsum().values
                )
                away_draws = (
                    (away_data["team_result"] == "D").astype(int).cumsum().values
                )
                away_losses = (
                    (away_data["team_result"] == "L").astype(int).cumsum().values
                )
                away_points = away_data["points"].cumsum().values

                # Assign back to the team's data
                for i, idx in enumerate(away_data.index):
                    results.loc[idx, "away_matches_played"] = away_matches[i]
                    results.loc[idx, "away_wins"] = away_wins[i]
                    results.loc[idx, "away_draws"] = away_draws[i]
                    results.loc[idx, "away_losses"] = away_losses[i]
                    results.loc[idx, "away_points"] = away_points[i]
                    results.loc[idx, "away_win_rate"] = (
                        away_wins[i] / away_matches[i]
                    ) * 100

                # Goals
                away_goals_scored = away_data["goals_scored"].cumsum().values
                away_goals_conceded = away_data["goals_conceded"].cumsum().values

                for i, idx in enumerate(away_data.index):
                    results.loc[idx, "away_goals_scored"] = away_goals_scored[i]
                    results.loc[idx, "away_goals_conceded"] = away_goals_conceded[i]
                    results.loc[idx, "away_goal_diff"] = (
                        away_goals_scored[i] - away_goals_conceded[i]
                    )

            # For matches without home/away stats, forward fill from team's history
            all_team_indices = results[team_mask].index
            for col in [
                "home_matches_played",
                "home_wins",
                "home_draws",
                "home_losses",
                "home_points",
                "home_win_rate",
                "home_goals_scored",
                "home_goals_conceded",
                "home_goal_diff",
                "away_matches_played",
                "away_wins",
                "away_draws",
                "away_losses",
                "away_points",
                "away_win_rate",
                "away_goals_scored",
                "away_goals_conceded",
                "away_goal_diff",
            ]:
                if col in results.columns:
                    # Fill NaN values by forward filling team values
                    # First get the series for this team and column, then fillna, then assign back
                    team_col_series = pd.Series(results.loc[all_team_indices, col])
                    results.loc[all_team_indices, col] = team_col_series.ffill()

        # Fill remaining NaN values with zeros
        home_away_cols = [
            col
            for col in results.columns
            if col.startswith("home_") or col.startswith("away_")
        ]
        results[home_away_cols] = results[home_away_cols].fillna(0)

        logger.info("Added home and away performance metrics")
        return results

    def add_form_indicators(
        self, team_results: pd.DataFrame, form_length: int = 5
    ) -> pd.DataFrame:
        """
        Add team form indicators based on recent results.

        Args:
            team_results: DataFrame with team results
            form_length: Number of recent matches to consider for form (default: 5)

        Returns:
            DataFrame with additional form indicators
        """
        if team_results.empty:
            logger.warning("Empty DataFrame provided for form indicators")
            return pd.DataFrame()

        # Create a copy to avoid modifying the original
        results = team_results.copy()

        # Convert result to numeric for easier calculations
        results["result_numeric"] = results["team_result"].map({"W": 3, "D": 1, "L": 0})

        # Calculate form by team
        for team in results["team"].unique():
            team_mask = results["team"] == team
            team_data = results[team_mask].sort_values("date")

            if len(team_data) < 1:
                continue

            # Get form as string of last N results (e.g., "WDLWW")
            for i in range(len(team_data)):
                # Use max to ensure we don't go below 0 when i < form_length
                start_idx = max(0, i - form_length + 1)
                end_idx = i + 1  # exclusive

                if (
                    i >= form_length - 1
                ):  # Only compute form string if we have enough matches
                    recent_matches = team_data.iloc[start_idx:end_idx]
                    form_string = "".join(recent_matches["team_result"].values)
                    results.loc[team_data.index[i], "form_string"] = form_string

                # Calculate form score with weights
                if i >= 0:  # Calculate for all matches
                    recent_matches = team_data.iloc[start_idx:end_idx]
                    form_values = recent_matches["result_numeric"].values

                    # Adjust weights to match the number of available matches
                    actual_length = len(form_values)
                    if actual_length > 0:
                        weights = np.exp(np.linspace(-1, 0, actual_length))
                        weights = weights / weights.sum()  # Normalize to sum to 1
                        form_score = np.sum(form_values * weights)
                        results.loc[team_data.index[i], "form_score"] = round(
                            form_score, 2
                        )

                    # Calculate momentum (form trend) if we have enough history
                    if i >= form_length:
                        prev_start_idx = max(0, i - form_length * 2 + 1)
                        prev_end_idx = i - form_length + 1
                        prev_matches = team_data.iloc[prev_start_idx:prev_end_idx]

                        if len(prev_matches) > 0:
                            prev_values = prev_matches["result_numeric"].values
                            prev_weights = np.exp(np.linspace(-1, 0, len(prev_values)))
                            prev_weights = prev_weights / prev_weights.sum()
                            prev_score = np.sum(prev_values * prev_weights)
                            results.loc[team_data.index[i], "momentum"] = round(
                                form_score - prev_score, 2
                            )

        # Drop temporary column
        results = results.drop(columns=["result_numeric"])

        # Fill NaN values for form indicators
        results["form_string"] = results["form_string"].fillna("")
        results["form_score"] = results["form_score"].fillna(0)
        results["momentum"] = results["momentum"].fillna(0)

        logger.info(f"Added form indicators with form length {form_length}")
        return results

    def extract_features(self, matches: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Extract all team performance features from match data.

        Args:
            matches: List of match data dictionaries

        Returns:
            DataFrame with team performance features
        """
        if not matches:
            logger.warning("No matches provided for feature extraction")
            return pd.DataFrame()

        logger.info(f"Extracting team performance features from {len(matches)} matches")

        # Prepare match dataframe
        match_df = self.prepare_match_dataframe(matches)

        # Calculate team results
        team_results = self.calculate_team_results(match_df)

        # Add rolling performance metrics
        team_results = self.add_rolling_performance_metrics(team_results)

        # Add home/away performance
        team_results = self.add_home_away_performance(team_results)

        # Add form indicators
        team_results = self.add_form_indicators(team_results)

        logger.info(
            f"Extracted team performance features: {len(team_results)} team-match entries with {len(team_results.columns)} features"
        )
        return team_results

    def create_match_features(
        self, team_results: pd.DataFrame, matches: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Create match features by combining home and away team performance.

        Args:
            team_results: DataFrame with team performance features
            matches: List of match data dictionaries

        Returns:
            DataFrame with match features
        """
        if team_results.empty or not matches:
            logger.warning(
                "Empty team results or no matches provided for match feature creation"
            )
            return pd.DataFrame()

        # Prepare a DataFrame of matches
        match_df = self.prepare_match_dataframe(matches)

        # Initialize a list to store match features
        match_features = []

        # Process each match
        for _, match in match_df.iterrows():
            match_id = match["match_id"]
            date = match["date"]
            home_team = match["home_team"]
            away_team = match["away_team"]

            # Get team data prior to the match
            home_data = team_results[
                (team_results["team"] == home_team) & (team_results["date"] < date)
            ].sort_values("date")
            away_data = team_results[
                (team_results["team"] == away_team) & (team_results["date"] < date)
            ].sort_values("date")

            # Skip if we don't have historical data for either team
            if home_data.empty or away_data.empty:
                logger.debug(
                    f"No historical data for match {match_id} (home: {home_team}, away: {away_team})"
                )
                continue

            # Get the most recent data for each team
            home_features = home_data.iloc[-1].copy()
            away_features = away_data.iloc[-1].copy()

            # Create a dictionary with match information
            feature_dict = {
                "match_id": match_id,
                "date": date,
                "competition": match["competition"],
                "season": match["season"],
                "home_team": home_team,
                "away_team": away_team,
                "home_score": match["home_score"],
                "away_score": match["away_score"],
                "result": match["result"],
            }

            # Add prefixed team performance features
            for col in home_features.index:
                # Skip certain columns
                if col in ["match_id", "date", "team", "opponent", "is_home"]:
                    continue

                feature_dict[f"home_{col}"] = home_features[col]
                feature_dict[f"away_{col}"] = away_features[col]

                # Calculate differentials for numeric features
                if isinstance(home_features[col], (int, float)) and isinstance(
                    away_features[col], (int, float)
                ):
                    feature_dict[f"diff_{col}"] = (
                        home_features[col] - away_features[col]
                    )

            match_features.append(feature_dict)

        # Convert to DataFrame
        match_feature_df = pd.DataFrame(match_features)

        logger.info(f"Created features for {len(match_feature_df)} matches")
        return match_feature_df
