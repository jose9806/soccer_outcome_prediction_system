"""
Extract temporal features from match data.

This module provides functionality for creating features related to:
- Time-based patterns (day of week, time of day, season phase)
- Team momentum and form over time
- Temporal trends in performance metrics
- Seasonality effects on match outcomes
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import calendar

logger = logging.getLogger(__name__)


class TemporalFeatures:
    """Extract and create temporal features from match data."""

    def __init__(self):
        """Initialize the temporal features extractor."""
        logger.info("Initialized TemporalFeatures")

    def prepare_match_dataframe(self, matches: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Prepare a DataFrame from match data for temporal feature extraction.

        Args:
            matches: List of match data dictionaries

        Returns:
            DataFrame with match data
        """
        if not matches:
            logger.warning("No matches provided for DataFrame preparation")
            return pd.DataFrame()

        # Extract basic match information
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

                match_data.append(match_dict)
            except Exception as e:
                logger.error(
                    f"Error processing match {match.get('match_id', 'unknown')}: {e}"
                )
                continue

        df = pd.DataFrame(match_data)

        # Convert date to datetime if not already
        if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(
            df["date"]
        ):
            df["date"] = pd.to_datetime(df["date"])

        # Sort by date
        if "date" in df.columns:
            df = df.sort_values("date")

        logger.info(f"Prepared DataFrame with {len(df)} matches")
        return df

    def add_calendar_features(self, match_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add calendar-based features.

        Args:
            match_df: DataFrame with match data

        Returns:
            DataFrame with calendar features added
        """
        if match_df.empty or "date" not in match_df.columns:
            logger.warning(
                "Empty DataFrame or missing date column provided for calendar features"
            )
            return match_df

        # Create a copy to avoid modifying the original
        df = match_df.copy()

        # Extract date components
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["dayofweek"] = df["date"].dt.dayofweek  # 0=Monday, 6=Sunday
        df["weekend"] = (
            df["dayofweek"].isin([5, 6]).astype(int)
        )  # 1 for weekend matches
        df["week_of_year"] = df["date"].dt.isocalendar().week
        df["quarter"] = df["date"].dt.quarter

        # Time of day features
        df["hour"] = df["date"].dt.hour

        # Create time of day category
        df["time_of_day"] = "afternoon"
        df.loc[df["hour"] < 12, "time_of_day"] = "morning"
        df.loc[df["hour"] >= 17, "time_of_day"] = "evening"
        df.loc[df["hour"] >= 20, "time_of_day"] = "night"

        # Season names
        seasons = {
            12: "winter",
            1: "winter",
            2: "winter",
            3: "spring",
            4: "spring",
            5: "spring",
            6: "summer",
            7: "summer",
            8: "summer",
            9: "autumn",
            10: "autumn",
            11: "autumn",
        }

        df["season_of_year"] = df["month"].map(seasons)

        # Day name
        df["day_name"] = df["dayofweek"].map(
            {
                0: "Monday",
                1: "Tuesday",
                2: "Wednesday",
                3: "Thursday",
                4: "Friday",
                5: "Saturday",
                6: "Sunday",
            }
        )

        # Month name
        df["month_name"] = df["month"].map(
            {
                1: "January",
                2: "February",
                3: "March",
                4: "April",
                5: "May",
                6: "June",
                7: "July",
                8: "August",
                9: "September",
                10: "October",
                11: "November",
                12: "December",
            }
        )

        logger.info("Added calendar features")
        return df

    def add_season_phase_features(self, match_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features related to the phase of the season.

        Args:
            match_df: DataFrame with match data

        Returns:
            DataFrame with season phase features added
        """
        if match_df.empty or "date" not in match_df.columns:
            logger.warning(
                "Empty DataFrame or missing date column provided for season phase features"
            )
            return match_df

        # Create a copy to avoid modifying the original
        df = match_df.copy()

        # Group by season and competition
        grouped = df.groupby(["season", "competition"])

        for (season, competition), group in grouped:
            # Sort by date
            group = group.sort_values("date")

            # Calculate the total duration of the season
            season_start = group["date"].min()
            season_end = group["date"].max()
            season_duration = (season_end - season_start).total_seconds()

            if season_duration == 0:
                logger.warning(
                    f"Season {season}, competition {competition} has 0 duration. Skipping."
                )
                continue

            # Calculate the proportion of the season completed for each match
            for idx in group.index:
                match_date = df.loc[idx, "date"]
                proportion = (
                    match_date - season_start
                ).total_seconds() / season_duration

                df.at[idx, "season_proportion"] = round(proportion, 3)

                # Divide season into phases (0-0.25: early, 0.25-0.75: mid, 0.75-1: late)
                if proportion < 0.25:
                    phase = "early"
                elif proportion < 0.75:
                    phase = "middle"
                else:
                    phase = "late"

                df.at[idx, "season_phase"] = phase

                # Calculate days elapsed and remaining in the season
                days_elapsed = (match_date - season_start).days
                days_remaining = (season_end - match_date).days

                df.at[idx, "days_elapsed_in_season"] = days_elapsed
                df.at[idx, "days_remaining_in_season"] = days_remaining

        # Fill any missing values
        df["season_proportion"] = df["season_proportion"].fillna(0)
        df["season_phase"] = df["season_phase"].fillna("unknown")
        df["days_elapsed_in_season"] = df["days_elapsed_in_season"].fillna(0)
        df["days_remaining_in_season"] = df["days_remaining_in_season"].fillna(0)

        logger.info("Added season phase features")
        return df

    def add_team_rest_features(self, match_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features related to team rest between matches.

        Args:
            match_df: DataFrame with match data

        Returns:
            DataFrame with team rest features added
        """
        if match_df.empty or "date" not in match_df.columns:
            logger.warning(
                "Empty DataFrame or missing date column provided for team rest features"
            )
            return match_df

        # Create a copy to avoid modifying the original
        df = match_df.copy()

        # Sort by date
        df = df.sort_values("date")

        # Dictionary to store the last match date for each team
        last_match_date = {}
        last_match_id = {}

        # Process each match
        for idx, match in df.iterrows():
            home_team = match["home_team"]
            away_team = match["away_team"]
            match_date = match["date"]
            match_id = match["match_id"]

            # Calculate rest days for home team
            if home_team in last_match_date:
                home_rest_days = (match_date - last_match_date[home_team]).days
                df.at[idx, "home_rest_days"] = home_rest_days
                df.at[idx, "home_previous_match_id"] = last_match_id[home_team]
            else:
                df.at[idx, "home_rest_days"] = None
                df.at[idx, "home_previous_match_id"] = None

            # Calculate rest days for away team
            if away_team in last_match_date:
                away_rest_days = (match_date - last_match_date[away_team]).days
                df.at[idx, "away_rest_days"] = away_rest_days
                df.at[idx, "away_previous_match_id"] = last_match_id[away_team]
            else:
                df.at[idx, "away_rest_days"] = None
                df.at[idx, "away_previous_match_id"] = None

            # Update last match date for both teams
            last_match_date[home_team] = match_date
            last_match_date[away_team] = match_date
            last_match_id[home_team] = match_id
            last_match_id[away_team] = match_id

        # Calculate rest days differential (positive means home team had more rest)
        df["rest_days_diff"] = df["home_rest_days"] - df["away_rest_days"]

        # Fill NaN values with median rest days
        home_median = df["home_rest_days"].median()
        away_median = df["away_rest_days"].median()

        df["home_rest_days"] = df["home_rest_days"].fillna(home_median)
        df["away_rest_days"] = df["away_rest_days"].fillna(away_median)
        df["rest_days_diff"] = df["rest_days_diff"].fillna(0)

        # Create categorical features for rest
        df["home_rest_category"] = pd.cut(
            df["home_rest_days"],
            bins=[-1, 2, 5, 10, 100],
            labels=["short", "normal", "long", "very_long"],
        )

        df["away_rest_category"] = pd.cut(
            df["away_rest_days"],
            bins=[-1, 2, 5, 10, 100],
            labels=["short", "normal", "long", "very_long"],
        )

        logger.info("Added team rest features")
        return df

    def add_fixture_congestion_features(
        self, match_df: pd.DataFrame, window_days: int = 30
    ) -> pd.DataFrame:
        """
        Add features related to fixture congestion.

        Args:
            match_df: DataFrame with match data
            window_days: Number of days to look at for fixture congestion

        Returns:
            DataFrame with fixture congestion features added
        """
        if match_df.empty or "date" not in match_df.columns:
            logger.warning(
                "Empty DataFrame or missing date column provided for fixture congestion features"
            )
            return match_df

        # Create a copy to avoid modifying the original
        df = match_df.copy()

        # Sort by date
        df = df.sort_values("date")

        # Process each match
        for idx, match in df.iterrows():
            home_team = match["home_team"]
            away_team = match["away_team"]
            match_date = match["date"]

            # Calculate previous matches in the window for home team
            home_previous_matches = df[
                (df["date"] < match_date)
                & (df["date"] >= match_date - pd.Timedelta(days=window_days))
                & ((df["home_team"] == home_team) | (df["away_team"] == home_team))
            ]

            df.at[idx, "home_matches_last_30d"] = len(home_previous_matches)

            # Calculate previous matches in the window for away team
            away_previous_matches = df[
                (df["date"] < match_date)
                & (df["date"] >= match_date - pd.Timedelta(days=window_days))
                & ((df["home_team"] == away_team) | (df["away_team"] == away_team))
            ]

            df.at[idx, "away_matches_last_30d"] = len(away_previous_matches)

            # Calculate future matches in the window for home team
            home_future_matches = df[
                (df["date"] > match_date)
                & (df["date"] <= match_date + pd.Timedelta(days=window_days))
                & ((df["home_team"] == home_team) | (df["away_team"] == home_team))
            ]

            df.at[idx, "home_matches_next_30d"] = len(home_future_matches)

            # Calculate future matches in the window for away team
            away_future_matches = df[
                (df["date"] > match_date)
                & (df["date"] <= match_date + pd.Timedelta(days=window_days))
                & ((df["home_team"] == away_team) | (df["away_team"] == away_team))
            ]

            df.at[idx, "away_matches_next_30d"] = len(away_future_matches)

        # Calculate congestion differential
        df["congestion_diff_last_30d"] = (
            df["home_matches_last_30d"] - df["away_matches_last_30d"]
        )
        df["congestion_diff_next_30d"] = (
            df["home_matches_next_30d"] - df["away_matches_next_30d"]
        )

        logger.info("Added fixture congestion features")
        return df

    def add_matchday_features(self, match_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features related to the matchday number within a season.

        Args:
            match_df: DataFrame with match data

        Returns:
            DataFrame with matchday features added
        """
        if match_df.empty or "date" not in match_df.columns:
            logger.warning(
                "Empty DataFrame or missing date column provided for matchday features"
            )
            return match_df

        # Create a copy to avoid modifying the original
        df = match_df.copy()

        # Group by season and competition
        grouped = df.groupby(["season", "competition"])

        for (season, competition), group in grouped:
            # Sort by date
            group = group.sort_values("date")

            # Assign matchday numbers
            matchdays = {}
            current_matchday = 0
            current_date = None

            for idx, match in group.iterrows():
                match_date = match["date"].date()  # Use date only, not time

                # If this is a new date, increment matchday
                if match_date != current_date:
                    current_matchday += 1
                    current_date = match_date

                # Assign matchday number
                df.at[idx, "matchday"] = current_matchday

            # Calculate total matchdays
            total_matchdays = current_matchday

            # Calculate normalized matchday (0-1 scale)
            for idx in group.index:
                matchday = df.at[idx, "matchday"]
                df.at[idx, "normalized_matchday"] = matchday / total_matchdays

        # Fill NaN values
        df["matchday"] = df["matchday"].fillna(0).astype(int)
        df["normalized_matchday"] = df["normalized_matchday"].fillna(0)

        logger.info("Added matchday features")
        return df

    def add_performance_trend_features(
        self, match_df: pd.DataFrame, window_sizes: List[int] = [3, 5, 10]
    ) -> pd.DataFrame:
        """
        Add features related to team performance trends over time.

        Args:
            match_df: DataFrame with match data
            window_sizes: List of window sizes for rolling trends

        Returns:
            DataFrame with performance trend features added
        """
        if match_df.empty or "date" not in match_df.columns:
            logger.warning(
                "Empty DataFrame or missing date column provided for performance trend features"
            )
            return match_df

        # Create a copy to avoid modifying the original
        df = match_df.copy()

        # Create a team results dataframe
        team_results = []

        for idx, match in df.iterrows():
            # Home team result
            home_result = {
                "team": match["home_team"],
                "date": match["date"],
                "opponent": match["away_team"],
                "is_home": True,
                "goals_scored": match["home_score"],
                "goals_conceded": match["away_score"],
                "match_id": match["match_id"],
            }

            # Determine result from team's perspective
            if match["home_score"] > match["away_score"]:
                home_result["result"] = "W"
                home_result["points"] = 3
            elif match["home_score"] == match["away_score"]:
                home_result["result"] = "D"
                home_result["points"] = 1
            else:
                home_result["result"] = "L"
                home_result["points"] = 0

            team_results.append(home_result)

            # Away team result
            away_result = {
                "team": match["away_team"],
                "date": match["date"],
                "opponent": match["home_team"],
                "is_home": False,
                "goals_scored": match["away_score"],
                "goals_conceded": match["home_score"],
                "match_id": match["match_id"],
            }

            # Determine result from team's perspective
            if match["away_score"] > match["home_score"]:
                away_result["result"] = "W"
                away_result["points"] = 3
            elif match["away_score"] == match["home_score"]:
                away_result["result"] = "D"
                away_result["points"] = 1
            else:
                away_result["result"] = "L"
                away_result["points"] = 0

            team_results.append(away_result)

        # Convert to DataFrame and sort by team and date
        team_df = pd.DataFrame(team_results)
        team_df = team_df.sort_values(["team", "date"])

        # Calculate trend features for each team
        for team in team_df["team"].unique():
            team_mask = team_df["team"] == team

            # Calculate streaks
            team_df.loc[team_mask, "result_numeric"] = team_df.loc[
                team_mask, "result"
            ].map({"W": 1, "D": 0, "L": -1})
            team_df.loc[team_mask, "win_streak"] = (
                team_df.loc[team_mask, "result"] == "W"
            ).astype(int)
            team_df.loc[team_mask, "loss_streak"] = (
                team_df.loc[team_mask, "result"] == "L"
            ).astype(int)

            # Calculate cumulative streaks
            win_streaks = team_df.loc[team_mask, "win_streak"].values
            loss_streaks = team_df.loc[team_mask, "loss_streak"].values

            cum_win_streak = np.zeros_like(win_streaks)
            cum_loss_streak = np.zeros_like(loss_streaks)

            for i in range(len(win_streaks)):
                if win_streaks[i] == 1:
                    cum_win_streak[i] = cum_win_streak[i - 1] + 1 if i > 0 else 1
                else:
                    cum_win_streak[i] = 0

                if loss_streaks[i] == 1:
                    cum_loss_streak[i] = cum_loss_streak[i - 1] + 1 if i > 0 else 1
                else:
                    cum_loss_streak[i] = 0

            team_df.loc[team_mask, "current_win_streak"] = cum_win_streak
            team_df.loc[team_mask, "current_loss_streak"] = cum_loss_streak

            # Calculate rolling metrics for each window size
            for window in window_sizes:
                if sum(team_mask) >= window:
                    # Rolling points
                    team_df.loc[team_mask, f"points_last_{window}"] = (
                        team_df.loc[team_mask, "points"]
                        .rolling(window, min_periods=1)
                        .sum()
                    )

                    # Rolling goal metrics
                    team_df.loc[team_mask, f"goals_scored_last_{window}"] = (
                        team_df.loc[team_mask, "goals_scored"]
                        .rolling(window, min_periods=1)
                        .sum()
                    )
                    team_df.loc[team_mask, f"goals_conceded_last_{window}"] = (
                        team_df.loc[team_mask, "goals_conceded"]
                        .rolling(window, min_periods=1)
                        .sum()
                    )
                    team_df.loc[team_mask, f"goal_diff_last_{window}"] = (
                        team_df.loc[team_mask, f"goals_scored_last_{window}"]
                        - team_df.loc[team_mask, f"goals_conceded_last_{window}"]
                    )

                    # Rolling win percentage
                    team_df.loc[team_mask, f"win_rate_last_{window}"] = (
                        team_df.loc[team_mask, "win_streak"]
                        .rolling(window, min_periods=1)
                        .mean()
                        * 100
                    )

                    # Rolling performance trend (slope of points)
                    points = (
                        team_df.loc[team_mask, "points"]
                        .rolling(window, min_periods=window)
                        .apply(
                            lambda x: (
                                np.polyfit(np.arange(len(x)), x, 1)[0]
                                if len(x) == window
                                else np.nan
                            )
                        )
                    )
                    team_df.loc[team_mask, f"points_trend_{window}"] = points

                    # Rolling goal difference trend
                    goal_diff = (
                        (
                            team_df.loc[team_mask, "goals_scored"]
                            - team_df.loc[team_mask, "goals_conceded"]
                        )
                        .rolling(window, min_periods=window)
                        .apply(
                            lambda x: (
                                np.polyfit(np.arange(len(x)), x, 1)[0]
                                if len(x) == window
                                else np.nan
                            )
                        )
                    )
                    team_df.loc[team_mask, f"goal_diff_trend_{window}"] = goal_diff

        # Add features back to the original dataframe
        for idx, match in df.iterrows():
            match_id = match["match_id"]

            # Get home team's previous match stats
            home_team = match["home_team"]
            home_team_prev_matches = team_df[
                (team_df["team"] == home_team) & (team_df["date"] < match["date"])
            ].sort_values("date")

            if not home_team_prev_matches.empty:
                last_match = home_team_prev_matches.iloc[-1]

                # Add streak features
                df.at[idx, "home_current_win_streak"] = last_match["current_win_streak"]
                df.at[idx, "home_current_loss_streak"] = last_match[
                    "current_loss_streak"
                ]

                # Add window-based features
                for window in window_sizes:
                    if f"points_last_{window}" in last_match:
                        df.at[idx, f"home_points_last_{window}"] = last_match[
                            f"points_last_{window}"
                        ]
                        df.at[idx, f"home_goals_scored_last_{window}"] = last_match[
                            f"goals_scored_last_{window}"
                        ]
                        df.at[idx, f"home_goals_conceded_last_{window}"] = last_match[
                            f"goals_conceded_last_{window}"
                        ]
                        df.at[idx, f"home_goal_diff_last_{window}"] = last_match[
                            f"goal_diff_last_{window}"
                        ]
                        df.at[idx, f"home_win_rate_last_{window}"] = last_match[
                            f"win_rate_last_{window}"
                        ]

                    if f"points_trend_{window}" in last_match:
                        df.at[idx, f"home_points_trend_{window}"] = last_match[
                            f"points_trend_{window}"
                        ]
                        df.at[idx, f"home_goal_diff_trend_{window}"] = last_match[
                            f"goal_diff_trend_{window}"
                        ]

            # Get away team's previous match stats
            away_team = match["away_team"]
            away_team_prev_matches = team_df[
                (team_df["team"] == away_team) & (team_df["date"] < match["date"])
            ].sort_values("date")

            if not away_team_prev_matches.empty:
                last_match = away_team_prev_matches.iloc[-1]

                # Add streak features
                df.at[idx, "away_current_win_streak"] = last_match["current_win_streak"]
                df.at[idx, "away_current_loss_streak"] = last_match[
                    "current_loss_streak"
                ]

                # Add window-based features
                for window in window_sizes:
                    if f"points_last_{window}" in last_match:
                        df.at[idx, f"away_points_last_{window}"] = last_match[
                            f"points_last_{window}"
                        ]
                        df.at[idx, f"away_goals_scored_last_{window}"] = last_match[
                            f"goals_scored_last_{window}"
                        ]
                        df.at[idx, f"away_goals_conceded_last_{window}"] = last_match[
                            f"goals_conceded_last_{window}"
                        ]
                        df.at[idx, f"away_goal_diff_last_{window}"] = last_match[
                            f"goal_diff_last_{window}"
                        ]
                        df.at[idx, f"away_win_rate_last_{window}"] = last_match[
                            f"win_rate_last_{window}"
                        ]

                    if f"points_trend_{window}" in last_match:
                        df.at[idx, f"away_points_trend_{window}"] = last_match[
                            f"points_trend_{window}"
                        ]
                        df.at[idx, f"away_goal_diff_trend_{window}"] = last_match[
                            f"goal_diff_trend_{window}"
                        ]

        # Fill NaN values
        trend_cols = [
            col
            for col in df.columns
            if "streak" in col or "last" in col or "trend" in col
        ]
        df[trend_cols] = df[trend_cols].fillna(0)

        logger.info("Added performance trend features")
        return df

    def extract_features(self, matches: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Extract all temporal features from match data.

        Args:
            matches: List of match data dictionaries

        Returns:
            DataFrame with temporal features
        """
        if not matches:
            logger.warning("No matches provided for feature extraction")
            return pd.DataFrame()

        logger.info(f"Extracting temporal features from {len(matches)} matches")

        # Prepare match dataframe
        match_df = self.prepare_match_dataframe(matches)

        # Add calendar features
        match_df = self.add_calendar_features(match_df)

        # Add season phase features
        match_df = self.add_season_phase_features(match_df)

        # Add team rest features
        match_df = self.add_team_rest_features(match_df)

        # Add fixture congestion features
        match_df = self.add_fixture_congestion_features(match_df)

        # Add matchday features
        match_df = self.add_matchday_features(match_df)

        # Add performance trend features
        match_df = self.add_performance_trend_features(match_df)

        logger.info(
            f"Extracted temporal features: {len(match_df)} matches with {len(match_df.columns)} features"
        )
        return match_df
