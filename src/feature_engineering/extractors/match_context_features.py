# src/feature_engineering/extractors/match_context_features.py
"""
Extract match context features from match data.

This module provides functionality for creating features related to the
context of a match, such as:
- Derby matches (matches between local rivals)
- Time of season (early, mid, late season)
- Historical matchup statistics
- Competition importance
- Rest days between matches
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MatchContextFeatures:
    """Extract and create match context features."""

    def __init__(
        self,
        derby_pairs: Optional[List[Tuple[str, str]]] = None,
        season_stages: Dict[float, str] = {0.0: "early", 0.33: "mid", 0.67: "late"},
    ):
        """
        Initialize the match context feature extractor.

        Args:
            derby_pairs: List of team pairs that are considered derbies (local rivals)
            season_stages: Dictionary mapping proportion of season completed to stage name
        """
        self.derby_pairs = derby_pairs or []
        self.season_stages = season_stages

        # Ensure derby pairs are symmetric (if A vs B is a derby, B vs A is also a derby)
        self._symmetrize_derby_pairs()

        logger.info(
            f"Initialized MatchContextFeatures with {len(self.derby_pairs)} derby pairs"
        )

    def _symmetrize_derby_pairs(self):
        """Ensure derby pairs are symmetric."""
        symmetric_pairs = set()

        for team1, team2 in self.derby_pairs:
            symmetric_pairs.add((team1, team2))
            symmetric_pairs.add((team2, team1))

        self.derby_pairs = list(symmetric_pairs)

    def prepare_match_dataframe(self, matches: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Prepare a DataFrame from match data for feature extraction.

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
                date_value = match.get("date")
                match_dict = {
                    "match_id": match.get("match_id"),
                    "date": pd.to_datetime(date_value) if date_value is not None else pd.NaT,
                    "competition": match.get("competition"),
                    "season": match.get("season"),
                    "stage": match.get("stage", "Regular Season"),
                    "home_team": match.get("home_team"),
                    "away_team": match.get("away_team"),
                    "home_score": match.get("home_score"),
                    "away_score": match.get("away_score"),
                    "result": match.get("result"),
                    "total_goals": match.get("total_goals"),
                    "both_teams_scored": match.get("both_teams_scored", False),
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

    def add_derby_feature(self, match_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a feature indicating whether a match is a derby.

        Args:
            match_df: DataFrame with match data

        Returns:
            DataFrame with derby feature added
        """
        if match_df.empty:
            logger.warning("Empty DataFrame provided for derby feature")
            return pd.DataFrame()

        # Create a copy to avoid modifying the original
        df = match_df.copy()

        # Create a set of derby pairs for faster lookup
        derby_set = set([(team1, team2) for team1, team2 in self.derby_pairs])

        # Add derby feature
        df["is_derby"] = False

        for idx, row in df.iterrows():
            home_team = row["home_team"]
            away_team = row["away_team"]

            if (home_team, away_team) in derby_set:
                df.at[idx, "is_derby"] = True

        logger.info(
            f"Added derby feature: {df['is_derby'].sum()} matches identified as derbies"
        )
        return df

    def add_season_stage_feature(self, match_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a feature indicating the stage of the season.

        Args:
            match_df: DataFrame with match data

        Returns:
            DataFrame with season stage feature added
        """
        if match_df.empty:
            logger.warning("Empty DataFrame provided for season stage feature")
            return pd.DataFrame()

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

                # Determine the stage based on the proportion
                stage = "early"
                for threshold, stage_name in sorted(self.season_stages.items()):
                    if proportion >= threshold:
                        stage = stage_name

                df.at[idx, "season_proportion"] = round(proportion, 3)
                df.at[idx, "season_stage"] = stage

        # Fill any missing values
        df["season_proportion"] = df["season_proportion"].fillna(0)
        df["season_stage"] = df["season_stage"].fillna("unknown")

        logger.info("Added season stage features")
        return df

    def add_historical_matchup_features(
        self, match_df: pd.DataFrame, lookback_matches: int = 5
    ) -> pd.DataFrame:
        """
        Add features based on historical matchups between the two teams.

        Args:
            match_df: DataFrame with match data
            lookback_matches: Number of previous matchups to consider

        Returns:
            DataFrame with historical matchup features added
        """
        if match_df.empty:
            logger.warning("Empty DataFrame provided for historical matchup features")
            return pd.DataFrame()

        # Create a copy to avoid modifying the original
        df = match_df.copy()

        # Sort by date to ensure chronological processing
        df = df.sort_values("date")

        # Initialize matchup history dictionary
        # Key: (team1, team2) tuple, Value: list of match results from team1's perspective
        matchup_history = {}

        # Process each match
        for idx, match in df.iterrows():
            home_team = match["home_team"]
            away_team = match["away_team"]
            date = match["date"]

            # Get previous matches between these teams
            previous_matches = df[
                (
                    ((df["home_team"] == home_team) & (df["away_team"] == away_team))
                    | ((df["home_team"] == away_team) & (df["away_team"] == home_team))
                )
                & (df["date"] < date)
            ].sort_values("date", ascending=False)

            # Initialize features
            df.at[idx, "h2h_played"] = len(previous_matches)

            if previous_matches.empty:
                continue

            # Calculate home team's performance in previous matchups
            home_wins = 0
            home_draws = 0
            home_losses = 0
            home_goals_scored = 0
            home_goals_conceded = 0

            # Limit to lookback matches
            for _, prev_match in previous_matches.head(lookback_matches).iterrows():
                prev_home = prev_match["home_team"]
                prev_away = prev_match["away_team"]

                if prev_home == home_team:
                    # Home team was also home in the previous match
                    home_score = prev_match["home_score"]
                    away_score = prev_match["away_score"]

                    if home_score > away_score:
                        home_wins += 1
                    elif home_score == away_score:
                        home_draws += 1
                    else:
                        home_losses += 1

                    home_goals_scored += home_score
                    home_goals_conceded += away_score
                else:
                    # Home team was away in the previous match
                    home_score = prev_match["away_score"]
                    away_score = prev_match["home_score"]

                    if home_score > away_score:
                        home_wins += 1
                    elif home_score == away_score:
                        home_draws += 1
                    else:
                        home_losses += 1

                    home_goals_scored += home_score
                    home_goals_conceded += away_score

            # Calculate features
            h2h_matches = min(len(previous_matches), lookback_matches)

            df.at[idx, f"h2h_last_{lookback_matches}_played"] = h2h_matches
            df.at[idx, f"h2h_last_{lookback_matches}_home_wins"] = home_wins
            df.at[idx, f"h2h_last_{lookback_matches}_home_draws"] = home_draws
            df.at[idx, f"h2h_last_{lookback_matches}_home_losses"] = home_losses

            if h2h_matches > 0:
                df.at[idx, f"h2h_last_{lookback_matches}_home_win_rate"] = round(
                    home_wins / h2h_matches * 100, 1
                )
                df.at[idx, f"h2h_last_{lookback_matches}_home_points_per_game"] = round(
                    (home_wins * 3 + home_draws) / h2h_matches, 2
                )
                df.at[
                    idx, f"h2h_last_{lookback_matches}_home_goals_scored_per_game"
                ] = round(home_goals_scored / h2h_matches, 2)
                df.at[
                    idx, f"h2h_last_{lookback_matches}_home_goals_conceded_per_game"
                ] = round(home_goals_conceded / h2h_matches, 2)
                df.at[idx, f"h2h_last_{lookback_matches}_home_goals_diff_per_game"] = (
                    round((home_goals_scored - home_goals_conceded) / h2h_matches, 2)
                )

        # Fill NaN values
        h2h_columns = [col for col in df.columns if col.startswith("h2h_")]
        df[h2h_columns] = df[h2h_columns].fillna(0)

        logger.info(
            f"Added historical matchup features with lookback of {lookback_matches} matches"
        )
        return df

    def add_competition_importance(
        self,
        match_df: pd.DataFrame,
        competition_importance: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Add feature indicating the importance of each competition.

        Args:
            match_df: DataFrame with match data
            competition_importance: Dictionary mapping competition names to importance scores (0-1)

        Returns:
            DataFrame with competition importance feature added
        """
        if match_df.empty:
            logger.warning(
                "Empty DataFrame provided for competition importance feature"
            )
            return pd.DataFrame()

        # Create a copy to avoid modifying the original
        df = match_df.copy()

        # Default importance by stage if not provided
        if competition_importance is None:
            # Create a simple default mapping
            unique_competitions = df["competition"].unique()
            competition_importance = {}

            for comp in unique_competitions:
                # Assign higher importance to competitions with "Champions" or similar words
                if any(
                    keyword in comp.lower()
                    for keyword in ["champion", "copa", "cup", "super"]
                ):
                    competition_importance[comp] = 0.9
                else:
                    competition_importance[comp] = 0.7

        # Add importance scores
        df["competition_importance"] = df["competition"].map(competition_importance)

        # Add stage importance modifier
        stage_importance = {
            "Regular Season": 0.0,
            "Group Stage": 0.0,
            "Round of 32": 0.1,
            "Round of 16": 0.2,
            "Quarter-finals": 0.3,
            "Semi-finals": 0.4,
            "Final": 0.5,
        }

        df["stage_importance_modifier"] = df["stage"].map(stage_importance).fillna(0.0)

        # Calculate total match importance
        df["match_importance"] = (
            df["competition_importance"] + df["stage_importance_modifier"]
        ).clip(0, 1)

        # Handle season stage importance (matches become more important later in the season)
        if "season_proportion" in df.columns:
            # Linear increase in importance as the season progresses
            season_importance = df["season_proportion"].clip(0, 1) * 0.3

            # Adjust match importance based on season stage
            df["match_importance"] = (df["match_importance"] + season_importance).clip(
                0, 1
            )

        logger.info("Added competition and match importance features")
        return df

    def add_rest_days_feature(self, match_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add feature indicating the number of days since each team's last match.

        Args:
            match_df: DataFrame with match data

        Returns:
            DataFrame with rest days feature added
        """
        if match_df.empty:
            logger.warning("Empty DataFrame provided for rest days feature")
            return pd.DataFrame()

        # Create a copy to avoid modifying the original
        df = match_df.copy()

        # Sort by date
        df = df.sort_values("date")

        # Dictionary to store the last match date for each team
        last_match_date = {}

        # Process each match
        for idx, match in df.iterrows():
            home_team = match["home_team"]
            away_team = match["away_team"]
            match_date = match["date"]

            # Calculate rest days for home team
            if home_team in last_match_date:
                home_rest_days = (match_date - last_match_date[home_team]).days
                df.at[idx, "home_rest_days"] = home_rest_days
            else:
                df.at[idx, "home_rest_days"] = None

            # Calculate rest days for away team
            if away_team in last_match_date:
                away_rest_days = (match_date - last_match_date[away_team]).days
                df.at[idx, "away_rest_days"] = away_rest_days
            else:
                df.at[idx, "away_rest_days"] = None

            # Update last match date for both teams
            last_match_date[home_team] = match_date
            last_match_date[away_team] = match_date

        # Calculate rest days differential (positive means home team had more rest)
        df["rest_days_diff"] = df["home_rest_days"] - df["away_rest_days"]

        # Fill NaN values with median rest days
        home_median = df["home_rest_days"].median()
        away_median = df["away_rest_days"].median()

        df["home_rest_days"] = df["home_rest_days"].fillna(home_median)
        df["away_rest_days"] = df["away_rest_days"].fillna(away_median)
        df["rest_days_diff"] = df["rest_days_diff"].fillna(0)

        logger.info("Added rest days features")
        return df

    def extract_features(self, matches: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Extract all match context features.

        Args:
            matches: List of match data dictionaries

        Returns:
            DataFrame with match context features
        """
        if not matches:
            logger.warning("No matches provided for feature extraction")
            return pd.DataFrame()

        logger.info(f"Extracting match context features from {len(matches)} matches")

        # Prepare match dataframe
        match_df = self.prepare_match_dataframe(matches)

        # Add derby feature
        match_df = self.add_derby_feature(match_df)

        # Add season stage feature
        match_df = self.add_season_stage_feature(match_df)

        # Add historical matchup features
        match_df = self.add_historical_matchup_features(match_df)

        # Add competition importance
        match_df = self.add_competition_importance(match_df)

        # Add rest days feature
        match_df = self.add_rest_days_feature(match_df)

        logger.info(
            f"Extracted match context features: {len(match_df)} matches with {len(match_df.columns)} features"
        )
        return match_df

    def detect_derbies(
        self,
        matches: List[Dict[str, Any]],
        min_matches: int = 10,
        threshold: float = 0.8,
    ) -> List[Tuple[str, str]]:
        """
        Automatically detect derby pairs based on matchup frequency and intensity.

        Args:
            matches: List of match data dictionaries
            min_matches: Minimum number of matches between teams to consider as potential derby
            threshold: Threshold for derby detection (higher means more selective)

        Returns:
            List of detected derby pairs
        """
        if not matches:
            logger.warning("No matches provided for derby detection")
            return []

        # Prepare match dataframe
        match_df = self.prepare_match_dataframe(matches)

        # Count matches between each pair of teams
        team_pairs = {}

        for _, match in match_df.iterrows():
            home_team = match["home_team"]
            away_team = match["away_team"]

            # Order teams alphabetically to ensure consistent keys
            pair = tuple(sorted([home_team, away_team]))

            if pair not in team_pairs:
                team_pairs[pair] = {
                    "count": 0,
                    "yellow_cards": 0,
                    "red_cards": 0,
                    "fouls": 0,
                    "goals": 0,
                }

            # Increment count
            team_pairs[pair]["count"] += 1

            # Add match intensity metrics if available
            if "full_time_stats" in match and isinstance(
                match["full_time_stats"], dict
            ):
                stats = match["full_time_stats"]

                if "yellow_cards" in stats:
                    team_pairs[pair]["yellow_cards"] += stats["yellow_cards"].get(
                        "home", 0
                    ) + stats["yellow_cards"].get("away", 0)

                if "red_cards" in stats:
                    team_pairs[pair]["red_cards"] += stats["red_cards"].get(
                        "home", 0
                    ) + stats["red_cards"].get("away", 0)

                if "fouls" in stats:
                    team_pairs[pair]["fouls"] += stats["fouls"].get("home", 0) + stats[
                        "fouls"
                    ].get("away", 0)

            # Add goals
            team_pairs[pair]["goals"] += match["home_score"] + match["away_score"]

        # Calculate intensity metrics for each pair
        for pair, stats in team_pairs.items():
            if stats["count"] >= min_matches:
                # Calculate per-match averages
                stats["yellow_per_match"] = stats["yellow_cards"] / stats["count"]
                stats["red_per_match"] = stats["red_cards"] / stats["count"]
                stats["fouls_per_match"] = stats["fouls"] / stats["count"]
                stats["goals_per_match"] = stats["goals"] / stats["count"]

                # Calculate an overall intensity score
                stats["intensity"] = (
                    (stats["yellow_per_match"] * 0.3)
                    + (stats["red_per_match"] * 0.4)
                    + (stats["fouls_per_match"] * 0.2)
                    + (stats["goals_per_match"] * 0.1)
                )

        # Filter pairs by match count and intensity
        filtered_pairs = {
            pair: stats
            for pair, stats in team_pairs.items()
            if stats["count"] >= min_matches
        }

        if not filtered_pairs:
            logger.warning(f"No team pairs with at least {min_matches} matches found")
            return []

        # Normalize intensity scores
        intensity_scores = [stats["intensity"] for stats in filtered_pairs.values()]
        max_intensity = max(intensity_scores)
        min_intensity = min(intensity_scores)

        intensity_range = max_intensity - min_intensity

        for pair, stats in filtered_pairs.items():
            if intensity_range > 0:
                stats["normalized_intensity"] = (
                    stats["intensity"] - min_intensity
                ) / intensity_range
            else:
                stats["normalized_intensity"] = 0.5

        # Detect derbies
        detected_derbies = []

        for pair, stats in filtered_pairs.items():
            if stats["normalized_intensity"] >= threshold:
                # Add both directions of the derby
                team1, team2 = pair
                detected_derbies.append((team1, team2))
                detected_derbies.append((team2, team1))

                logger.info(
                    f"Detected derby: {team1} vs {team2} (intensity: {stats['normalized_intensity']:.2f})"
                )

        logger.info(f"Detected {len(detected_derbies)//2} derby pairs")
        return detected_derbies
