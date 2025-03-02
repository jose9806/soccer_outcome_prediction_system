"""
Feature engineering module for soccer match prediction.

This module provides functionality to extract meaningful features from raw match data,
including historical team performance metrics, form indicators, and match context features.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any


class FeatureEngineer:
    """Extracts predictive features from soccer match data."""

    def __init__(self, matches_df: pd.DataFrame):
        """
        Initialize the feature engineer with historical match data.

        Args:
            matches_df: DataFrame containing all historical matches
        """
        self.matches_df = matches_df.copy()

        # Convert date to datetime if it's a string
        if (
            "date" in self.matches_df.columns
            and not pd.api.types.is_datetime64_any_dtype(self.matches_df["date"])
        ):
            self.matches_df["date"] = pd.to_datetime(self.matches_df["date"])

        # Sort by date
        self.matches_df = self.matches_df.sort_values("date")

        # Create unique identifiers for teams if not present
        if "home_team_id" not in self.matches_df.columns:
            self._create_team_ids()

    def _create_team_ids(self):
        """Create unique identifiers for teams."""
        # Get all unique team names
        home_teams = self.matches_df["home_team"].unique()
        away_teams = self.matches_df["away_team"].unique()
        all_teams = np.unique(np.concatenate([home_teams, away_teams]))

        # Create team ID mapping
        team_id_map = {team: idx for idx, team in enumerate(all_teams)}

        # Add team IDs to the DataFrame
        self.matches_df["home_team_id"] = self.matches_df["home_team"].map(team_id_map)
        self.matches_df["away_team_id"] = self.matches_df["away_team"].map(team_id_map)

    def extract_match_features(
        self,
        home_team: str,
        away_team: str,
        match_date: datetime,
        season: str,
        stage: str = "Regular Season",
    ) -> Dict[str, Any]:
        """
        Extract features for a specific match.

        Args:
            home_team: Name of the home team
            away_team: Name of the away team
            match_date: Date of the match
            season: Season identifier
            stage: Tournament stage

        Returns:
            Dictionary containing all match features
        """
        # Create base feature dictionary
        features = {
            "match_date": match_date,
            "home_team": home_team,
            "away_team": away_team,
            "season": season,
            "stage": stage,
        }

        # Filter matches before the current match date
        past_matches = self.matches_df[self.matches_df["date"] < match_date].copy()

        # Extract historical performance features
        features.update(self._extract_team_stats(past_matches, home_team, away_team))

        # Extract form features (last N matches)
        features.update(self._extract_form_features(past_matches, home_team, away_team))

        # Extract head-to-head features
        features.update(self._extract_h2h_features(past_matches, home_team, away_team))

        # Extract season performance features
        features.update(
            self._extract_season_stats(past_matches, home_team, away_team, season)
        )

        # Extract competition stage features
        features.update(
            self._extract_stage_features(past_matches, home_team, away_team, stage)
        )

        return features

    def _extract_team_stats(
        self, past_matches: pd.DataFrame, home_team: str, away_team: str
    ) -> Dict[str, float]:
        """
        Extract historical team performance statistics.

        Args:
            past_matches: DataFrame with past matches
            home_team: Home team name
            away_team: Away team name

        Returns:
            Dictionary with team statistics features
        """
        features = {}

        # Home team statistics
        home_stats = self._calculate_team_stats(past_matches, home_team)
        for key, value in home_stats.items():
            features[f"home_{key}"] = value

        # Away team statistics
        away_stats = self._calculate_team_stats(past_matches, away_team)
        for key, value in away_stats.items():
            features[f"away_{key}"] = value

        return features

    def _calculate_team_stats(
        self, past_matches: pd.DataFrame, team: str
    ) -> Dict[str, float]:
        """
        Calculate performance statistics for a specific team.

        Args:
            past_matches: DataFrame with past matches
            team: Team name

        Returns:
            Dictionary with team statistics
        """
        # Filter matches for the team
        team_home_matches = past_matches[past_matches["home_team"] == team]
        team_away_matches = past_matches[past_matches["away_team"] == team]

        # Count total matches
        total_matches = len(team_home_matches) + len(team_away_matches)

        if total_matches == 0:
            # If no past matches, return default values
            return {
                "total_matches": 0,
                "win_rate": 0.0,
                "draw_rate": 0.0,
                "loss_rate": 0.0,
                "goals_scored_avg": 0.0,
                "goals_conceded_avg": 0.0,
                "clean_sheet_rate": 0.0,
                "failed_to_score_rate": 0.0,
                "points_per_game": 0.0,
                "home_win_rate": 0.0,
                "away_win_rate": 0.0,
            }

        # Calculate wins, draws, losses
        home_wins = len(
            team_home_matches[
                team_home_matches["home_score"] > team_home_matches["away_score"]
            ]
        )
        away_wins = len(
            team_away_matches[
                team_away_matches["away_score"] > team_away_matches["home_score"]
            ]
        )
        home_draws = len(
            team_home_matches[
                team_home_matches["home_score"] == team_home_matches["away_score"]
            ]
        )
        away_draws = len(
            team_away_matches[
                team_away_matches["away_score"] == team_away_matches["home_score"]
            ]
        )
        home_losses = len(
            team_home_matches[
                team_home_matches["home_score"] < team_home_matches["away_score"]
            ]
        )
        away_losses = len(
            team_away_matches[
                team_away_matches["away_score"] < team_away_matches["home_score"]
            ]
        )

        total_wins = home_wins + away_wins
        total_draws = home_draws + away_draws
        total_losses = home_losses + away_losses

        # Calculate goals
        home_goals_scored = team_home_matches["home_score"].sum()
        away_goals_scored = team_away_matches["away_score"].sum()
        home_goals_conceded = team_home_matches["away_score"].sum()
        away_goals_conceded = team_away_matches["home_score"].sum()

        total_goals_scored = home_goals_scored + away_goals_scored
        total_goals_conceded = home_goals_conceded + away_goals_conceded

        # Calculate clean sheets
        home_clean_sheets = len(team_home_matches[team_home_matches["away_score"] == 0])
        away_clean_sheets = len(team_away_matches[team_away_matches["home_score"] == 0])
        total_clean_sheets = home_clean_sheets + away_clean_sheets

        # Calculate failed to score
        home_failed_to_score = len(
            team_home_matches[team_home_matches["home_score"] == 0]
        )
        away_failed_to_score = len(
            team_away_matches[team_away_matches["away_score"] == 0]
        )
        total_failed_to_score = home_failed_to_score + away_failed_to_score

        # Calculate points
        total_points = total_wins * 3 + total_draws
        home_matches = len(team_home_matches)
        away_matches = len(team_away_matches)

        # Create statistics dictionary
        stats = {
            "total_matches": total_matches,
            "win_rate": total_wins / total_matches if total_matches > 0 else 0,
            "draw_rate": total_draws / total_matches if total_matches > 0 else 0,
            "loss_rate": total_losses / total_matches if total_matches > 0 else 0,
            "goals_scored_avg": (
                total_goals_scored / total_matches if total_matches > 0 else 0
            ),
            "goals_conceded_avg": (
                total_goals_conceded / total_matches if total_matches > 0 else 0
            ),
            "clean_sheet_rate": (
                total_clean_sheets / total_matches if total_matches > 0 else 0
            ),
            "failed_to_score_rate": (
                total_failed_to_score / total_matches if total_matches > 0 else 0
            ),
            "points_per_game": total_points / total_matches if total_matches > 0 else 0,
            "home_win_rate": home_wins / home_matches if home_matches > 0 else 0,
            "away_win_rate": away_wins / away_matches if away_matches > 0 else 0,
        }

        return stats

    def _extract_form_features(
        self,
        past_matches: pd.DataFrame,
        home_team: str,
        away_team: str,
        form_matches: int = 5,
    ) -> Dict[str, float]:
        """
        Extract recent form features for both teams.

        Args:
            past_matches: DataFrame with past matches
            home_team: Home team name
            away_team: Away team name
            form_matches: Number of recent matches to consider for form

        Returns:
            Dictionary with form features
        """
        features = {}

        # Home team form
        home_form = self._calculate_team_form(past_matches, home_team, form_matches)
        for key, value in home_form.items():
            features[f"home_form_{key}"] = value

        # Away team form
        away_form = self._calculate_team_form(past_matches, away_team, form_matches)
        for key, value in away_form.items():
            features[f"away_form_{key}"] = value

        return features

    def _calculate_team_form(
        self, past_matches: pd.DataFrame, team: str, form_matches: int = 5
    ) -> Dict[str, float]:
        """
        Calculate recent form statistics for a team.

        Args:
            past_matches: DataFrame with past matches
            team: Team name
            form_matches: Number of recent matches to consider

        Returns:
            Dictionary with form statistics
        """
        # Filter matches for the team
        team_home_matches = past_matches[past_matches["home_team"] == team].copy()
        team_away_matches = past_matches[past_matches["away_team"] == team].copy()

        # Combine and sort by date (most recent first)
        team_matches = pd.concat([team_home_matches, team_away_matches])
        team_matches = team_matches.sort_values("date", ascending=False)

        # Take only the most recent N matches
        recent_matches = team_matches.head(form_matches)

        # Count total recent matches (could be less than form_matches)
        total_matches = len(recent_matches)

        if total_matches == 0:
            # If no recent matches, return default values
            return {
                "total_matches": 0,
                "win_rate": 0.0,
                "draw_rate": 0.0,
                "loss_rate": 0.0,
                "goals_scored_avg": 0.0,
                "goals_conceded_avg": 0.0,
                "clean_sheet_rate": 0.0,
                "points_per_game": 0.0,
            }

        # Calculate results for recent matches
        wins = 0
        draws = 0
        losses = 0
        goals_scored = 0
        goals_conceded = 0
        clean_sheets = 0

        for _, match in recent_matches.iterrows():
            if match["home_team"] == team:
                # Team played at home
                if match["home_score"] > match["away_score"]:
                    wins += 1
                elif match["home_score"] == match["away_score"]:
                    draws += 1
                else:
                    losses += 1

                goals_scored += match["home_score"]
                goals_conceded += match["away_score"]

                if match["away_score"] == 0:
                    clean_sheets += 1
            else:
                # Team played away
                if match["away_score"] > match["home_score"]:
                    wins += 1
                elif match["away_score"] == match["home_score"]:
                    draws += 1
                else:
                    losses += 1

                goals_scored += match["away_score"]
                goals_conceded += match["home_score"]

                if match["home_score"] == 0:
                    clean_sheets += 1

        # Calculate points
        points = wins * 3 + draws

        # Create form statistics dictionary
        form_stats = {
            "matches": total_matches,
            "win_rate": wins / total_matches,
            "draw_rate": draws / total_matches,
            "loss_rate": losses / total_matches,
            "goals_scored_avg": goals_scored / total_matches,
            "goals_conceded_avg": goals_conceded / total_matches,
            "clean_sheet_rate": clean_sheets / total_matches,
            "points_per_game": points / total_matches,
        }

        return form_stats

    def _extract_h2h_features(
        self,
        past_matches: pd.DataFrame,
        home_team: str,
        away_team: str,
        h2h_matches: int = 10,
    ) -> Dict[str, float]:
        """
        Extract head-to-head features between the two teams.

        Args:
            past_matches: DataFrame with past matches
            home_team: Home team name
            away_team: Away team name
            h2h_matches: Maximum number of past head-to-head matches to consider

        Returns:
            Dictionary with head-to-head features
        """
        # Filter for matches between the two teams
        h2h_df = past_matches[
            (
                (past_matches["home_team"] == home_team)
                & (past_matches["away_team"] == away_team)
            )
            | (
                (past_matches["home_team"] == away_team)
                & (past_matches["away_team"] == home_team)
            )
        ].copy()

        # Sort by date (most recent first) and take the last N
        h2h_df = h2h_df.sort_values("date", ascending=False).head(h2h_matches)

        total_matches = len(h2h_df)

        if total_matches == 0:
            # If no past h2h matches, return default values
            return {
                "h2h_matches": 0,
                "h2h_home_win_rate": 0.0,
                "h2h_away_win_rate": 0.0,
                "h2h_draw_rate": 0.0,
                "h2h_home_goals_avg": 0.0,
                "h2h_away_goals_avg": 0.0,
                "h2h_total_goals_avg": 0.0,
                "h2h_btts_rate": 0.0,
            }

        # Calculate h2h statistics
        home_wins = 0
        away_wins = 0
        draws = 0
        home_goals = 0
        away_goals = 0
        btts_count = 0

        for _, match in h2h_df.iterrows():
            if match["home_team"] == home_team and match["away_team"] == away_team:
                # Direct match with same home/away setup
                if match["home_score"] > match["away_score"]:
                    home_wins += 1
                elif match["home_score"] < match["away_score"]:
                    away_wins += 1
                else:
                    draws += 1

                home_goals += match["home_score"]
                away_goals += match["away_score"]

                if match["home_score"] > 0 and match["away_score"] > 0:
                    btts_count += 1
            else:
                # Reversed match (current away team was home)
                if match["home_score"] > match["away_score"]:
                    away_wins += 1  # Current away team won
                elif match["home_score"] < match["away_score"]:
                    home_wins += 1  # Current home team won
                else:
                    draws += 1

                away_goals += match["home_score"]
                home_goals += match["away_score"]

                if match["home_score"] > 0 and match["away_score"] > 0:
                    btts_count += 1

        # Create h2h statistics dictionary
        h2h_stats = {
            "h2h_matches": total_matches,
            "h2h_home_win_rate": home_wins / total_matches,
            "h2h_away_win_rate": away_wins / total_matches,
            "h2h_draw_rate": draws / total_matches,
            "h2h_home_goals_avg": home_goals / total_matches,
            "h2h_away_goals_avg": away_goals / total_matches,
            "h2h_total_goals_avg": (home_goals + away_goals) / total_matches,
            "h2h_btts_rate": btts_count / total_matches,
        }

        return h2h_stats

    def _extract_season_stats(
        self, past_matches: pd.DataFrame, home_team: str, away_team: str, season: str
    ) -> Dict[str, float]:
        """
        Extract season-specific performance statistics.

        Args:
            past_matches: DataFrame with past matches
            home_team: Home team name
            away_team: Away team name
            season: Season identifier

        Returns:
            Dictionary with season-specific features
        """
        # Filter matches for the current season
        season_matches = past_matches[past_matches["season"] == season].copy()

        if len(season_matches) == 0:
            # If no matches in the current season yet, return default values
            return {
                "home_season_matches": 0,
                "away_season_matches": 0,
                "home_season_points": 0,
                "away_season_points": 0,
                "home_season_position": 0,
                "away_season_position": 0,
                "home_season_goals_scored_avg": 0.0,
                "away_season_goals_scored_avg": 0.0,
                "home_season_goals_conceded_avg": 0.0,
                "away_season_goals_conceded_avg": 0.0,
            }

        # Calculate season statistics using a league table
        league_table = self._calculate_league_table(season_matches)

        # Get team stats from the league table
        home_stats = league_table.get(
            home_team,
            {
                "matches": 0,
                "points": 0,
                "position": 0,
                "goals_scored_avg": 0.0,
                "goals_conceded_avg": 0.0,
            },
        )

        away_stats = league_table.get(
            away_team,
            {
                "matches": 0,
                "points": 0,
                "position": 0,
                "goals_scored_avg": 0.0,
                "goals_conceded_avg": 0.0,
            },
        )

        # Create season statistics features
        season_stats = {
            "home_season_matches": home_stats["matches"],
            "away_season_matches": away_stats["matches"],
            "home_season_points": home_stats["points"],
            "away_season_points": away_stats["points"],
            "home_season_position": home_stats["position"],
            "away_season_position": away_stats["position"],
            "home_season_goals_scored_avg": home_stats["goals_scored_avg"],
            "away_season_goals_scored_avg": away_stats["goals_scored_avg"],
            "home_season_goals_conceded_avg": home_stats["goals_conceded_avg"],
            "away_season_goals_conceded_avg": away_stats["goals_conceded_avg"],
        }

        return season_stats

    def _calculate_league_table(
        self, season_matches: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate a league table from the given matches.

        Args:
            season_matches: DataFrame with matches from a specific season

        Returns:
            Dictionary mapping team names to their statistics
        """
        # Initialize dictionary to store team statistics
        teams_stats = {}

        # Process each match to update team statistics
        for _, match in season_matches.iterrows():
            home_team = match["home_team"]
            away_team = match["away_team"]

            # Initialize team stats if not present
            if home_team not in teams_stats:
                teams_stats[home_team] = {
                    "matches": 0,
                    "wins": 0,
                    "draws": 0,
                    "losses": 0,
                    "goals_scored": 0,
                    "goals_conceded": 0,
                    "points": 0,
                }

            if away_team not in teams_stats:
                teams_stats[away_team] = {
                    "matches": 0,
                    "wins": 0,
                    "draws": 0,
                    "losses": 0,
                    "goals_scored": 0,
                    "goals_conceded": 0,
                    "points": 0,
                }

            # Update home team stats
            teams_stats[home_team]["matches"] += 1
            teams_stats[home_team]["goals_scored"] += match["home_score"]
            teams_stats[home_team]["goals_conceded"] += match["away_score"]

            # Update away team stats
            teams_stats[away_team]["matches"] += 1
            teams_stats[away_team]["goals_scored"] += match["away_score"]
            teams_stats[away_team]["goals_conceded"] += match["home_score"]

            # Update points and results
            if match["home_score"] > match["away_score"]:
                teams_stats[home_team]["wins"] += 1
                teams_stats[home_team]["points"] += 3
                teams_stats[away_team]["losses"] += 1
            elif match["home_score"] < match["away_score"]:
                teams_stats[away_team]["wins"] += 1
                teams_stats[away_team]["points"] += 3
                teams_stats[home_team]["losses"] += 1
            else:
                teams_stats[home_team]["draws"] += 1
                teams_stats[home_team]["points"] += 1
                teams_stats[away_team]["draws"] += 1
                teams_stats[away_team]["points"] += 1

        # Calculate averages and add them to team stats
        for team, stats in teams_stats.items():
            if stats["matches"] > 0:
                stats["goals_scored_avg"] = stats["goals_scored"] / stats["matches"]
                stats["goals_conceded_avg"] = stats["goals_conceded"] / stats["matches"]
            else:
                stats["goals_scored_avg"] = 0.0
                stats["goals_conceded_avg"] = 0.0

        # Sort teams by points to determine position
        sorted_teams = sorted(
            teams_stats.items(),
            key=lambda x: (
                x[1]["points"],
                x[1]["goals_scored"] - x[1]["goals_conceded"],
                x[1]["goals_scored"],
            ),
            reverse=True,
        )

        # Add position to team stats
        for position, (team, _) in enumerate(sorted_teams, 1):
            teams_stats[team]["position"] = position

        return teams_stats

    def _extract_stage_features(
        self, past_matches: pd.DataFrame, home_team: str, away_team: str, stage: str
    ) -> Dict[str, float]:
        """
        Extract features specific to the tournament stage.

        Args:
            past_matches: DataFrame with past matches
            home_team: Home team name
            away_team: Away team name
            stage: Tournament stage

        Returns:
            Dictionary with stage-specific features
        """
        # Filter matches for the specific stage
        stage_matches = past_matches[past_matches["stage"] == stage].copy()

        features = {}

        # Calculate home team stage performance
        home_stage_stats = self._calculate_stage_stats(stage_matches, home_team)
        for key, value in home_stage_stats.items():
            features[f"home_{key}"] = value

        # Calculate away team stage performance
        away_stage_stats = self._calculate_stage_stats(stage_matches, away_team)
        for key, value in away_stage_stats.items():
            features[f"away_{key}"] = value

        return features

    def _calculate_stage_stats(
        self, stage_matches: pd.DataFrame, team: str
    ) -> Dict[str, float]:
        """
        Calculate statistics for a specific tournament stage.

        Args:
            stage_matches: DataFrame with matches from a specific stage
            team: Team name

        Returns:
            Dictionary with stage statistics
        """
        # Filter matches for the team
        team_home_matches = stage_matches[stage_matches["home_team"] == team]
        team_away_matches = stage_matches[stage_matches["away_team"] == team]

        total_matches = len(team_home_matches) + len(team_away_matches)

        if total_matches == 0:
            # If no matches in this stage, return default values
            return {
                "stage_matches": 0,
                "stage_win_rate": 0.0,
                "stage_goals_scored_avg": 0.0,
                "stage_goals_conceded_avg": 0.0,
            }

        # Calculate wins
        home_wins = len(
            team_home_matches[
                team_home_matches["home_score"] > team_home_matches["away_score"]
            ]
        )
        away_wins = len(
            team_away_matches[
                team_away_matches["away_score"] > team_away_matches["home_score"]
            ]
        )
        total_wins = home_wins + away_wins

        # Calculate goals
        home_goals_scored = team_home_matches["home_score"].sum()
        away_goals_scored = team_away_matches["away_score"].sum()
        home_goals_conceded = team_home_matches["away_score"].sum()
        away_goals_conceded = team_away_matches["home_score"].sum()

        total_goals_scored = home_goals_scored + away_goals_scored
        total_goals_conceded = home_goals_conceded + away_goals_conceded

        # Create stage statistics dictionary
        stage_stats = {
            "stage_matches": total_matches,
            "stage_win_rate": total_wins / total_matches,
            "stage_goals_scored_avg": total_goals_scored / total_matches,
            "stage_goals_conceded_avg": total_goals_conceded / total_matches,
        }

        return stage_stats

    def extract_match_events_features(
        self,
        home_team: str,
        away_team: str,
        match_date: datetime,
        season: str,
        stage: str = "Regular Season",
    ) -> Dict[str, Any]:
        """
        Extract features specifically for predicting match events.

        Args:
            home_team: Name of the home team
            away_team: Name of the away team
            match_date: Date of the match
            season: Season identifier
            stage: Tournament stage

        Returns:
            Dictionary containing all match event features
        """
        # Start with basic match features
        features = self.extract_match_features(
            home_team, away_team, match_date, season, stage
        )

        # Filter matches before the current match date
        past_matches = self.matches_df[self.matches_df["date"] < match_date].copy()

        # Add event-specific features
        features.update(
            self._extract_cards_features(past_matches, home_team, away_team)
        )
        features.update(
            self._extract_corner_features(past_matches, home_team, away_team)
        )
        features.update(
            self._extract_fouls_features(past_matches, home_team, away_team)
        )
        features.update(self._extract_btts_features(past_matches, home_team, away_team))

        return features

    def _extract_cards_features(
        self, past_matches: pd.DataFrame, home_team: str, away_team: str
    ) -> Dict[str, float]:
        """
        Extract features related to yellow/red cards.

        Args:
            past_matches: DataFrame with past matches
            home_team: Home team name
            away_team: Away team name

        Returns:
            Dictionary with card-related features
        """
        features = {}

        # Function to calculate card stats for a team
        def calculate_card_stats(team):
            team_home_matches = past_matches[
                (past_matches["home_team"] == team)
                & ("full_time_stats" in past_matches.columns)
            ]
            team_away_matches = past_matches[
                (past_matches["away_team"] == team)
                & ("full_time_stats" in past_matches.columns)
            ]

            home_yellow_cards = []
            away_yellow_cards = []

            # Extract yellow cards from full_time_stats
            for _, match in team_home_matches.iterrows():
                if (
                    isinstance(match.get("full_time_stats"), dict)
                    and "yellow_cards" in match["full_time_stats"]
                ):
                    home_yellow_cards.append(
                        match["full_time_stats"]["yellow_cards"]["home"]
                    )

            for _, match in team_away_matches.iterrows():
                if (
                    isinstance(match.get("full_time_stats"), dict)
                    and "yellow_cards" in match["full_time_stats"]
                ):
                    away_yellow_cards.append(
                        match["full_time_stats"]["yellow_cards"]["away"]
                    )

            # Calculate averages
            yellows_received_home = (
                np.mean(home_yellow_cards) if home_yellow_cards else 0
            )
            yellows_received_away = (
                np.mean(away_yellow_cards) if away_yellow_cards else 0
            )

            # Calculate total yellow card stats
            total_yellows_received = (
                (sum(home_yellow_cards) + sum(away_yellow_cards))
                / (len(home_yellow_cards) + len(away_yellow_cards))
                if (home_yellow_cards or away_yellow_cards)
                else 0
            )

            stats = {
                "yellow_cards_per_game": total_yellows_received,
                "yellow_cards_home": yellows_received_home,
                "yellow_cards_away": yellows_received_away,
            }

            return stats

        # Calculate card stats for each team
        home_cards = calculate_card_stats(home_team)
        away_cards = calculate_card_stats(away_team)

        # Add to features dictionary
        for key, value in home_cards.items():
            features[f"home_{key}"] = value

        for key, value in away_cards.items():
            features[f"away_{key}"] = value

        # Calculate league-wide card averages
        all_home_cards = []
        all_away_cards = []

        for _, match in past_matches.iterrows():
            if (
                isinstance(match.get("full_time_stats"), dict)
                and "yellow_cards" in match["full_time_stats"]
            ):
                all_home_cards.append(match["full_time_stats"]["yellow_cards"]["home"])
                all_away_cards.append(match["full_time_stats"]["yellow_cards"]["away"])

        # League averages
        features["league_avg_yellow_cards"] = (
            np.mean(all_home_cards + all_away_cards)
            if (all_home_cards or all_away_cards)
            else 0
        )
        features["league_avg_home_yellow_cards"] = (
            np.mean(all_home_cards) if all_home_cards else 0
        )
        features["league_avg_away_yellow_cards"] = (
            np.mean(all_away_cards) if all_away_cards else 0
        )

        return features

    def _extract_corner_features(
        self, past_matches: pd.DataFrame, home_team: str, away_team: str
    ) -> Dict[str, float]:
        """
        Extract features related to corner kicks.

        Args:
            past_matches: DataFrame with past matches
            home_team: Home team name
            away_team: Away team name

        Returns:
            Dictionary with corner-related features
        """
        features = {}

        # Function to calculate corner stats for a team
        def calculate_corner_stats(team):
            team_home_matches = past_matches[
                (past_matches["home_team"] == team)
                & ("full_time_stats" in past_matches.columns)
            ]
            team_away_matches = past_matches[
                (past_matches["away_team"] == team)
                & ("full_time_stats" in past_matches.columns)
            ]

            home_corners_won = []
            home_corners_conceded = []
            away_corners_won = []
            away_corners_conceded = []

            # Extract corners from full_time_stats
            for _, match in team_home_matches.iterrows():
                if (
                    isinstance(match.get("full_time_stats"), dict)
                    and "corner_kicks" in match["full_time_stats"]
                ):
                    home_corners_won.append(
                        match["full_time_stats"]["corner_kicks"]["home"]
                    )
                    home_corners_conceded.append(
                        match["full_time_stats"]["corner_kicks"]["away"]
                    )

            for _, match in team_away_matches.iterrows():
                if (
                    isinstance(match.get("full_time_stats"), dict)
                    and "corner_kicks" in match["full_time_stats"]
                ):
                    away_corners_won.append(
                        match["full_time_stats"]["corner_kicks"]["away"]
                    )
                    away_corners_conceded.append(
                        match["full_time_stats"]["corner_kicks"]["home"]
                    )

            # Calculate averages
            corners_won_home = np.mean(home_corners_won) if home_corners_won else 0
            corners_conceded_home = (
                np.mean(home_corners_conceded) if home_corners_conceded else 0
            )
            corners_won_away = np.mean(away_corners_won) if away_corners_won else 0
            corners_conceded_away = (
                np.mean(away_corners_conceded) if away_corners_conceded else 0
            )

            # Calculate total stats
            total_corners_won = (
                (sum(home_corners_won) + sum(away_corners_won))
                / (len(home_corners_won) + len(away_corners_won))
                if (home_corners_won or away_corners_won)
                else 0
            )
            total_corners_conceded = (
                (sum(home_corners_conceded) + sum(away_corners_conceded))
                / (len(home_corners_conceded) + len(away_corners_conceded))
                if (home_corners_conceded or away_corners_conceded)
                else 0
            )

            stats = {
                "corners_per_game": total_corners_won,
                "corners_conceded_per_game": total_corners_conceded,
                "corners_home": corners_won_home,
                "corners_away": corners_won_away,
                "corners_conceded_home": corners_conceded_home,
                "corners_conceded_away": corners_conceded_away,
            }

            return stats

        # Calculate corner stats for each team
        home_corners = calculate_corner_stats(home_team)
        away_corners = calculate_corner_stats(away_team)

        # Add to features dictionary
        for key, value in home_corners.items():
            features[f"home_{key}"] = value

        for key, value in away_corners.items():
            features[f"away_{key}"] = value

        # Calculate league-wide corner averages
        all_matches_corners = []

        for _, match in past_matches.iterrows():
            if (
                isinstance(match.get("full_time_stats"), dict)
                and "corner_kicks" in match["full_time_stats"]
            ):
                home_corners = match["full_time_stats"]["corner_kicks"]["home"]
                away_corners = match["full_time_stats"]["corner_kicks"]["away"]
                all_matches_corners.append(home_corners + away_corners)

        features["league_avg_match_corners"] = (
            np.mean(all_matches_corners) if all_matches_corners else 0
        )

        return features

    def _extract_fouls_features(
        self, past_matches: pd.DataFrame, home_team: str, away_team: str
    ) -> Dict[str, float]:
        """
        Extract features related to fouls.

        Args:
            past_matches: DataFrame with past matches
            home_team: Home team name
            away_team: Away team name

        Returns:
            Dictionary with foul-related features
        """
        features = {}

        # Function to calculate foul stats for a team
        def calculate_foul_stats(team):
            team_home_matches = past_matches[
                (past_matches["home_team"] == team)
                & ("full_time_stats" in past_matches.columns)
            ]
            team_away_matches = past_matches[
                (past_matches["away_team"] == team)
                & ("full_time_stats" in past_matches.columns)
            ]

            home_fouls_committed = []
            home_fouls_suffered = []
            away_fouls_committed = []
            away_fouls_suffered = []

            # Extract fouls from full_time_stats
            for _, match in team_home_matches.iterrows():
                if (
                    isinstance(match.get("full_time_stats"), dict)
                    and "fouls" in match["full_time_stats"]
                ):
                    home_fouls_committed.append(
                        match["full_time_stats"]["fouls"]["home"]
                    )
                    home_fouls_suffered.append(
                        match["full_time_stats"]["fouls"]["away"]
                    )

            for _, match in team_away_matches.iterrows():
                if (
                    isinstance(match.get("full_time_stats"), dict)
                    and "fouls" in match["full_time_stats"]
                ):
                    away_fouls_committed.append(
                        match["full_time_stats"]["fouls"]["away"]
                    )
                    away_fouls_suffered.append(
                        match["full_time_stats"]["fouls"]["home"]
                    )

            # Calculate averages
            fouls_committed_home = (
                np.mean(home_fouls_committed) if home_fouls_committed else 0
            )
            fouls_suffered_home = (
                np.mean(home_fouls_suffered) if home_fouls_suffered else 0
            )
            fouls_committed_away = (
                np.mean(away_fouls_committed) if away_fouls_committed else 0
            )
            fouls_suffered_away = (
                np.mean(away_fouls_suffered) if away_fouls_suffered else 0
            )

            # Calculate total stats
            total_fouls_committed = (
                (sum(home_fouls_committed) + sum(away_fouls_committed))
                / (len(home_fouls_committed) + len(away_fouls_committed))
                if (home_fouls_committed or away_fouls_committed)
                else 0
            )
            total_fouls_suffered = (
                (sum(home_fouls_suffered) + sum(away_fouls_suffered))
                / (len(home_fouls_suffered) + len(away_fouls_suffered))
                if (home_fouls_suffered or away_fouls_suffered)
                else 0
            )

            stats = {
                "fouls_committed_per_game": total_fouls_committed,
                "fouls_suffered_per_game": total_fouls_suffered,
                "fouls_committed_home": fouls_committed_home,
                "fouls_committed_away": fouls_committed_away,
                "fouls_suffered_home": fouls_suffered_home,
                "fouls_suffered_away": fouls_suffered_away,
            }

            return stats

        # Calculate foul stats for each team
        home_fouls = calculate_foul_stats(home_team)
        away_fouls = calculate_foul_stats(away_team)

        # Add to features dictionary
        for key, value in home_fouls.items():
            features[f"home_{key}"] = value

        for key, value in away_fouls.items():
            features[f"away_{key}"] = value

        # Calculate league-wide foul averages
        all_match_fouls = []

        for _, match in past_matches.iterrows():
            if (
                isinstance(match.get("full_time_stats"), dict)
                and "fouls" in match["full_time_stats"]
            ):
                home_fouls = match["full_time_stats"]["fouls"]["home"]
                away_fouls = match["full_time_stats"]["fouls"]["away"]
                all_match_fouls.append(home_fouls + away_fouls)

        features["league_avg_match_fouls"] = (
            np.mean(all_match_fouls) if all_match_fouls else 0
        )

        return features

    def _extract_btts_features(
        self, past_matches: pd.DataFrame, home_team: str, away_team: str
    ) -> Dict[str, float]:
        """
        Extract 'Both Teams To Score' (BTTS) related features.

        Args:
            past_matches: DataFrame with past matches
            home_team: Home team name
            away_team: Away team name

        Returns:
            Dictionary with BTTS-related features
        """
        features = {}

        # Calculate BTTS stats for home team
        home_team_matches = past_matches[
            (past_matches["home_team"] == home_team)
            | (past_matches["away_team"] == home_team)
        ]
        home_btts_count = 0

        for _, match in home_team_matches.iterrows():
            if match["home_score"] > 0 and match["away_score"] > 0:
                home_btts_count += 1

        features["home_team_btts_rate"] = (
            home_btts_count / len(home_team_matches)
            if len(home_team_matches) > 0
            else 0
        )

        # Calculate BTTS stats for away team
        away_team_matches = past_matches[
            (past_matches["home_team"] == away_team)
            | (past_matches["away_team"] == away_team)
        ]
        away_btts_count = 0

        for _, match in away_team_matches.iterrows():
            if match["home_score"] > 0 and match["away_score"] > 0:
                away_btts_count += 1

        features["away_team_btts_rate"] = (
            away_btts_count / len(away_team_matches)
            if len(away_team_matches) > 0
            else 0
        )

        # Calculate BTTS rate when home team plays at home
        home_at_home_matches = past_matches[past_matches["home_team"] == home_team]
        home_at_home_btts_count = 0

        for _, match in home_at_home_matches.iterrows():
            if match["home_score"] > 0 and match["away_score"] > 0:
                home_at_home_btts_count += 1

        features["home_team_home_btts_rate"] = (
            home_at_home_btts_count / len(home_at_home_matches)
            if len(home_at_home_matches) > 0
            else 0
        )

        # Calculate BTTS rate when away team plays away
        away_at_away_matches = past_matches[past_matches["away_team"] == away_team]
        away_at_away_btts_count = 0

        for _, match in away_at_away_matches.iterrows():
            if match["home_score"] > 0 and match["away_score"] > 0:
                away_at_away_btts_count += 1

        features["away_team_away_btts_rate"] = (
            away_at_away_btts_count / len(away_at_away_matches)
            if len(away_at_away_matches) > 0
            else 0
        )

        # Calculate overall league BTTS rate
        all_btts_count = 0

        for _, match in past_matches.iterrows():
            if match["home_score"] > 0 and match["away_score"] > 0:
                all_btts_count += 1

        features["league_btts_rate"] = (
            all_btts_count / len(past_matches) if len(past_matches) > 0 else 0
        )

        return features
