"""
Statistics aggregation module.

This module provides functions to aggregate and transform team and match statistics
into meaningful features for prediction models.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict

from src.features.config import MATCH_STATS, ADVANCED_STATS, RECENCY_WEIGHT


class StatsAggregator:
    """
    Aggregate statistics from historical match data.

    Features include:
    - Rolling averages with different window sizes
    - Weighted recent performance metrics
    - Head-to-head statistics between teams
    - Home/away performance differences
    """

    def __init__(self, matches_df: pd.DataFrame):
        """Initialize statistics aggregator.

        Args:
            matches_df: DataFrame with match data (optional)
        """
        self.matches_df = matches_df

    def set_matches_df(self, matches_df: pd.DataFrame):
        """Set matches DataFrame.

        Args:
            matches_df: DataFrame with match data
        """
        self.matches_df = matches_df

    def get_team_matches_before(
        self, team: str, date: datetime, days: int
    ) -> pd.DataFrame:
        """Get team matches before a specific date.

        Args:
            team: Team name
            date: Reference date
            days: Number of days to look back (None for all history)

        Returns:
            DataFrame with team matches
        """
        if self.matches_df is None:
            return pd.DataFrame()

        # Set start date if days specified
        start_date = date - timedelta(days=days) if days else datetime(1900, 1, 1)

        # Find matches for the team before the date
        team_matches = self.matches_df[
            (
                (self.matches_df["home_team"] == team)
                | (self.matches_df["away_team"] == team)
            )
            & (self.matches_df["date"] < date)
            & (self.matches_df["date"] >= start_date)
        ].copy()

        # Add is_home column
        team_matches["is_home"] = team_matches["home_team"] == team

        # Add team score and opponent columns for easier analysis
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

        team_matches["opponent"] = np.where(
            team_matches["is_home"],
            team_matches["away_team"],
            team_matches["home_team"],
        )

        # Sort by date (oldest to newest)
        team_matches = team_matches.sort_values("date")

        return team_matches

    def get_head_to_head_matches(
        self, team1: str, team2: str, date: datetime, days: int
    ) -> pd.DataFrame:
        """Get head-to-head matches between two teams.

        Args:
            team1: First team name
            team2: Second team name
            date: Reference date
            days: Number of days to look back (None for all history)

        Returns:
            DataFrame with head-to-head matches
        """
        if self.matches_df is None:
            return pd.DataFrame()

        # Set start date if days specified
        start_date = date - timedelta(days=days) if days else datetime(1900, 1, 1)

        # Find head-to-head matches
        h2h_matches = self.matches_df[
            (
                (
                    (self.matches_df["home_team"] == team1)
                    & (self.matches_df["away_team"] == team2)
                )
                | (
                    (self.matches_df["home_team"] == team2)
                    & (self.matches_df["away_team"] == team1)
                )
            )
            & (self.matches_df["date"] < date)
            & (self.matches_df["date"] >= start_date)
        ].copy()

        # Sort by date (oldest to newest)
        h2h_matches = h2h_matches.sort_values("date")

        return h2h_matches

    def extract_match_stats(
        self, team_matches: pd.DataFrame, is_home: bool | None, prefix: str = ""
    ) -> Dict[str, float]:
        """Extract statistics from match data.

        Args:
            team_matches: DataFrame with team matches
            is_home: Whether to extract home or away stats
            prefix: Prefix for stat names

        Returns:
            Dictionary with statistics
        """
        stats = {}
        home_away = "home" if is_home else "away"
        opposite = "away" if is_home else "home"

        # Filter matches by home/away if needed
        if is_home is not None:
            matches = team_matches[team_matches["is_home"] == is_home]
        else:
            matches = team_matches

        if matches.empty:
            return {}

        # Extract full-time statistics
        for stat in MATCH_STATS + ADVANCED_STATS:
            # Try to get the stat with full path
            full_path = f"full_time_stats.{stat}.{home_away}"
            opp_path = f"full_time_stats.{stat}.{opposite}"

            # Check if the columns exist in the DataFrame
            if full_path in matches.columns:
                # Calculate averages
                avg_value = matches[full_path].mean()
                stats[f"{prefix}{stat}_avg"] = avg_value

                # Calculate standard deviation
                std_value = matches[full_path].std()
                if not np.isnan(std_value):
                    stats[f"{prefix}{stat}_std"] = std_value

                # Calculate trends (difference between recent and older matches)
                if len(matches) >= 5:
                    recent = matches.iloc[-3:][full_path].mean()
                    older = matches.iloc[:-3][full_path].mean()
                    trend = recent - older
                    stats[f"{prefix}{stat}_trend"] = trend

            # Calculate differentials if both columns exist
            if full_path in matches.columns and opp_path in matches.columns:
                diff = matches[full_path] - matches[opp_path]
                stats[f"{prefix}{stat}_diff_avg"] = diff.mean()

        return stats

    def calculate_weighted_stats(
        self, team_matches: pd.DataFrame, decay: float = RECENCY_WEIGHT
    ) -> Dict[str, float]:
        """Calculate exponentially weighted statistics.

        Args:
            team_matches: DataFrame with team matches
            decay: Decay factor for older matches

        Returns:
            Dictionary with weighted statistics
        """
        weighted_stats = {}

        if team_matches.empty:
            return weighted_stats

        # Sort by date (oldest to newest)
        matches = team_matches.sort_values("date")

        # Calculate weights
        n_matches = len(matches)
        weights = np.array([decay ** (n_matches - i - 1) for i in range(n_matches)])
        weights = weights / weights.sum()  # Normalize

        # Calculate weighted stats for both home and away
        for is_home in [True, False, None]:
            if is_home is None:
                prefix = "weighted_"
                filtered_matches = matches
            else:
                prefix = f"weighted_{'home' if is_home else 'away'}_"
                filtered_matches = matches[matches["is_home"] == is_home]

            if filtered_matches.empty:
                continue

            # Recalculate weights for filtered matches
            n_filtered = len(filtered_matches)
            filtered_weights = np.array(
                [decay ** (n_filtered - i - 1) for i in range(n_filtered)]
            )
            filtered_weights = filtered_weights / filtered_weights.sum()

            # Calculate weighted averages for each stat
            for stat in MATCH_STATS + ADVANCED_STATS:
                for side in ["home", "away"]:
                    col = f"full_time_stats.{stat}.{side}"
                    if col in filtered_matches.columns:
                        weighted_avg = np.average(
                            filtered_matches[col], weights=filtered_weights
                        )
                        weighted_stats[f"{prefix}{stat}_{side}"] = weighted_avg

        # Add weighted goal-related stats
        weighted_stats["weighted_goals_scored"] = np.average(
            matches["team_score"], weights=weights
        )
        weighted_stats["weighted_goals_conceded"] = np.average(
            matches["opponent_score"], weights=weights
        )

        return weighted_stats

    def calculate_head_to_head_stats(
        self, team1: str, team2: str, date: datetime, days: int
    ) -> Dict[str, float]:
        """Calculate head-to-head statistics.

        Args:
            team1: First team name
            team2: Second team name
            date: Reference date
            days: Number of days to look back (None for all history)

        Returns:
            Dictionary with head-to-head statistics
        """
        h2h_matches = self.get_head_to_head_matches(team1, team2, date, days)

        stats = {}

        if h2h_matches.empty:
            return stats

        # Calculate basic stats
        n_matches = len(h2h_matches)
        stats["h2h_matches_count"] = n_matches

        # Calculate team1's performance
        team1_home_matches = h2h_matches[h2h_matches["home_team"] == team1]
        team1_away_matches = h2h_matches[h2h_matches["away_team"] == team1]

        # Win rates
        team1_home_wins = team1_home_matches[
            team1_home_matches["home_score"] > team1_home_matches["away_score"]
        ]
        team1_away_wins = team1_away_matches[
            team1_away_matches["away_score"] > team1_away_matches["home_score"]
        ]

        team1_wins = len(team1_home_wins) + len(team1_away_wins)

        # Draw count
        draws = h2h_matches[h2h_matches["home_score"] == h2h_matches["away_score"]]

        if n_matches > 0:
            stats["h2h_team1_win_rate"] = team1_wins / n_matches
            stats["h2h_draw_rate"] = len(draws) / n_matches
            stats["h2h_team2_win_rate"] = (
                1 - stats["h2h_team1_win_rate"] - stats["h2h_draw_rate"]
            )

        # Goal stats
        team1_home_goals = team1_home_matches["home_score"].sum()
        team1_away_goals = team1_away_matches["away_score"].sum()
        team1_home_conceded = team1_home_matches["away_score"].sum()
        team1_away_conceded = team1_away_matches["home_score"].sum()

        team1_goals = team1_home_goals + team1_away_goals
        team1_conceded = team1_home_conceded + team1_away_conceded

        if n_matches > 0:
            stats["h2h_team1_avg_goals"] = team1_goals / n_matches
            stats["h2h_team1_avg_conceded"] = team1_conceded / n_matches
            stats["h2h_team2_avg_goals"] = team1_conceded / n_matches
            stats["h2h_team2_avg_conceded"] = team1_goals / n_matches

        # Recent H2H trend
        if n_matches >= 3:
            recent_matches = h2h_matches.sort_values("date", ascending=False).head(3)

            team1_recent_home = recent_matches[recent_matches["home_team"] == team1]
            team1_recent_away = recent_matches[recent_matches["away_team"] == team1]

            team1_recent_home_points = team1_recent_home.apply(
                lambda m: (
                    3
                    if m["home_score"] > m["away_score"]
                    else (1 if m["home_score"] == m["away_score"] else 0)
                ),
                axis=1,
            ).sum()

            team1_recent_away_points = team1_recent_away.apply(
                lambda m: (
                    3
                    if m["away_score"] > m["home_score"]
                    else (1 if m["away_score"] == m["home_score"] else 0)
                ),
                axis=1,
            ).sum()

            max_possible_points = 3 * len(recent_matches)
            if max_possible_points > 0:
                stats["h2h_team1_recent_form"] = (
                    team1_recent_home_points + team1_recent_away_points
                ) / max_possible_points
                stats["h2h_team2_recent_form"] = 1 - stats["h2h_team1_recent_form"]

        return stats

    def aggregate_team_stats(
        self, team: str, date: datetime, days: int = 365, is_home: bool = False
    ) -> Dict[str, float]:
        """Aggregate comprehensive statistics for a team.

        Args:
            team: Team name
            date: Reference date
            days: Number of days to look back
            is_home: Whether the team is playing at home in target match

        Returns:
            Dictionary with aggregated statistics
        """
        # Get team matches before the date
        team_matches = self.get_team_matches_before(team, date, days)

        if team_matches.empty:
            return {}

        # Aggregate statistics
        stats = {}

        # Basic stats (all matches)
        all_stats = self.extract_match_stats(team_matches, None, "all_")
        stats.update(all_stats)

        # Home stats
        home_matches = team_matches[team_matches["is_home"]]
        if not home_matches.empty:
            home_stats = self.extract_match_stats(home_matches, True, "home_")
            stats.update(home_stats)

        # Away stats
        away_matches = team_matches[~team_matches["is_home"]]
        if not away_matches.empty:
            away_stats = self.extract_match_stats(away_matches, False, "away_")
            stats.update(away_stats)

        # Weighted stats
        weighted_stats = self.calculate_weighted_stats(team_matches)
        stats.update(weighted_stats)

        # Performance metrics
        n_matches = len(team_matches)
        wins = team_matches[
            (
                team_matches["is_home"]
                & (team_matches["home_score"] > team_matches["away_score"])
            )
            | (
                ~team_matches["is_home"]
                & (team_matches["away_score"] > team_matches["home_score"])
            )
        ]
        draws = team_matches[team_matches["home_score"] == team_matches["away_score"]]

        if n_matches > 0:
            stats["win_rate"] = len(wins) / n_matches
            stats["draw_rate"] = len(draws) / n_matches
            stats["loss_rate"] = 1 - stats["win_rate"] - stats["draw_rate"]

        # Home/away specific performance
        if not home_matches.empty:
            home_wins = home_matches[
                home_matches["home_score"] > home_matches["away_score"]
            ]
            home_draws = home_matches[
                home_matches["home_score"] == home_matches["away_score"]
            ]
            stats["home_win_rate"] = len(home_wins) / len(home_matches)
            stats["home_draw_rate"] = len(home_draws) / len(home_matches)
            stats["home_loss_rate"] = (
                1 - stats["home_win_rate"] - stats["home_draw_rate"]
            )

        if not away_matches.empty:
            away_wins = away_matches[
                away_matches["away_score"] > away_matches["home_score"]
            ]
            away_draws = away_matches[
                away_matches["away_score"] == away_matches["home_score"]
            ]
            stats["away_win_rate"] = len(away_wins) / len(away_matches)
            stats["away_draw_rate"] = len(away_draws) / len(away_matches)
            stats["away_loss_rate"] = (
                1 - stats["away_win_rate"] - stats["away_draw_rate"]
            )

        # Recent form (last 5 matches)
        if len(team_matches) >= 5:
            last_5 = team_matches.sort_values("date", ascending=False).head(5)

            # Convert results to points (3 for win, 1 for draw, 0 for loss)
            points = []
            for _, match in last_5.iterrows():
                if match["is_home"]:
                    if match["home_score"] > match["away_score"]:
                        points.append(3)
                    elif match["home_score"] == match["away_score"]:
                        points.append(1)
                    else:
                        points.append(0)
                else:
                    if match["away_score"] > match["home_score"]:
                        points.append(3)
                    elif match["away_score"] == match["home_score"]:
                        points.append(1)
                    else:
                        points.append(0)

            # Calculate form score (0-1 scale)
            max_points = 3 * 5  # 15 points maximum
            form_score = sum(points) / max_points
            stats["recent_form_score"] = form_score

            # Add actual points
            stats["recent_points"] = sum(points)

            # Add form trend (positive = improving, negative = declining)
            if len(points) >= 3:
                recent_points = sum(points[:2])
                older_points = sum(points[2:])
                trend = recent_points - older_points
                stats["form_trend"] = trend

        return stats
