"""
Elo rating system for soccer teams.

This module implements an Elo rating system for calculating team strength ratings
based on match results and provides features for match prediction.
"""

import os
import json
import pandas as pd
import math
from datetime import datetime
from typing import Dict, Tuple, Optional

from src.features.config import ELO_CONFIG


class EloRatingSystem:
    """
    Elo rating system for soccer teams.

    Features include:
    - Team strength ratings
    - Rating changes over time
    - Match outcome probabilities
    - Rating-based features for prediction
    """

    def __init__(
        self,
        initial_rating: float = ELO_CONFIG["initial_rating"],
        k_factor: float = ELO_CONFIG["k_factor"],
        home_advantage: float = ELO_CONFIG["home_advantage"],
    ):
        """Initialize Elo rating system.

        Args:
            initial_rating: Initial rating for new teams
            k_factor: K-factor for rating updates
            home_advantage: Home advantage bonus
        """
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.ratings = {}  # Team ratings
        self.rating_history = {}  # Historical ratings by date

    def get_rating(self, team: str, date: Optional[datetime] = None) -> float:
        """Get team rating at a specific date.

        Args:
            team: Team name
            date: Date for which to get rating (None for current)

        Returns:
            Team Elo rating
        """
        if date is None:
            # Return current rating
            return self.ratings.get(team, self.initial_rating)

        # Get historical rating closest to date
        if team not in self.rating_history:
            return self.initial_rating

        history = self.rating_history[team]

        # Get ratings before the given date
        past_ratings = [(d, r) for d, r in history if d <= date]

        if not past_ratings:
            return self.initial_rating

        # Return most recent rating before the date
        return max(past_ratings, key=lambda x: x[0])[1]

    def expected_result(self, home_rating: float, away_rating: float) -> float:
        """Calculate expected match result based on Elo ratings.

        Args:
            home_rating: Home team rating
            away_rating: Away team rating

        Returns:
            Expected result (probability of home team winning)
        """
        # Include home advantage
        adjusted_home_rating = home_rating + self.home_advantage

        # Calculate expected outcome (logistic function)
        exp = (adjusted_home_rating - away_rating) / 400.0
        return 1.0 / (1.0 + math.pow(10, -exp))

    def update_ratings(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        date: datetime,
        importance: float = 1.0,
    ) -> Tuple[float, float]:
        """Update team ratings based on match result.

        Args:
            home_team: Home team name
            away_team: Away team name
            home_score: Home team score
            away_score: Away team score
            date: Match date
            importance: Match importance factor (scales K-factor)

        Returns:
            Tuple of (home_rating_change, away_rating_change)
        """
        # Get current ratings
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)

        # Ensure teams are in the history dictionary
        if home_team not in self.rating_history:
            self.rating_history[home_team] = []
        if away_team not in self.rating_history:
            self.rating_history[away_team] = []

        # Calculate match result (1 for home win, 0.5 for draw, 0 for away win)
        if home_score > away_score:
            actual_result = 1.0
        elif home_score == away_score:
            actual_result = 0.5
        else:
            actual_result = 0.0

        # Calculate expected result
        expected_result = self.expected_result(home_rating, away_rating)

        # Calculate rating changes
        # Adjust K-factor by match importance and goal difference
        goal_diff = abs(home_score - away_score)
        goal_multiplier = 1.0
        if goal_diff == 2:
            goal_multiplier = 1.5
        elif goal_diff >= 3:
            goal_multiplier = 1.75 + (goal_diff - 3) * 0.25  # Cap at 3.0
            goal_multiplier = min(3.0, goal_multiplier)

        effective_k = self.k_factor * importance * goal_multiplier

        # Calculate rating changes
        home_rating_change = effective_k * (actual_result - expected_result)
        away_rating_change = effective_k * (expected_result - actual_result)

        # Update ratings
        new_home_rating = home_rating + home_rating_change
        new_away_rating = away_rating + away_rating_change

        self.ratings[home_team] = new_home_rating
        self.ratings[away_team] = new_away_rating

        # Update rating history
        self.rating_history[home_team].append((date, new_home_rating))
        self.rating_history[away_team].append((date, new_away_rating))

        return home_rating_change, away_rating_change

    def build_ratings_from_history(self, matches_df: pd.DataFrame):
        """Build team ratings from historical match data.

        Args:
            matches_df: DataFrame with match data
        """
        # Reset ratings
        self.ratings = {}
        self.rating_history = {}

        # Sort matches by date
        sorted_matches = matches_df.sort_values("date")

        # Update ratings for each match
        for _, match in sorted_matches.iterrows():
            home_team = match["home_team"]
            away_team = match["away_team"]
            home_score = match["home_score"]
            away_score = match["away_score"]
            date = match["date"]

            # Extract importance if available
            importance = 1.0
            if "match_importance" in match:
                importance = match["match_importance"]

            # Update ratings
            self.update_ratings(
                home_team, away_team, home_score, away_score, date, importance
            )

    def get_rating_features(
        self, home_team: str, away_team: str, match_date: datetime
    ) -> Dict[str, float]:
        """Get Elo rating features for a match.

        Args:
            home_team: Home team name
            away_team: Away team name
            match_date: Match date

        Returns:
            Dictionary with rating features
        """
        # Get team ratings
        home_rating = self.get_rating(home_team, match_date)
        away_rating = self.get_rating(away_team, match_date)

        # Calculate expected result
        win_probability = self.expected_result(home_rating, away_rating)

        # Calculate rating difference
        rating_diff = home_rating - away_rating
        effective_rating_diff = rating_diff + self.home_advantage

        # Calculate form (rating change over last 5 matches)
        home_trend = self._calculate_rating_trend(home_team, match_date)
        away_trend = self._calculate_rating_trend(away_team, match_date)

        features = {
            "home_elo": home_rating,
            "away_elo": away_rating,
            "elo_difference": rating_diff,
            "effective_elo_diff": effective_rating_diff,
            "home_win_probability": win_probability,
            "draw_probability": 1
            - abs(2 * win_probability - 1),  # Peak at win_prob = 0.5
            "away_win_probability": 1 - win_probability,
            "home_elo_trend": home_trend,
            "away_elo_trend": away_trend,
        }

        return features

    def _calculate_rating_trend(self, team: str, date: datetime) -> float:
        """Calculate team's rating trend (change over recent matches).

        Args:
            team: Team name
            date: Match date

        Returns:
            Rating trend (positive = improving, negative = declining)
        """
        if team not in self.rating_history:
            return 0.0

        # Get historical ratings before the date
        past_ratings = [(d, r) for d, r in self.rating_history[team] if d < date]

        if len(past_ratings) < 5:
            return 0.0

        # Get last 5 ratings
        recent_ratings = sorted(past_ratings, key=lambda x: x[0], reverse=True)[:5]

        if not recent_ratings:
            return 0.0

        # Calculate trend (difference between newest and oldest rating)
        newest_rating = recent_ratings[0][1]
        oldest_rating = recent_ratings[-1][1]

        return newest_rating - oldest_rating

    def save_ratings(self, filepath: str):
        """Save current ratings to file.

        Args:
            filepath: Path to save ratings
        """
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Convert rating history from datetime to string
        serializable_history = {}
        for team, history in self.rating_history.items():
            serializable_history[team] = [(d.isoformat(), r) for d, r in history]

        # Create data to save
        data = {
            "config": {
                "initial_rating": self.initial_rating,
                "k_factor": self.k_factor,
                "home_advantage": self.home_advantage,
            },
            "ratings": self.ratings,
            "rating_history": serializable_history,
        }

        # Save to file
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_ratings(self, filepath: str):
        """Load ratings from file.

        Args:
            filepath: Path to load ratings from
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        # Load configuration
        config = data.get("config", {})
        self.initial_rating = config.get("initial_rating", self.initial_rating)
        self.k_factor = config.get("k_factor", self.k_factor)
        self.home_advantage = config.get("home_advantage", self.home_advantage)

        # Load ratings
        self.ratings = data.get("ratings", {})

        # Load rating history (convert string dates back to datetime)
        serialized_history = data.get("rating_history", {})
        self.rating_history = {}
        for team, history in serialized_history.items():
            self.rating_history[team] = [
                (datetime.fromisoformat(d), r) for d, r in history
            ]
