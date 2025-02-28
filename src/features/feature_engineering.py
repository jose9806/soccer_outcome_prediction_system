"""
Feature engineering module.

This module handles the creation of high-level features from basic team and match statistics.
"""

import math
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any

from src.features.team_features import TeamFeatureExtractor
from src.features.temporal_features import TemporalFeatureExtractor
from src.features.elo_ratings import EloRatingSystem
from src.features.stats_aggregator import StatsAggregator
from src.features.config import TIME_WINDOWS, EVENT_TYPES


class FeatureEngineer:
    """
    Feature engineering for match prediction.

    This class combines features from various sources and creates derived features
    for match prediction tasks.
    """

    def __init__(self, matches_df: pd.DataFrame):
        """Initialize feature engineer.

        Args:
            matches_df: DataFrame with match data (optional)
        """
        self.matches_df = matches_df

        # Initialize extractors
        self.team_extractor = TeamFeatureExtractor()
        self.temporal_extractor = TemporalFeatureExtractor(matches_df)
        self.elo_system = EloRatingSystem()
        self.stats_aggregator = StatsAggregator(matches_df)

        if matches_df is not None:
            self.build_initial_features()

    def set_matches_df(self, matches_df: pd.DataFrame):
        """Set matches DataFrame and initialize extractors.

        Args:
            matches_df: DataFrame with match data
        """
        self.matches_df = matches_df
        self.team_extractor.load_all_matches()
        self.temporal_extractor.set_matches_df(matches_df)
        self.stats_aggregator.set_matches_df(matches_df)

        # Build initial features
        self.build_initial_features()

    def build_initial_features(self):
        """Build initial features from match data."""
        # Build Elo ratings
        if self.matches_df is not None:
            self.elo_system.build_ratings_from_history(self.matches_df)

    def extract_match_features(
        self,
        home_team: str,
        away_team: str,
        match_date: datetime,
        season: str,
        stage: str,
    ) -> Dict[str, Any]:
        """Extract comprehensive features for a match.

        Args:
            home_team: Home team name
            away_team: Away team name
            match_date: Match date
            season: Season identifier (optional)
            stage: Tournament stage (optional)

        Returns:
            Dictionary with match features
        """
        # Convert match_date to datetime if it's a string
        if isinstance(match_date, str):
            match_date = pd.to_datetime(match_date)

        # If season not provided, extract from match_date
        if season is None:
            season = str(match_date.year)

        # Initialize features dictionary
        features = {
            "home_team": home_team,
            "away_team": away_team,
            "match_date": match_date,
            "season": season,
            "stage": stage,
        }

        # Add team identity features (one-hot encoding would be added later in pipeline)
        features["home_team_id"] = home_team
        features["away_team_id"] = away_team

        # Add temporal features
        temporal_features = self.temporal_extractor.extract_temporal_features(
            match_date, home_team, away_team, season, stage or "Regular Season"
        )
        features.update(temporal_features)

        # Add Elo rating features
        elo_features = self.elo_system.get_rating_features(
            home_team, away_team, match_date
        )
        features.update(elo_features)

        # Add team statistics for various time windows
        for window in TIME_WINDOWS:
            # Home team stats
            home_stats = self.stats_aggregator.aggregate_team_stats(
                home_team, match_date, days=window, is_home=True
            )
            home_prefixed = {f"home_{k}_{window}d": v for k, v in home_stats.items()}
            features.update(home_prefixed)

            # Away team stats
            away_stats = self.stats_aggregator.aggregate_team_stats(
                away_team, match_date, days=window, is_home=False
            )
            away_prefixed = {f"away_{k}_{window}d": v for k, v in away_stats.items()}
            features.update(away_prefixed)

        # Add head-to-head features (all-time and recent)
        h2h_all = self.stats_aggregator.calculate_head_to_head_stats(
            home_team, away_team, match_date, days=3650  # ~10 years
        )
        features.update(h2h_all)

        h2h_recent = self.stats_aggregator.calculate_head_to_head_stats(
            home_team, away_team, match_date, days=365 * 2  # Last 2 years
        )
        h2h_recent_prefixed = {f"{k}_2y": v for k, v in h2h_recent.items()}
        features.update(h2h_recent_prefixed)

        # Add derived features
        derived_features = self._create_derived_features(features)
        features.update(derived_features)

        return features

    def _create_derived_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Create derived features from existing features.

        Args:
            features: Dictionary with existing features

        Returns:
            Dictionary with derived features
        """
        derived = {}

        # Form difference (home - away)
        if (
            "home_recent_form_score_90d" in features
            and "away_recent_form_score_90d" in features
        ):
            derived["form_diff_90d"] = (
                features["home_recent_form_score_90d"]
                - features["away_recent_form_score_90d"]
            )

        # Goal scoring power ratio
        if (
            "home_weighted_goals_scored_90d" in features
            and "away_weighted_goals_scored_90d" in features
        ):
            home_goals = features["home_weighted_goals_scored_90d"]
            away_goals = features["away_weighted_goals_scored_90d"]

            if away_goals > 0:
                derived["goal_power_ratio"] = home_goals / away_goals

        # Defense strength ratio
        if (
            "home_weighted_goals_conceded_90d" in features
            and "away_weighted_goals_conceded_90d" in features
        ):
            home_conceded = features["home_weighted_goals_conceded_90d"]
            away_conceded = features["away_weighted_goals_conceded_90d"]

            if home_conceded > 0:
                derived["defense_strength_ratio"] = away_conceded / home_conceded

        # Possession dominance
        if (
            "home_possession_avg_90d" in features
            and "away_possession_avg_90d" in features
        ):
            derived["possession_dominance"] = (
                features["home_possession_avg_90d"]
                - features["away_possession_avg_90d"]
            )

        # Relative performance metrics (normalize by league/competition average)
        # Would require additional league-level statistics

        # Convert rates to odds ratios
        for side in ["home", "away"]:
            for window in [90, 180, 365]:
                win_rate_key = f"{side}_win_rate_{window}d"
                if win_rate_key in features and features[win_rate_key] > 0:
                    odds_key = f"{side}_win_odds_{window}d"
                    win_rate = features[win_rate_key]
                    # Add this safety check:
                    if win_rate >= 0.99:
                        derived[odds_key] = 99.0  # Cap the odds at a high value
                    else:
                        derived[odds_key] = win_rate / (1 - win_rate)

        # Calculate expected goals features
        if (
            "home_weighted_goals_scored_90d" in features
            and "away_weighted_goals_conceded_90d" in features
        ):
            # Home expected goals based on home attack strength and away defense weakness
            home_xg = (
                features["home_weighted_goals_scored_90d"]
                + features["away_weighted_goals_conceded_90d"]
            ) / 2
            derived["home_expected_goals"] = home_xg

        if (
            "away_weighted_goals_scored_90d" in features
            and "home_weighted_goals_conceded_90d" in features
        ):
            # Away expected goals based on away attack strength and home defense weakness
            away_xg = (
                features["away_weighted_goals_scored_90d"]
                + features["home_weighted_goals_conceded_90d"]
            ) / 2
            derived["away_expected_goals"] = away_xg

        if "home_expected_goals" in derived and "away_expected_goals" in derived:
            total_xg = derived["home_expected_goals"] + derived["away_expected_goals"]
            derived["total_expected_goals"] = total_xg

            # Probability features based on expected goals
            home_xg = derived["home_expected_goals"]
            away_xg = derived["away_expected_goals"]

            # Simple win probability estimates based on xG difference
            xg_diff = home_xg - away_xg
            derived["xg_diff"] = xg_diff

            # Convert xG diff to win probability using a logistic function
            derived["xg_based_home_win_prob"] = 1 / (1 + np.exp(-xg_diff))
            derived["xg_based_away_win_prob"] = 1 / (1 + np.exp(xg_diff))

            # Estimate draw probability based on how close the xG values are
            derived["xg_based_draw_prob"] = 1 - (
                abs(xg_diff) / 3
            )  # Simple linear scaling
            derived["xg_based_draw_prob"] = max(
                0.1, min(0.5, derived["xg_based_draw_prob"])
            )  # Cap between 0.1 and 0.5

        # Create event probability features
        for event in EVENT_TYPES:
            self._add_event_probability_features(features, derived, event)

        return derived

    def _add_event_probability_features(
        self, features: Dict[str, Any], derived: Dict[str, Any], event: str
    ):
        """Add probability features for specific match events.

        Args:
            features: Dictionary with existing features
            derived: Dictionary to add derived features to
            event: Event type (yellow_cards, corner_kicks, etc.)
        """
        # Calculate average event counts for both teams
        home_event_key_90d = f"home_{event}_avg_90d"
        away_event_key_90d = f"away_{event}_avg_90d"

        if home_event_key_90d in features and away_event_key_90d in features:
            home_avg = features[home_event_key_90d]
            away_avg = features[away_event_key_90d]

            # Estimate total events in match
            total_events = home_avg + away_avg
            derived[f"expected_{event}"] = total_events

            # Create probability features for different thresholds
            thresholds = {
                "yellow_cards": [2, 4, 6, 8],
                "red_cards": [0, 1, 2],
                "fouls": [10, 15, 20, 25, 30],
                "corner_kicks": [5, 8, 10, 12],
                "goals": [0, 1, 2, 3, 4],
                "both_teams_scored": [0.5],  # Special case
            }

            if event in thresholds:
                for threshold in thresholds[event]:
                    # Simple probability modeling based on average counts
                    if event == "both_teams_scored":
                        # Estimate both teams to score probability based on teams' scoring records
                        if (
                            "home_weighted_goals_scored_90d" in features
                            and "away_weighted_goals_scored_90d" in features
                        ):
                            home_scoring_prob = min(
                                0.95, features["home_weighted_goals_scored_90d"] / 2
                            )
                            away_scoring_prob = min(
                                0.95, features["away_weighted_goals_scored_90d"] / 2
                            )
                            btts_prob = home_scoring_prob * away_scoring_prob
                            derived[f"prob_{event}"] = btts_prob
                    else:
                        # For count-based events, estimate probability of exceeding threshold
                        # Use a simple model based on the average count
                        lambda_param = total_events

                        if event == "goals":
                            # Goals need special handling due to their lower counts
                            prob = self._poisson_probability_at_least(
                                lambda_param, threshold
                            )
                        else:
                            # For higher count events, approximate with normal distribution
                            # This is a simplification - could use more sophisticated models
                            z_score = (threshold - lambda_param) / max(
                                1, lambda_param**0.5
                            )
                            prob = 1 - self._standard_normal_cdf(z_score)

                        derived[f"prob_{event}_over_{threshold}"] = prob

    def _poisson_probability_at_least(self, lambda_param: float, k: int) -> float:
        """Calculate Poisson probability of at least k events.

        Args:
            lambda_param: Poisson parameter (average)
            k: Threshold value

        Returns:
            Probability of at least k events
        """
        if k == 0:
            return 1.0

        # Use survival function (1 - CDF) for numerical stability
        # This is an approximation for simplicity
        cumulative_prob = 0.0
        for i in range(k):
            # P(X = i) = e^(-lambda) * lambda^i / i!
            term = np.exp(-lambda_param) * (lambda_param**i) / math.factorial(i)
            cumulative_prob += term

        return 1.0 - cumulative_prob

    def _standard_normal_cdf(self, x: float) -> float:
        """Approximate standard normal CDF.

        Args:
            x: Z-score

        Returns:
            CDF value
        """
        # Simple approximation of the standard normal CDF
        return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * x * (1 + 0.2316419 * x**2)))
