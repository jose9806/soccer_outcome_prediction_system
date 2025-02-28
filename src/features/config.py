"""
Configuration parameters for feature extraction and dataset creation.
"""

# Time windows for analyzing team performance (in days)
TIME_WINDOWS = [30, 60, 90, 180, 365]

# Weight for recent matches (higher = faster decay)
RECENCY_WEIGHT = 0.97

# Minimum number of matches required for reliable features
MIN_MATCHES = 5

# Match statistics to extract and aggregate
MATCH_STATS = [
    "possession",
    "shots_on_goal",
    "shots_off_goal",
    "blocked_shots",
    "corner_kicks",
    "goal_attempts",
    "goalkeeper_saves",
    "offsides",
    "fouls",
    "yellow_cards",
    "red_cards",
]

# Advanced statistics (if available)
ADVANCED_STATS = [
    "expected_goals",
    "big_chances",
]

# Event types for probability prediction
EVENT_TYPES = [
    "yellow_cards",
    "red_cards",
    "fouls",
    "corner_kicks",
    "goals",
    "both_teams_scored",
]

# Define season periods
SEASON_PERIODS = {
    "early_season": [1, 2, 3],  # First 3 months
    "mid_season": [4, 5, 6, 7, 8],  # Middle months
    "late_season": [9, 10, 11, 12],  # Final months
}

# Elo rating parameters
ELO_CONFIG = {
    "initial_rating": 1500,
    "k_factor": 20,
    "home_advantage": 100,
}

# Paths
DATA_PATH = "data/processed"
FEATURES_PATH = "data/features/extracted"

# Dataset paths
DATASETS = {
    "match_outcome": "datasets/match_outcome",
    "expected_goals": "datasets/expected_goals",
    "events": "datasets/events",
}
