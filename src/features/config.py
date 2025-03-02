"""
Configuration for feature extraction and dataset creation.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Data paths
DATA_PATH = BASE_DIR / "data" / "raw"
FEATURES_PATH = BASE_DIR / "data" / "features"
DATASETS_PATH = BASE_DIR / "data" / "datasets"

# Define specific paths for different dataset types
DATASETS = {
    "match_outcome": DATASETS_PATH / "match_outcome",
    "events": DATASETS_PATH / "events",
    "expected_goals": DATASETS_PATH / "expected_goals",
}

# Feature sets
BASIC_FEATURES = ["match_date", "home_team", "away_team", "season", "stage"]

TEAM_FEATURES = [
    "home_total_matches",
    "away_total_matches",
    "home_win_rate",
    "away_win_rate",
    "home_draw_rate",
    "away_draw_rate",
    "home_loss_rate",
    "away_loss_rate",
    "home_goals_scored_avg",
    "away_goals_scored_avg",
    "home_goals_conceded_avg",
    "away_goals_conceded_avg",
    "home_clean_sheet_rate",
    "away_clean_sheet_rate",
    "home_failed_to_score_rate",
    "away_failed_to_score_rate",
    "home_points_per_game",
    "away_points_per_game",
]

FORM_FEATURES = [
    "home_form_matches",
    "away_form_matches",
    "home_form_win_rate",
    "away_form_win_rate",
    "home_form_draw_rate",
    "away_form_draw_rate",
    "home_form_loss_rate",
    "away_form_loss_rate",
    "home_form_goals_scored_avg",
    "away_form_goals_scored_avg",
    "home_form_goals_conceded_avg",
    "away_form_goals_conceded_avg",
]

H2H_FEATURES = [
    "h2h_matches",
    "h2h_home_win_rate",
    "h2h_away_win_rate",
    "h2h_draw_rate",
    "h2h_home_goals_avg",
    "h2h_away_goals_avg",
    "h2h_total_goals_avg",
    "h2h_btts_rate",
]

SEASON_FEATURES = [
    "home_season_matches",
    "away_season_matches",
    "home_season_points",
    "away_season_points",
    "home_season_position",
    "away_season_position",
    "home_season_goals_scored_avg",
    "away_season_goals_scored_avg",
    "home_season_goals_conceded_avg",
    "away_season_goals_conceded_avg",
]

EVENT_FEATURES = {
    "cards": [
        "home_yellow_cards_per_game",
        "away_yellow_cards_per_game",
        "home_yellow_cards_home",
        "away_yellow_cards_home",
        "home_yellow_cards_away",
        "away_yellow_cards_away",
        "league_avg_yellow_cards",
        "league_avg_home_yellow_cards",
        "league_avg_away_yellow_cards",
    ],
    "corners": [
        "home_corners_per_game",
        "away_corners_per_game",
        "home_corners_conceded_per_game",
        "away_corners_conceded_per_game",
        "home_corners_home",
        "away_corners_home",
        "home_corners_away",
        "away_corners_away",
        "league_avg_match_corners",
    ],
    "fouls": [
        "home_fouls_committed_per_game",
        "away_fouls_committed_per_game",
        "home_fouls_suffered_per_game",
        "away_fouls_suffered_per_game",
        "home_fouls_committed_home",
        "away_fouls_committed_home",
        "home_fouls_committed_away",
        "away_fouls_committed_away",
        "league_avg_match_fouls",
    ],
    "btts": [
        "home_team_btts_rate",
        "away_team_btts_rate",
        "home_team_home_btts_rate",
        "away_team_away_btts_rate",
        "league_btts_rate",
    ],
}

# Thresholds for classification tasks
THRESHOLDS = {
    "over_under_goals": [0.5, 1.5, 2.5, 3.5, 4.5],
    "yellow_cards": [1.5, 2.5, 3.5, 4.5, 5.5],
    "corners": [7.5, 8.5, 9.5, 10.5, 11.5],
    "fouls": [15.5, 20.5, 25.5],
}

# Create necessary directories
for path in [DATA_PATH, FEATURES_PATH, DATASETS_PATH] + list(DATASETS.values()):
    path.mkdir(parents=True, exist_ok=True)
