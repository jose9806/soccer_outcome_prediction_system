"""
Feature extraction script.

This script extracts features from raw match data and saves them to disk.
"""

import os
import sys
import json
import argparse
import pandas as pd
from typing import Dict, List
from tqdm import tqdm
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.feature_engineering import FeatureEngineer
from src.features.config import DATA_PATH, FEATURES_PATH
from src.datasets.match_outcome_dataset import MatchOutcomeDataset
from src.datasets.events_dataset import EventsDataset


def load_matches(data_path: str, season_filter: List[str]) -> pd.DataFrame:
    """Load all match data.

    Args:
        data_path: Path to processed match data
        season_filter: List of seasons to include (None for all)

    Returns:
        DataFrame with all matches
    """
    all_matches = []

    logging.info(f"Loading match data from {data_path}")

    # Walk through all files in the data path
    for root, _, files in os.walk(data_path):
        for file in tqdm(files, desc="Loading match files"):
            if file.endswith(".json"):
                with open(os.path.join(root, file), "r") as f:
                    try:
                        match_data = json.load(f)

                        # Check if match has the required data
                        if (
                            "season" not in match_data
                            or "home_team" not in match_data
                            or "away_team" not in match_data
                            or "home_score" not in match_data
                            or "away_score" not in match_data
                        ):
                            continue

                        # Apply season filter if provided
                        if (
                            season_filter
                            and match_data.get("season") not in season_filter
                        ):
                            continue

                        all_matches.append(match_data)
                    except json.JSONDecodeError:
                        logging.error(f"Error decoding {os.path.join(root, file)}")

    logging.info(f"Loaded {len(all_matches)} matches")

    # Convert to DataFrame
    matches_df = pd.DataFrame(all_matches)

    # Convert dates to datetime
    if "date" in matches_df.columns:
        matches_df["date"] = pd.to_datetime(matches_df["date"])

    # Sort by date
    matches_df.sort_values("date", inplace=True)

    return matches_df


def extract_features(
    matches_df: pd.DataFrame, output_path: str
) -> Dict[str, pd.DataFrame]:
    """Extract features and create datasets.

    Args:
        matches_df: DataFrame with match data
        output_path: Path to save datasets

    Returns:
        Dictionary with created datasets
    """
    # Initialize feature engineer
    feature_engineer = FeatureEngineer(matches_df)

    # Initialize dataset creators
    match_outcome_dataset = MatchOutcomeDataset(matches_df, feature_engineer)
    events_dataset = EventsDataset(matches_df, feature_engineer)

    datasets = {}

    # Create match outcome dataset
    logging.info("Creating match outcome dataset")
    outcome_df = match_outcome_dataset.create_match_outcome_dataset()
    outcome_path = os.path.join(output_path, "match_outcome_dataset.csv")
    match_outcome_dataset.save_dataset(outcome_df, outcome_path)
    datasets["match_outcome"] = outcome_df
    logging.info(
        f"Saved match outcome dataset with {len(outcome_df)} samples to {outcome_path}"
    )

    # Create expected goals dataset
    logging.info("Creating expected goals dataset")
    xg_df = match_outcome_dataset.create_expected_goals_dataset()
    xg_path = os.path.join(output_path, "expected_goals_dataset.csv")
    match_outcome_dataset.save_dataset(xg_df, xg_path)
    datasets["expected_goals"] = xg_df
    logging.info(f"Saved expected goals dataset with {len(xg_df)} samples to {xg_path}")

    # Create individual event datasets
    for event_type in [
        "yellow_cards",
        "fouls",
        "corner_kicks",
        "goals",
        "both_teams_scored",
    ]:
        try:
            logging.info(f"Creating {event_type} dataset")
            event_df = events_dataset.create_events_dataset(event_type)
            if not event_df.empty:
                event_path = os.path.join(output_path, f"{event_type}_dataset.csv")
                events_dataset.save_dataset(event_df, event_path)
                datasets[event_type] = event_df
                logging.info(
                    f"Saved {event_type} dataset with {len(event_df)} samples to {event_path}"
                )
            else:
                logging.warning(f"No data available for {event_type} dataset")
        except Exception as e:
            logging.error(f"Error creating {event_type} dataset: {e}")

    # Create combined events dataset
    logging.info("Creating combined events dataset")
    combined_df = events_dataset.create_multiple_events_dataset()
    if not combined_df.empty:
        combined_path = os.path.join(output_path, "combined_events_dataset.csv")
        events_dataset.save_dataset(combined_df, combined_path)
        datasets["combined_events"] = combined_df
        logging.info(
            f"Saved combined events dataset with {len(combined_df)} samples to {combined_path}"
        )

    return datasets


def create_time_based_splits(datasets: Dict[str, pd.DataFrame], output_path: str):
    """Create time-based train/validation/test splits.

    Args:
        datasets: Dictionary with datasets
        output_path: Path to save splits
    """
    for name, df in datasets.items():
        if df.empty or "date" not in df.columns:
            continue

        # Sort by date
        df = df.sort_values("date")

        # Determine split points (70% train, 15% validation, 15% test)
        n_samples = len(df)
        train_idx = int(n_samples * 0.7)
        val_idx = int(n_samples * 0.85)

        # Create splits
        train_df = df.iloc[:train_idx]
        val_df = df.iloc[train_idx:val_idx]
        test_df = df.iloc[val_idx:]

        # Save splits
        os.makedirs(os.path.join(output_path, "splits", name), exist_ok=True)

        train_path = os.path.join(output_path, "splits", name, "train.csv")
        val_path = os.path.join(output_path, "splits", name, "val.csv")
        test_path = os.path.join(output_path, "splits", name, "test.csv")

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        logging.info(f"Created time-based splits for {name} dataset:")
        logging.info(f"  Train: {len(train_df)} samples")
        logging.info(f"  Validation: {len(val_df)} samples")
        logging.info(f"  Test: {len(test_df)} samples")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Extract features from match data")
    parser.add_argument(
        "--data-path",
        "-d",
        type=str,
        default=DATA_PATH,
        help="Path to processed match data",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        default=FEATURES_PATH,
        help="Path to save extracted features",
    )
    parser.add_argument(
        "--seasons",
        "-s",
        type=str,
        nargs="+",
        help="Seasons to include (e.g. 2017 2018)",
    )
    parser.add_argument(
        "--log-level",
        "-l",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Load match data
    matches_df = load_matches(args.data_path, args.seasons)

    if matches_df.empty:
        logging.error("No matches found. Check data path and season filter.")
        return

    # Extract features and create datasets
    datasets = extract_features(matches_df, args.output_path)

    # Create time-based train/validation/test splits
    create_time_based_splits(datasets, args.output_path)

    logging.info("Feature extraction completed successfully")


if __name__ == "__main__":
    main()
