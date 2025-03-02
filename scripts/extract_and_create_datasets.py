#!/usr/bin/env python3
"""
Feature extraction and dataset creation script.

This script extracts features from raw match data and creates structured datasets 
for various prediction tasks. The pipeline includes:
1. Loading match data
2. Feature engineering
3. Dataset creation for match outcomes, expected goals, and various in-game events
4. Creating train/validation/test splits
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir.parent))

from src.features.feature_engineering import FeatureEngineer
from src.features.config import DATA_PATH, FEATURES_PATH, DATASETS_PATH, DATASETS
from src.datasets.match_outcome_dataset import MatchOutcomeDataset
from src.datasets.events_dataset import EventsDataset


def configure_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Configure logging for the script.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
    """
    level = getattr(logging, log_level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Add tqdm-aware handler for progress bars
    class TqdmLoggingHandler(logging.Handler):
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    tqdm_handler = TqdmLoggingHandler()
    tqdm_handler.setLevel(level)
    tqdm_handler.setFormatter(formatter)
    root_logger.addHandler(tqdm_handler)


def load_matches(
    data_path: str, season_filter: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load all match data from JSON files.

    Args:
        data_path: Path to processed match data
        season_filter: List of seasons to include (None for all)

    Returns:
        DataFrame with all matches
    """
    logging.info(f"Loading match data from {data_path}")

    all_matches = []

    # Walk through all files in the data path
    for root, _, files in os.walk(data_path):
        # Filter JSON files
        json_files = [f for f in files if f.endswith(".json")]

        if not json_files:
            continue

        logging.info(f"Found {len(json_files)} JSON files in {root}")

        # Process each JSON file
        for file in tqdm(
            json_files, desc=f"Loading files from {os.path.basename(root)}"
        ):
            try:
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    match_data = json.load(f)

                    # Check if match has the required data
                    if not all(
                        k in match_data
                        for k in [
                            "home_team",
                            "away_team",
                            "home_score",
                            "away_score",
                            "date",
                        ]
                    ):
                        logging.warning(f"Skipping incomplete match data in {file}")
                        continue

                    # Apply season filter if provided
                    if season_filter and match_data.get("season") not in season_filter:
                        continue

                    all_matches.append(match_data)
            except json.JSONDecodeError:
                logging.error(f"Error decoding {os.path.join(root, file)}")
            except Exception as e:
                logging.error(f"Error processing {os.path.join(root, file)}: {str(e)}")

    logging.info(f"Loaded {len(all_matches)} matches")

    if not all_matches:
        logging.error("No matches found. Check data path and season filter.")
        return pd.DataFrame()

    # Convert to DataFrame
    matches_df = pd.DataFrame(all_matches)

    # Convert dates to datetime
    if "date" in matches_df.columns:
        # Handle different date formats
        try:
            matches_df["date"] = pd.to_datetime(matches_df["date"])
        except:
            # Try parsing as Unix timestamp
            try:
                matches_df["date"] = pd.to_datetime(matches_df["date"], unit="s")
            except:
                logging.warning("Failed to parse date column")

    # Sort by date
    if "date" in matches_df.columns:
        matches_df = matches_df.sort_values("date")

    return matches_df


def create_datasets(
    matches_df: pd.DataFrame, output_path: str
) -> Dict[str, pd.DataFrame]:
    """
    Create datasets for various prediction tasks.

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
    if not outcome_df.empty:
        outcome_dir = os.path.join(output_path, "match_outcome")
        os.makedirs(outcome_dir, exist_ok=True)
        outcome_path = os.path.join(outcome_dir, "match_outcome_3way.csv")
        match_outcome_dataset.save_dataset(outcome_df, outcome_path)
        datasets["match_outcome"] = outcome_df
        logging.info(
            f"Saved match outcome dataset with {len(outcome_df)} samples to {outcome_path}"
        )
    else:
        logging.warning("Failed to create match outcome dataset")

    # Create expected goals dataset
    logging.info("Creating expected goals dataset")
    xg_df = match_outcome_dataset.create_expected_goals_dataset()
    if not xg_df.empty:
        xg_dir = os.path.join(output_path, "expected_goals")
        os.makedirs(xg_dir, exist_ok=True)

        # Create separate datasets for home, away, and total goals
        home_goals_df = xg_df.copy()
        away_goals_df = xg_df.copy()
        total_goals_df = xg_df.copy()

        home_goals_path = os.path.join(xg_dir, "home_goals.csv")
        away_goals_path = os.path.join(xg_dir, "away_goals.csv")
        total_goals_path = os.path.join(xg_dir, "total_goals.csv")

        match_outcome_dataset.save_dataset(home_goals_df, home_goals_path)
        match_outcome_dataset.save_dataset(away_goals_df, away_goals_path)
        match_outcome_dataset.save_dataset(total_goals_df, total_goals_path)

        datasets["expected_goals"] = {
            "home_goals": home_goals_df,
            "away_goals": away_goals_df,
            "total_goals": total_goals_df,
        }

        logging.info(f"Saved expected goals datasets")
    else:
        logging.warning("Failed to create expected goals dataset")

    # Create individual event datasets
    event_types = [
        "yellow_cards",
        "fouls",
        "corner_kicks",
        "goals",
        "both_teams_scored",
    ]

    for event_type in event_types:
        try:
            logging.info(f"Creating {event_type} dataset")
            event_df = events_dataset.create_events_dataset(event_type)

            if not event_df.empty:
                event_dir = os.path.join(output_path, "events")
                os.makedirs(event_dir, exist_ok=True)
                event_path = os.path.join(event_dir, f"{event_type}.csv")
                events_dataset.save_dataset(event_df, event_path)
                datasets[event_type] = event_df
                logging.info(
                    f"Saved {event_type} dataset with {len(event_df)} samples to {event_path}"
                )
            else:
                logging.warning(f"No data available for {event_type} dataset")
        except Exception as e:
            logging.error(f"Error creating {event_type} dataset: {str(e)}")

    # Create combined events dataset
    try:
        logging.info("Creating combined events dataset")
        combined_df = events_dataset.create_multiple_events_dataset()

        if not combined_df.empty:
            event_dir = os.path.join(output_path, "events")
            os.makedirs(event_dir, exist_ok=True)
            combined_path = os.path.join(event_dir, "combined_events.csv")
            events_dataset.save_dataset(combined_df, combined_path)
            datasets["combined_events"] = combined_df
            logging.info(
                f"Saved combined events dataset with {len(combined_df)} samples to {combined_path}"
            )
        else:
            logging.warning("No data available for combined events dataset")
    except Exception as e:
        logging.error(f"Error creating combined events dataset: {str(e)}")

    return datasets


def create_time_based_splits(datasets: Dict[str, pd.DataFrame], output_path: str):
    """
    Create time-based train/validation/test splits.

    Args:
        datasets: Dictionary with datasets
        output_path: Path to save splits
    """
    logging.info("Creating time-based train/validation/test splits")

    for name, df in datasets.items():
        if isinstance(df, dict):
            # Handle nested dataset dict (e.g., expected_goals)
            for sub_name, sub_df in df.items():
                create_split_for_dataset(
                    sub_df, os.path.join(output_path, name), sub_name
                )
        else:
            create_split_for_dataset(df, output_path, name)


def create_split_for_dataset(df: pd.DataFrame, output_path: str, name: str):
    """
    Create train/validation/test split for a single dataset.

    Args:
        df: Dataset DataFrame
        output_path: Path to save splits
        name: Name of the dataset
    """
    if df.empty or "date" not in df.columns:
        logging.warning(
            f"Cannot create split for {name}: no date column or empty dataset"
        )
        return

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

    # Create output directory
    split_dir = os.path.join(output_path, "splits", name)
    os.makedirs(split_dir, exist_ok=True)

    # Save splits
    train_path = os.path.join(split_dir, "train.csv")
    val_path = os.path.join(split_dir, "val.csv")
    test_path = os.path.join(split_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logging.info(f"Created time-based splits for {name} dataset:")
    logging.info(f"  Train: {len(train_df)} samples")
    logging.info(f"  Validation: {len(val_df)} samples")
    logging.info(f"  Test: {len(test_df)} samples")


def create_custom_datasets(datasets: Dict[str, pd.DataFrame], output_path: str):
    """
    Create custom datasets for specific prediction thresholds.

    Args:
        datasets: Dictionary with datasets
        output_path: Path to save custom datasets
    """
    logging.info("Creating custom datasets for specific thresholds")

    # Create datasets for over/under goals thresholds
    if "match_outcome" in datasets:
        df = datasets["match_outcome"]
        for threshold in [0.5, 1.5, 2.5, 3.5, 4.5]:
            try:
                # Create dataset for this threshold
                threshold_str = str(threshold).replace(".", "_")
                over_df = df.copy()
                under_df = df.copy()

                # Set target columns
                over_df["target"] = over_df[f"over_{threshold}_goals"]
                under_df["target"] = under_df[f"under_{threshold}_goals"]

                # Save datasets
                os.makedirs(os.path.join(output_path, "events"), exist_ok=True)
                over_path = os.path.join(
                    output_path, "events", f"goals_over_{threshold_str}.csv"
                )
                under_path = os.path.join(
                    output_path, "events", f"goals_under_{threshold_str}.csv"
                )

                over_df.to_csv(over_path, index=False)
                under_df.to_csv(under_path, index=False)

                logging.info(f"Created custom dataset for goals over/under {threshold}")
            except Exception as e:
                logging.error(
                    f"Error creating dataset for threshold {threshold}: {str(e)}"
                )

    # Create datasets for yellow cards thresholds
    if "yellow_cards" in datasets:
        df = datasets["yellow_cards"]
        for threshold in [1.5, 2.5, 3.5, 4.5, 5.5]:
            try:
                # Create dataset for this threshold
                threshold_str = str(threshold).replace(".", "_")
                over_df = df.copy()

                # Set target column
                over_df["target"] = over_df[f"yellow_cards_over_{threshold}"]

                # Save dataset
                os.makedirs(os.path.join(output_path, "events"), exist_ok=True)
                over_path = os.path.join(
                    output_path, "events", f"yellow_cards_over_{threshold_str}.csv"
                )

                over_df.to_csv(over_path, index=False)

                logging.info(
                    f"Created custom dataset for yellow cards over {threshold}"
                )
            except Exception as e:
                logging.error(
                    f"Error creating dataset for yellow cards threshold {threshold}: {str(e)}"
                )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Extract features and create datasets from match data"
    )
    parser.add_argument(
        "--data-path", "-d", type=str, default=DATA_PATH, help="Path to raw match data"
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        default=DATASETS_PATH,
        help="Path to save datasets",
    )
    parser.add_argument(
        "--features-path",
        "-f",
        type=str,
        default=FEATURES_PATH,
        help="Path to save extracted features",
    )
    parser.add_argument(
        "--seasons",
        "-s",
        type=str,
        nargs="+",
        help="Seasons to include (e.g., 2017 2018)",
    )
    parser.add_argument(
        "--create-splits",
        "-c",
        action="store_true",
        help="Create train/validation/test splits",
    )
    parser.add_argument(
        "--custom-datasets",
        action="store_true",
        help="Create additional custom datasets for specific thresholds",
    )
    parser.add_argument(
        "--log-level",
        "-l",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument("--log-file", type=str, help="Path to log file (optional)")

    args = parser.parse_args()

    # Configure logging
    configure_logging(args.log_level, args.log_file)

    # Create output directories
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.features_path, exist_ok=True)

    # Record start time for performance measurement
    start_time = datetime.now()
    logging.info(f"Starting dataset creation pipeline at {start_time}")

    # Load match data
    matches_df = load_matches(args.data_path, args.seasons)

    if matches_df.empty:
        logging.error("No matches found. Check data path and season filter.")
        return 1

    # Create datasets
    datasets = create_datasets(matches_df, args.output_path)

    if not datasets:
        logging.error("Failed to create any datasets")
        return 1

    # Create train/validation/test splits if requested
    if args.create_splits:
        create_time_based_splits(datasets, args.output_path)

    # Create custom datasets if requested
    if args.custom_datasets:
        create_custom_datasets(datasets, args.output_path)

    # Record end time and report duration
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logging.info(f"Dataset creation completed at {end_time}")
    logging.info(f"Total duration: {duration:.2f} seconds")

    return 0


if __name__ == "__main__":
    sys.exit(main())
