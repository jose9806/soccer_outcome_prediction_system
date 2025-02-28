"""
Dataset creation script.

This script creates datasets for different prediction tasks from extracted features.
"""

import os
import sys

import argparse
import pandas as pd

from typing import List

import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.config import FEATURES_PATH, DATASETS


def preprocess_dataset(
    df: pd.DataFrame, target_col: str, drop_cols: List[str]
) -> pd.DataFrame:
    """Preprocess dataset for machine learning.

    Args:
        df: Input DataFrame
        target_col: Target column name
        drop_cols: Columns to drop (optional)

    Returns:
        Preprocessed DataFrame
    """
    # Make a copy to avoid modifying the original
    dataset = df.copy()

    # Drop unnecessary columns
    if drop_cols:
        cols_to_drop = [col for col in drop_cols if col in dataset.columns]
        dataset = dataset.drop(columns=cols_to_drop)

    # Handle date columns
    date_cols = [col for col in dataset.columns if "date" in col.lower()]
    for col in date_cols:
        if pd.api.types.is_datetime64_any_dtype(dataset[col]):
            # Extract useful features from dates
            dataset[f"{col}_year"] = dataset[col].dt.year
            dataset[f"{col}_month"] = dataset[col].dt.month
            dataset[f"{col}_day"] = dataset[col].dt.day
            dataset[f"{col}_dayofweek"] = dataset[col].dt.dayofweek

            # Drop original date column
            dataset = dataset.drop(columns=[col])

    # Handle categorical columns
    cat_cols = [
        col
        for col in dataset.columns
        if col not in [target_col] and dataset[col].dtype == "object"
    ]

    # For non-team categorical columns, perform one-hot encoding
    for col in cat_cols:
        if "team" not in col.lower():
            # One-hot encode
            dummies = pd.get_dummies(dataset[col], prefix=col, drop_first=False)
            dataset = pd.concat([dataset, dummies], axis=1)
            dataset = dataset.drop(columns=[col])

    # For team columns, we will handle them separately for prediction
    # (we'll keep them for now)

    # Handle missing values
    # For numeric columns, fill with mean
    num_cols = dataset.select_dtypes(include=["number"]).columns
    for col in num_cols:
        if dataset[col].isna().any():
            dataset[col] = dataset[col].fillna(dataset[col].mean())

    # For categorical columns, fill with mode
    cat_cols = [col for col in dataset.columns if col not in num_cols]
    for col in cat_cols:
        if dataset[col].isna().any():
            dataset[col] = dataset[col].fillna(dataset[col].mode()[0])

    return dataset


def create_match_outcome_datasets(input_path: str, output_path: str):
    """Create match outcome datasets.

    Args:
        input_path: Path to input features
        output_path: Path to save datasets
    """
    # Load match outcome dataset
    outcome_path = os.path.join(input_path, "match_outcome_dataset.csv")
    if not os.path.exists(outcome_path):
        logging.error(f"Match outcome dataset not found at {outcome_path}")
        return

    logging.info(f"Loading match outcome dataset from {outcome_path}")
    outcome_df = pd.read_csv(outcome_path)

    # Convert date to datetime if present
    if "match_date" in outcome_df.columns:
        outcome_df["match_date"] = pd.to_datetime(outcome_df["match_date"])

    # Create dataset for 3-way outcome prediction (home win, draw, away win)
    logging.info("Creating 3-way outcome prediction dataset")

    # Columns to drop for 3-way outcome prediction
    drop_cols = [
        "is_home_win",
        "is_draw",
        "is_away_win",  # Binary outcome indicators
        "home_goals",
        "away_goals",
        "total_goals",
        "goal_difference",  # Goal-related
        "both_teams_scored",  # Additional labels
    ] + [
        col
        for col in outcome_df.columns
        if col.startswith("over_") or col.startswith("under_")
    ]

    outcome_3way_df = preprocess_dataset(outcome_df, "outcome_num", drop_cols)

    # Save 3-way outcome dataset
    os.makedirs(output_path, exist_ok=True)
    outcome_3way_path = os.path.join(output_path, "match_outcome_3way.csv")
    outcome_3way_df.to_csv(outcome_3way_path, index=False)
    logging.info(
        f"Saved 3-way outcome dataset with {len(outcome_3way_df)} samples to {outcome_3way_path}"
    )

    # Create dataset for expected goals prediction
    logging.info("Creating expected goals prediction dataset")

    # Load expected goals dataset if available, otherwise use outcome dataset
    xg_path = os.path.join(input_path, "expected_goals_dataset.csv")
    if os.path.exists(xg_path):
        xg_df = pd.read_csv(xg_path)
        if "match_date" in xg_df.columns:
            xg_df["match_date"] = pd.to_datetime(xg_df["match_date"])
    else:
        xg_df = outcome_df

    # Columns to drop for expected goals prediction
    drop_cols = [
        "outcome",
        "outcome_num",  # Outcome labels
        "is_home_win",
        "is_draw",
        "is_away_win",  # Binary outcome indicators
        "both_teams_scored",  # Additional labels
    ] + [
        col
        for col in xg_df.columns
        if col.startswith("over_") or col.startswith("under_")
    ]

    # Create separate datasets for home and away goals
    home_goals_df = preprocess_dataset(
        xg_df,
        "home_goals",
        drop_cols + ["away_goals", "total_goals", "goal_difference"],
    )
    away_goals_df = preprocess_dataset(
        xg_df,
        "away_goals",
        drop_cols + ["home_goals", "total_goals", "goal_difference"],
    )
    total_goals_df = preprocess_dataset(
        xg_df,
        "total_goals",
        drop_cols + ["home_goals", "away_goals", "goal_difference"],
    )

    # Save expected goals datasets
    home_goals_path = os.path.join(output_path, "home_goals.csv")
    away_goals_path = os.path.join(output_path, "away_goals.csv")
    total_goals_path = os.path.join(output_path, "total_goals.csv")

    home_goals_df.to_csv(home_goals_path, index=False)
    away_goals_df.to_csv(away_goals_path, index=False)
    total_goals_df.to_csv(total_goals_path, index=False)

    logging.info(
        f"Saved home goals dataset with {len(home_goals_df)} samples to {home_goals_path}"
    )
    logging.info(
        f"Saved away goals dataset with {len(away_goals_df)} samples to {away_goals_path}"
    )
    logging.info(
        f"Saved total goals dataset with {len(total_goals_df)} samples to {total_goals_path}"
    )


def create_events_datasets(input_path: str, output_path: str):
    """Create event prediction datasets.

    Args:
        input_path: Path to input features
        output_path: Path to save datasets
    """
    # Event types to process
    event_types = ["yellow_cards", "fouls", "corner_kicks", "both_teams_scored"]

    for event_type in event_types:
        # Load event dataset
        event_path = os.path.join(input_path, f"{event_type}_dataset.csv")
        if not os.path.exists(event_path):
            logging.warning(f"Event dataset for {event_type} not found at {event_path}")
            continue

        logging.info(f"Loading {event_type} dataset from {event_path}")
        event_df = pd.read_csv(event_path)

        # Convert date to datetime if present
        if "match_date" in event_df.columns:
            event_df["match_date"] = pd.to_datetime(event_df["match_date"])

        # Create dataset for different event thresholds
        if event_type == "both_teams_scored":
            # Binary classification (both teams scored or not)
            target_col = "both_teams_scored"
            drop_cols = [
                col
                for col in event_df.columns
                if col != target_col
                and (
                    col.startswith("outcome")
                    or col.startswith("is_")
                    or col.startswith("home_goals")
                    or col.startswith("away_goals")
                    or col.startswith("total_goals")
                    or col.startswith("over_")
                    or col.startswith("under_")
                )
            ]

            btts_df = preprocess_dataset(event_df, target_col, drop_cols)

            # Save BTTS dataset
            os.makedirs(output_path, exist_ok=True)
            btts_path = os.path.join(output_path, "both_teams_scored.csv")
            btts_df.to_csv(btts_path, index=False)
            logging.info(
                f"Saved BTTS dataset with {len(btts_df)} samples to {btts_path}"
            )
        else:
            # Find threshold columns for this event type
            threshold_cols = [
                col
                for col in event_df.columns
                if col.startswith(f"{event_type}_over_")
                or col.startswith(f"{event_type}_under_")
            ]

            if not threshold_cols:
                # Use total count as target
                target_col = f"total_{event_type}"
                if target_col not in event_df.columns:
                    logging.warning(f"No target column found for {event_type}")
                    continue

                drop_cols = [
                    col
                    for col in event_df.columns
                    if col != target_col
                    and (
                        col.startswith("outcome")
                        or col.startswith("is_")
                        or col.startswith("home_goals")
                        or col.startswith("away_goals")
                        or col.startswith("total_goals")
                        or col.startswith("over_")
                        or col.startswith("under_")
                    )
                ]

                event_count_df = preprocess_dataset(event_df, target_col, drop_cols)

                # Save event count dataset
                os.makedirs(output_path, exist_ok=True)
                event_count_path = os.path.join(output_path, f"{event_type}_count.csv")
                event_count_df.to_csv(event_count_path, index=False)
                logging.info(
                    f"Saved {event_type} count dataset with {len(event_count_df)} samples to {event_count_path}"
                )
            else:
                # Create datasets for each threshold
                for threshold_col in threshold_cols:
                    threshold_val = threshold_col.split("_")[-1]
                    direction = "over" if "over" in threshold_col else "under"

                    drop_cols = [
                        col
                        for col in event_df.columns
                        if col != threshold_col
                        and (
                            col.startswith("outcome")
                            or col.startswith("is_")
                            or col.startswith("home_goals")
                            or col.startswith("away_goals")
                            or col.startswith("total_goals")
                            or col.startswith("over_")
                            or col.startswith("under_")
                        )
                    ]

                    threshold_df = preprocess_dataset(
                        event_df, threshold_col, drop_cols
                    )

                    # Save threshold dataset
                    os.makedirs(output_path, exist_ok=True)
                    threshold_path = os.path.join(
                        output_path, f"{event_type}_{direction}_{threshold_val}.csv"
                    )
                    threshold_df.to_csv(threshold_path, index=False)
                    logging.info(
                        f"Saved {event_type} {direction} {threshold_val} dataset with {len(threshold_df)} samples to {threshold_path}"
                    )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Create datasets for prediction tasks")
    parser.add_argument(
        "--input-path",
        "-i",
        type=str,
        default=FEATURES_PATH,
        help="Path to input features",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        default=None,
        help="Path to save datasets (defaults to config.DATASETS paths)",
    )
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        choices=["outcome", "events", "all"],
        default="all",
        help="Prediction task type to create datasets for",
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

    # Determine output paths
    if args.output_path:
        outcome_path = os.path.join(args.output_path, "match_outcome")
        events_path = os.path.join(args.output_path, "events")
    else:
        outcome_path = DATASETS["match_outcome"]
        events_path = DATASETS["events"]

    # Create datasets
    if args.task in ["outcome", "all"]:
        os.makedirs(outcome_path, exist_ok=True)
        create_match_outcome_datasets(args.input_path, outcome_path)

    if args.task in ["events", "all"]:
        os.makedirs(events_path, exist_ok=True)
        create_events_datasets(args.input_path, events_path)

    logging.info("Dataset creation completed successfully")


if __name__ == "__main__":
    main()
