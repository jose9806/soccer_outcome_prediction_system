#!/usr/bin/env python
"""
Prediction script.

This script makes predictions for upcoming matches using trained models.
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import joblib

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.feature_engineering import FeatureEngineer
from src.features.config import DATA_PATH


def load_model(model_path):
    """Load a trained model.

    Args:
        model_path: Path to the trained model

    Returns:
        Loaded model
    """
    return joblib.load(model_path)


def load_matches_data(data_path=DATA_PATH):
    """Load historical match data.

    Args:
        data_path: Path to match data

    Returns:
        DataFrame with matches
    """
    all_matches = []

    # Walk through all files in the data path
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(root, file), "r") as f:
                    try:
                        match_data = json.load(f)
                        all_matches.append(match_data)
                    except json.JSONDecodeError:
                        logging.error(f"Error decoding {os.path.join(root, file)}")

    # Convert to DataFrame
    matches_df = pd.DataFrame(all_matches)

    # Convert dates to datetime
    if "date" in matches_df.columns:
        matches_df["date"] = pd.to_datetime(matches_df["date"])

    # Sort by date
    matches_df.sort_values("date", inplace=True)

    return matches_df


def prepare_match_features(
    home_team, away_team, match_date, matches_df, season=None, stage="Regular Season"
):
    """Prepare features for a match.

    Args:
        home_team: Home team name
        away_team: Away team name
        match_date: Match date (string or datetime)
        matches_df: DataFrame with historical matches
        season: Season identifier (optional)
        stage: Tournament stage (optional)

    Returns:
        Dictionary with match features
    """
    # Convert date to datetime if it's a string
    if isinstance(match_date, str):
        match_date = pd.to_datetime(match_date)

    # If season not provided, extract from match_date
    if season is None:
        season = str(match_date.year)

    # Create feature engineer
    feature_engineer = FeatureEngineer(matches_df)

    # Extract features
    features = feature_engineer.extract_match_features(
        home_team, away_team, match_date, season, stage
    )

    return features


def print_prediction_results(predictions, metadata=None):
    """Print prediction results in a readable format.

    Args:
        predictions: Dictionary with predictions
        metadata: Additional metadata
    """
    # Print header
    print("\n" + "=" * 60)
    if metadata:
        print(f"Match: {metadata['home_team']} vs {metadata['away_team']}")
        print(f"Date: {metadata['match_date'].strftime('%Y-%m-%d')}")
        print(f"Competition: {metadata.get('competition', 'Unknown')}")
        print(f"Stage: {metadata.get('stage', 'Regular Season')}")
        print("-" * 60)

    # Print match outcome predictions
    if "match_outcome" in predictions:
        outcome = predictions["match_outcome"]
        print("Match Outcome Prediction:")

        # For 3-way outcome
        if "probabilities" in outcome:
            probs = outcome["probabilities"]
            print(f"  Home Win: {probs[0]:.2%}")
            print(f"  Draw: {probs[1]:.2%}")
            print(f"  Away Win: {probs[2]:.2%}")

            # Highlight most likely outcome
            most_likely_idx = np.argmax(probs)
            outcomes = ["Home Win", "Draw", "Away Win"]
            print(
                f"  Most Likely Outcome: {outcomes[most_likely_idx]} ({probs[most_likely_idx]:.2%})"
            )
        else:
            print(f"  Predicted Outcome: {outcome['prediction']}")

        print()

    # Print expected goals predictions
    if "expected_goals" in predictions:
        xg = predictions["expected_goals"]
        print("Expected Goals Prediction:")
        if metadata:
            print(f"  {metadata['home_team']}: {xg['home_goals']:.2f}")
            print(f"  {metadata['away_team']}: {xg['away_goals']:.2f}")
        else:
            print(f"  Home Team: {xg['home_goals']:.2f}")
            print(f"  Away Team: {xg['away_goals']:.2f}")
        print(f"  Total Goals: {xg['total_goals']:.2f}")

        # Print implied scoreline
        home_goals = round(xg["home_goals"])
        away_goals = round(xg["away_goals"])
        home_name = metadata['home_team'] if metadata else 'Home Team'
        away_name = metadata['away_team'] if metadata else 'Away Team'
        print(
            f"  Implied Scoreline: {home_name} {home_goals}-{away_goals} {away_name}"
        )

        print()

    # Print event predictions
    if "events" in predictions:
        events = predictions["events"]
        print("Event Predictions:")

        # Format each event prediction
        for event_type, event_pred in events.items():
            if "both_teams_scored" in event_type:
                prob = event_pred["probability"]
                print(f"  Both Teams to Score: {prob:.2%}")
            elif "yellow_cards" in event_type:
                if "count" in event_type:
                    count = event_pred["prediction"]
                    print(f"  Yellow Cards: {count:.1f}")
                else:
                    # Extract threshold from event_type
                    parts = event_type.split("_")
                    threshold = parts[-1]
                    direction = parts[-2]
                    prob = event_pred["probability"]
                    print(f"  Yellow Cards {direction} {threshold}: {prob:.2%}")
            elif "corner_kicks" in event_type:
                if "count" in event_type:
                    count = event_pred["prediction"]
                    print(f"  Corner Kicks: {count:.1f}")
                else:
                    # Extract threshold from event_type
                    parts = event_type.split("_")
                    threshold = parts[-1]
                    direction = parts[-2]
                    prob = event_pred["probability"]
                    print(f"  Corner Kicks {direction} {threshold}: {prob:.2%}")

    print("=" * 60 + "\n")


def predict_match(
    home_team,
    away_team,
    match_date,
    models_path,
    matches_df=None,
    season=None,
    stage="Regular Season",
    competition="Unknown",
):
    """Make predictions for a match.

    Args:
        home_team: Home team name
        away_team: Away team name
        match_date: Match date (string or datetime)
        models_path: Path to trained models
        matches_df: DataFrame with historical matches (optional)
        season: Season identifier (optional)
        stage: Tournament stage (optional)
        competition: Competition name (optional)

    Returns:
        Dictionary with predictions
    """
    # Load historical matches data if not provided
    if matches_df is None:
        matches_df = load_matches_data()

    # Prepare match features
    features = prepare_match_features(
        home_team, away_team, match_date, matches_df, season, stage
    )

    # Convert features to DataFrame
    features_df = pd.DataFrame([features])

    # Convert date column to string to avoid issues with joblib
    if "match_date" in features_df.columns:
        features_df["match_date"] = features_df["match_date"].astype(str)

    # Initialize predictions dictionary
    predictions = {}
    metadata = {
        "home_team": home_team,
        "away_team": away_team,
        "match_date": (
            features["match_date"]
            if isinstance(features["match_date"], datetime)
            else pd.to_datetime(features["match_date"])
        ),
        "competition": competition,
        "stage": stage,
    }

    # Predict match outcome
    outcome_path = os.path.join(models_path, "outcome", "match_outcome_3way.joblib")
    if os.path.exists(outcome_path):
        outcome_model = load_model(outcome_path)

        # Make prediction
        try:
            outcome_pred = outcome_model.predict(features_df)[0]
            outcome_proba = outcome_model.predict_proba(features_df)[0]

            # Add to predictions
            predictions["match_outcome"] = {
                "prediction": outcome_pred,
                "probabilities": outcome_proba.tolist(),
            }
        except Exception as e:
            logging.error(f"Error predicting match outcome: {e}")

    # Predict expected goals
    try:
        xg_predictions = {}

        # Predict home goals
        home_goals_path = os.path.join(
            models_path, "expected_goals", "home_goals.joblib"
        )
        if os.path.exists(home_goals_path):
            home_goals_model = load_model(home_goals_path)
            home_goals_pred = home_goals_model.predict(features_df)[0]
            xg_predictions["home_goals"] = home_goals_pred

        # Predict away goals
        away_goals_path = os.path.join(
            models_path, "expected_goals", "away_goals.joblib"
        )
        if os.path.exists(away_goals_path):
            away_goals_model = load_model(away_goals_path)
            away_goals_pred = away_goals_model.predict(features_df)[0]
            xg_predictions["away_goals"] = away_goals_pred

        # Predict total goals
        total_goals_path = os.path.join(
            models_path, "expected_goals", "total_goals.joblib"
        )
        if os.path.exists(total_goals_path):
            total_goals_model = load_model(total_goals_path)
            total_goals_pred = total_goals_model.predict(features_df)[0]
            xg_predictions["total_goals"] = total_goals_pred

        # Add to predictions if we have any expected goals predictions
        if xg_predictions:
            predictions["expected_goals"] = xg_predictions
    except Exception as e:
        logging.error(f"Error predicting expected goals: {e}")

    # Predict events
    try:
        events_dir = os.path.join(models_path, "events")
        if os.path.exists(events_dir):
            event_predictions = {}

            # Find event models
            event_models = [f for f in os.listdir(events_dir) if f.endswith(".joblib")]

            for event_model_file in event_models:
                event_type = os.path.splitext(event_model_file)[0]
                event_model_path = os.path.join(events_dir, event_model_file)

                event_model = load_model(event_model_path)

                try:
                    # Check if it's a classification or regression task
                    if hasattr(event_model, "predict_proba"):
                        # Classification task
                        event_pred = event_model.predict(features_df)[0]
                        event_proba = event_model.predict_proba(features_df)[0]

                        # For binary classification, use probability of positive class
                        if len(event_proba) == 2:
                            event_predictions[event_type] = {
                                "prediction": event_pred,
                                "probability": event_proba[1],
                            }
                        else:
                            event_predictions[event_type] = {
                                "prediction": event_pred,
                                "probabilities": event_proba.tolist(),
                            }
                    else:
                        # Regression task
                        event_pred = event_model.predict(features_df)[0]
                        event_predictions[event_type] = {"prediction": event_pred}
                except Exception as e:
                    logging.error(f"Error predicting {event_type}: {e}")

            # Add to predictions if we have any event predictions
            if event_predictions:
                predictions["events"] = event_predictions
    except Exception as e:
        logging.error(f"Error predicting events: {e}")

    # Print results
    print_prediction_results(predictions, metadata)

    return predictions, metadata


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Make match predictions")
    parser.add_argument("--home-team", required=True, help="Home team name")
    parser.add_argument("--away-team", required=True, help="Away team name")
    parser.add_argument("--date", required=True, help="Match date (YYYY-MM-DD)")
    parser.add_argument(
        "--season", help="Season identifier (defaults to year from date)"
    )
    parser.add_argument("--stage", default="Regular Season", help="Tournament stage")
    parser.add_argument("--competition", default="Unknown", help="Competition name")
    parser.add_argument(
        "--model-path", default="trained_models", help="Path to trained models"
    )
    parser.add_argument(
        "--data-path", default=DATA_PATH, help="Path to historical match data"
    )
    parser.add_argument("--output", help="Path to save prediction results as JSON")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Parse match date
    match_date = pd.to_datetime(args.date)

    # Set season if not provided
    season = args.season or str(match_date.year)

    # Load historical match data
    matches_df = load_matches_data(args.data_path)

    # Make predictions
    predictions, metadata = predict_match(
        args.home_team,
        args.away_team,
        match_date,
        args.model_path,
        matches_df,
        season,
        args.stage,
        args.competition,
    )

    # Save predictions if output path provided
    if args.output:
        # Convert datetime to string for JSON serialization
        serializable_metadata = {
            k: (v.isoformat() if isinstance(v, datetime) else v)
            for k, v in metadata.items()
        }

        output_data = {"metadata": serializable_metadata, "predictions": predictions}

        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)

        logging.info(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
