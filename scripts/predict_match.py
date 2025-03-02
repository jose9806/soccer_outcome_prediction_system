#!/usr/bin/env python3
"""
Match prediction script.

This script uses trained models to make predictions for soccer matches.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir.parent))

from src.ai_models.unified_predictor import UnifiedPredictor


def configure_logging(log_level: str):
    """Configure logging."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def format_prediction_output(prediction: dict) -> str:
    """
    Format the prediction for display.

    Args:
        prediction: Prediction dictionary

    Returns:
        Formatted string representation
    """
    output = []

    # Match info
    match_info = prediction["match_info"]
    output.append("=" * 50)
    output.append(f"Match: {match_info['home_team']} vs {match_info['away_team']}")
    output.append(f"Date: {match_info['match_date']}")
    output.append("=" * 50)

    # Match outcome prediction
    if "match_outcome" in prediction["predictions"]:
        outcome = prediction["predictions"]["match_outcome"]
        output.append("\n[MATCH OUTCOME]")
        output.append(f"Prediction: {outcome['prediction'].upper()}")
        output.append("Probabilities:")

        for outcome_type, prob in outcome["probabilities"].items():
            output.append(f"  {outcome_type}: {prob:.2%}")

    # Expected goals prediction
    if "expected_goals" in prediction["predictions"]:
        xg = prediction["predictions"]["expected_goals"]
        output.append("\n[EXPECTED GOALS]")

        if "home_goals" in xg and "away_goals" in xg:
            output.append(
                f"Predicted score: {match_info['home_team']} {xg['home_goals']} - {xg['away_goals']} {match_info['away_team']}"
            )

        if "total_goals" in xg:
            output.append(f"Total goals: {xg['total_goals']}")

        if "implied_outcome" in xg:
            output.append(f"Implied outcome: {xg['implied_outcome'].upper()}")

    # Events predictions
    if "events" in prediction["predictions"]:
        events = prediction["predictions"]["events"]
        output.append("\n[EVENT PREDICTIONS]")

        # Yellow cards
        if "yellow_cards" in events:
            output.append("\nYellow Cards:")
            output.append(f"  Predicted count: {events['yellow_cards']['count']}")

            if "thresholds" in events["yellow_cards"]:
                output.append("  Thresholds:")
                for threshold, data in events["yellow_cards"]["thresholds"].items():
                    over_prob = next(
                        (v for k, v in data["probabilities"].items() if "over" in k), 0
                    )
                    under_prob = next(
                        (v for k, v in data["probabilities"].items() if "under" in k), 0
                    )
                    output.append(f"    Over {threshold}: {over_prob:.2%}")
                    output.append(f"    Under {threshold}: {under_prob:.2%}")

        # Corner kicks
        if "corner_kicks" in events:
            output.append("\nCorner Kicks:")
            output.append(f"  Predicted count: {events['corner_kicks']['count']}")

            if "thresholds" in events["corner_kicks"]:
                output.append("  Thresholds:")
                for threshold, data in events["corner_kicks"]["thresholds"].items():
                    over_prob = next(
                        (v for k, v in data["probabilities"].items() if "over" in k), 0
                    )
                    under_prob = next(
                        (v for k, v in data["probabilities"].items() if "under" in k), 0
                    )
                    output.append(f"    Over {threshold}: {over_prob:.2%}")
                    output.append(f"    Under {threshold}: {under_prob:.2%}")

        # Fouls
        if "fouls" in events:
            output.append("\nFouls:")
            output.append(f"  Predicted count: {events['fouls']['count']}")

            if "thresholds" in events["fouls"]:
                output.append("  Thresholds:")
                for threshold, data in events["fouls"]["thresholds"].items():
                    over_prob = next(
                        (v for k, v in data["probabilities"].items() if "over" in k), 0
                    )
                    under_prob = next(
                        (v for k, v in data["probabilities"].items() if "under" in k), 0
                    )
                    output.append(f"    Over {threshold}: {over_prob:.2%}")
                    output.append(f"    Under {threshold}: {under_prob:.2%}")

        # Both teams to score
        if "both_teams_to_score" in events:
            btts = events["both_teams_to_score"]
            output.append("\nBoth Teams To Score:")
            output.append(f"  Prediction: {'YES' if btts['prediction'] else 'NO'}")

            yes_prob = btts["probabilities"].get("btts_yes_probability", 0)
            no_prob = btts["probabilities"].get("btts_no_probability", 0)
            output.append(f"  Yes: {yes_prob:.2%}")
