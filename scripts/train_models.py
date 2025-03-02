#!/usr/bin/env python3
"""
Model training script.

This script trains various prediction models and saves them to disk.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir.parent))

from src.ai_models import (
    MatchOutcomeModel,
    ExpectedGoalsModel,
    EventPredictionModel,
    BTTSModel,
)

try:
    from src.ai_models import DeepMatchOutcomeModel, DeepExpectedGoalsModel

    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False


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


def train_match_outcome_model(
    datasets_dir: str, models_dir: str, algorithm: str, deep_learning: bool
) -> None:
    """
    Train and save a match outcome prediction model.

    Args:
        datasets_dir: Directory containing datasets
        models_dir: Directory to save models
        algorithm: ML algorithm to use ('random_forest' or 'gradient_boosting')
        deep_learning: Whether to use deep learning
    """
    logging.info("Training match outcome model")

    if deep_learning and DEEP_LEARNING_AVAILABLE:
        model = DeepMatchOutcomeModel(data_dir=datasets_dir)
        model.prepare_data()
        model.build_model()
        model.train(epochs=50, batch_size=64)
    else:
        model = MatchOutcomeModel(algorithm=algorithm, data_dir=datasets_dir)
        model.prepare_data()
        model.build_pipeline()
        model.train()

    model.save_model(models_dir)
    logging.info("Match outcome model training completed")


def train_expected_goals_models(
    datasets_dir: str, models_dir: str, algorithm: str, deep_learning: bool
) -> None:
    """
    Train and save expected goals prediction models.

    Args:
        datasets_dir: Directory containing datasets
        models_dir: Directory to save models
        algorithm: ML algorithm to use ('random_forest' or 'gradient_boosting')
        deep_learning: Whether to use deep learning
    """
    for target_type in ["total_goals", "home_goals", "away_goals"]:
        logging.info(f"Training {target_type} model")

        if deep_learning and DEEP_LEARNING_AVAILABLE:
            model = DeepExpectedGoalsModel(
                target_type=target_type, data_dir=datasets_dir
            )
            model.prepare_data()
            model.build_model()
            model.train(epochs=50, batch_size=64)
        else:
            model = ExpectedGoalsModel(
                target_type=target_type, algorithm=algorithm, data_dir=datasets_dir
            )
            model.prepare_data()
            model.build_pipeline()
            model.train()

        model.save_model(models_dir)
        logging.info(f"{target_type} model training completed")


def train_event_prediction_models(
    datasets_dir: str, models_dir: str, algorithm: str
) -> None:
    """
    Train and save event prediction models.

    Args:
        datasets_dir: Directory containing datasets
        models_dir: Directory to save models
        algorithm: ML algorithm to use ('random_forest' or 'gradient_boosting')
    """
    # Define event types and thresholds
    event_config = {
        "yellow_cards": [1.5, 2.5, 3.5, 4.5, 5.5],
        "corner_kicks": [7.5, 8.5, 9.5, 10.5, 11.5],
        "fouls": [15.5, 20.5, 25.5],
    }

    for event_type, thresholds in event_config.items():
        # Train count prediction model
        logging.info(f"Training {event_type} count prediction model")
        model = EventPredictionModel(
            event_type=event_type,
            prediction_type="count",
            algorithm=algorithm,
            data_dir=datasets_dir,
        )
        model.prepare_data()
        model.build_pipeline()
        model.train()
        model.save_model(models_dir)
        logging.info(f"{event_type} count model training completed")

        # Train over/under models for each threshold
        for threshold in thresholds:
            threshold_str = str(threshold).replace(".", "_")
            logging.info(f"Training {event_type} over/under {threshold} model")

            model = EventPredictionModel(
                event_type=event_type,
                prediction_type="over_under",
                threshold=threshold,
                algorithm=algorithm,
                data_dir=datasets_dir,
            )
            model.prepare_data()
            model.build_pipeline()
            model.train()
            model.save_model(models_dir)
            logging.info(
                f"{event_type} over/under {threshold} model training completed"
            )


def train_btts_model(datasets_dir: str, models_dir: str, algorithm: str) -> None:
    """
    Train and save a BTTS prediction model.

    Args:
        datasets_dir: Directory containing datasets
        models_dir: Directory to save models
        algorithm: ML algorithm to use ('random_forest' or 'gradient_boosting')
    """
    logging.info("Training BTTS model")
    model = BTTSModel(algorithm=algorithm, data_dir=datasets_dir)
    model.prepare_data()
    model.build_pipeline()
    model.train()
    model.save_model(models_dir)
    logging.info("BTTS model training completed")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train prediction models")
    parser.add_argument(
        "--datasets-dir",
        type=str,
        default="data/datasets",
        help="Directory containing datasets",
    )
    parser.add_argument(
        "--models-dir", type=str, default="models", help="Directory to save models"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["random_forest", "gradient_boosting"],
        default="random_forest",
        help="ML algorithm to use",
    )
    parser.add_argument(
        "--deep-learning", action="store_true", help="Use deep learning models"
    )
    parser.add_argument(
        "--model-types",
        type=str,
        nargs="+",
        choices=["match_outcome", "expected_goals", "events", "btts", "all"],
        default=["all"],
        help="Types of models to train",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Configure logging
    configure_logging(args.log_level)

    # Check deep learning availability
    if args.deep_learning and not DEEP_LEARNING_AVAILABLE:
        logging.warning(
            "Deep learning requested but TensorFlow is not available. Using traditional ML models instead."
        )
        args.deep_learning = False

    # Create models directory
    os.makedirs(args.models_dir, exist_ok=True)

    # Determine which models to train
    train_all = "all" in args.model_types
    train_match_outcome = train_all or "match_outcome" in args.model_types
    train_expected_goals = train_all or "expected_goals" in args.model_types
    train_events = train_all or "events" in args.model_types
    train_btts = train_all or "btts" in args.model_types

    # Train selected models
    if train_match_outcome:
        train_match_outcome_model(
            args.datasets_dir, args.models_dir, args.algorithm, args.deep_learning
        )

    if train_expected_goals:
        train_expected_goals_models(
            args.datasets_dir, args.models_dir, args.algorithm, args.deep_learning
        )

    if train_events:
        train_event_prediction_models(
            args.datasets_dir, args.models_dir, args.algorithm
        )

    if train_btts:
        train_btts_model(args.datasets_dir, args.models_dir, args.algorithm)

    logging.info("Model training completed")


if __name__ == "__main__":
    main()
