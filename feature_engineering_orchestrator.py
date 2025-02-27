"""
Example script demonstrating how to use the feature engineering pipeline.

This script shows how to:
1. Load processed match data
2. Extract, combine, and select features (including temporal features)
3. Save engineered features for model training
"""

import logging
import json
from pathlib import Path

from src.feature_engineering.pipelines.feature_pipeline import FeaturePipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function."""

    # Example 1: Run the full pipeline
    run_full_pipeline()

    # Example 2: Run pipeline with specific settings
    run_custom_pipeline()

    # Example 3: Examine engineered features
    examine_features()


def run_full_pipeline():
    """Run the full feature engineering pipeline with default settings."""
    logger.info("=== Running full feature engineering pipeline ===")

    # Create pipeline
    pipeline = FeaturePipeline(
        processed_data_dir="data/processed",
        features_data_dir="data/features",
        derby_file="data/features/derbies.json",
    )

    # Run the pipeline
    results = pipeline.run()

    # Log the results
    if results.get("success", False):
        logger.info(
            f"Pipeline succeeded: processed {results.get('matches_processed', 0)} matches"
        )
        logger.info(f"Selected {results.get('selected_features', 0)} features")
        logger.info(f"Duration: {results.get('duration_seconds', 0):.2f} seconds")

        # Log the types of features extracted
        feature_sets = results.get("feature_sets", {})
        logger.info("Extracted feature sets:")
        for feature_set, count in feature_sets.items():
            logger.info(f"  - {feature_set}: {count} rows")
    else:
        logger.error(f"Pipeline failed: {results.get('error', 'Unknown error')}")


def run_custom_pipeline():
    """Run the pipeline with custom settings for specific needs."""
    logger.info("=== Running custom feature engineering pipeline ===")

    # Create pipeline
    pipeline = FeaturePipeline(
        processed_data_dir="data/processed",
        features_data_dir="data/features/custom",
        derby_file="data/features/custom/derbies.json",
    )

    # Specify seasons to process
    seasons = ["2023", "2024", "2025"]
    logger.info(f"Processing seasons: {seasons}")

    # Specify target variable and maximum number of features
    target = "result"  # Could be 'home_score', 'away_score', 'total_goals', etc.
    max_features = 50

    logger.info(f"Target variable: {target}, Maximum features: {max_features}")

    # Run the pipeline with custom settings
    results = pipeline.run(
        seasons=seasons, target=target, max_features=max_features, save=True
    )

    # Log the results
    if results.get("success", False):
        logger.info(
            f"Pipeline succeeded: processed {results.get('matches_processed', 0)} matches"
        )
        logger.info(
            f"Selected {len(results.get('selected_feature_names', []))} features"
        )

        # Print top 10 selected features
        top_features = results.get("selected_feature_names", [])[:10]
        logger.info(f"Top features: {top_features}")

        # Log temporal feature inclusion
        feature_sets = results.get("feature_sets", {})
        if "temporal_features" in feature_sets:
            logger.info(
                f"Included {feature_sets['temporal_features']} temporal features"
            )
    else:
        logger.error(f"Pipeline failed: {results.get('error', 'Unknown error')}")


def examine_features():
    """Examine the engineered features."""
    logger.info("=== Examining engineered features ===")

    features_dir = Path("data/features")

    # Check if features data exists
    if not features_dir.exists():
        logger.error("Features directory does not exist.")
        return

    # Get list of feature files
    feature_files = list(features_dir.glob("*.csv"))
    logger.info(
        f"Found {len(feature_files)} feature files: {[f.name for f in feature_files]}"
    )

    if not feature_files:
        logger.warning("No feature files found.")
        return

    # Check for temporal features specifically
    temporal_file = features_dir / "temporal_features.csv"
    if temporal_file.exists():
        logger.info(f"Found temporal features file: {temporal_file}")

        # Print sample of temporal feature columns if possible
        try:
            import pandas as pd

            df = pd.read_csv(temporal_file)
            temporal_columns = df.columns.tolist()
            logger.info(f"Temporal feature file has {len(temporal_columns)} columns")
            logger.info(f"Sample temporal features: {temporal_columns[:10]}")
        except Exception as e:
            logger.error(f"Error examining temporal features: {e}")

    # Check for selected features file
    selected_features_file = features_dir / "selected_feature_names.json"

    if selected_features_file.exists():
        try:
            with open(selected_features_file, "r", encoding="utf-8") as f:
                selected_features = json.load(f)

            logger.info(f"Found {len(selected_features)} selected features")

            # Print top 20 features
            logger.info("Top 20 selected features:")
            for i, feature in enumerate(selected_features[:20]):
                logger.info(f"  {i+1}. {feature}")

            # Check if any temporal features were selected
            temporal_features_selected = [
                f
                for f in selected_features
                if any(
                    keyword in f
                    for keyword in [
                        "dayofweek",
                        "weekend",
                        "season_phase",
                        "rest_days",
                        "matchday",
                        "congestion",
                        "trend",
                    ]
                )
            ]

            if temporal_features_selected:
                logger.info(
                    f"Temporal features in top selected features: {len(temporal_features_selected)}"
                )
                logger.info(
                    f"Sample temporal features selected: {temporal_features_selected[:5]}"
                )

        except Exception as e:
            logger.error(f"Error loading selected features: {e}")


if __name__ == "__main__":
    main()
