# example_data_processing.py
"""
Example script demonstrating how to use the data processing module.

This script shows how to:
1. Load raw match data
2. Process a specific season or all seasons
3. Examine and use the processed data
"""

import json
import logging
from pathlib import Path

from src.data_processing.pipelines import ExtractionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function."""

    # Example 1: Process a single season
    process_single_season()

    # Example 2: Process all seasons
    process_all_seasons()

    # Example 3: Examine processed data
    examine_processed_data()


def process_single_season():
    """Process a single season of match data."""
    logger.info("=== Processing a single season ===")

    # Create pipeline
    pipeline = ExtractionPipeline(
        raw_data_dir="data/raw",
        processed_data_dir="data/processed",
        mapping_file="data/mappings.json",
    )

    # Process a specific season
    season = "2025"  # Change to the season you want to process
    result = pipeline.process_season(season)

    # Log the result
    logger.info(f"Processed {result['processed_matches']} matches for season {season}")
    logger.info(
        f"Found {result['invalid_matches']} invalid matches and {result['inconsistent_matches']} inconsistent matches"
    )


def process_all_seasons():
    """Process all available seasons of match data."""
    logger.info("=== Processing all seasons ===")

    # Create pipeline
    pipeline = ExtractionPipeline(
        raw_data_dir="data/raw",
        processed_data_dir="data/processed",
        mapping_file="data/mappings.json",
    )

    # Process all seasons
    results = pipeline.run()

    # Log the overall results
    logger.info(
        f"Processed {results['total_processed_matches']} matches across {results['seasons_processed']} seasons"
    )
    logger.info(f"Duration: {results['duration_seconds']:.2f} seconds")

    # Log results for each season
    for season_result in results["season_results"]:
        season = season_result["season"]
        if season_result["success"]:
            logger.info(
                f"Season {season}: {season_result['processed_matches']} matches processed"
            )
        else:
            logger.error(
                f"Season {season}: Processing failed - {season_result.get('error', 'Unknown error')}"
            )


def examine_processed_data():
    """Examine the processed data."""
    logger.info("=== Examining processed data ===")

    processed_dir = Path("data/processed")

    # Check if processed data exists
    if not processed_dir.exists():
        logger.error("Processed data directory does not exist.")
        return

    # Get all seasons
    seasons = [d.name for d in processed_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(seasons)} processed seasons: {seasons}")

    if not seasons:
        logger.warning("No processed seasons found.")
        return

    # Select the first season for examination
    season = seasons[0]
    season_dir = processed_dir / season

    # Get all match files
    match_files = list(season_dir.glob("*.json"))
    logger.info(f"Found {len(match_files)} match files for season {season}")

    if not match_files:
        logger.warning(f"No match files found for season {season}.")
        return

    # Examine the first match
    match_file = match_files[0]
    logger.info(f"Examining match file: {match_file}")

    with open(match_file, "r", encoding="utf-8") as f:
        match_data = json.load(f)

    # Print some key information
    logger.info(f"Match ID: {match_data.get('match_id')}")
    logger.info(f"Date: {match_data.get('date')}")
    logger.info(
        f"Teams: {match_data.get('home_team')} vs {match_data.get('away_team')}"
    )
    logger.info(
        f"Score: {match_data.get('home_score')} - {match_data.get('away_score')}"
    )
    logger.info(f"Competition: {match_data.get('competition')}")

    # Check for standardization
    if "original_home_team" in match_data:
        logger.info(
            f"Team standardization: {match_data.get('original_home_team')} -> {match_data.get('home_team')}"
        )

    if "original_competition" in match_data:
        logger.info(
            f"Competition standardization: {match_data.get('original_competition')} -> {match_data.get('competition')}"
        )


if __name__ == "__main__":
    main()
