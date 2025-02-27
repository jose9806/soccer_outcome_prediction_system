import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import datetime

from ..extractors.json_extractor import JsonExtractor
from ..extractors.match_extractor import MatchExtractor
from ..transformers.stat_normalizer import StatNormalizer
from ..transformers.team_standardizer import TeamStandardizer
from ..validators.schema_validator import SchemaValidator
from ..validators.consistency_checker import ConsistencyChecker

logger = logging.getLogger(__name__)


class ExtractionPipeline:
    """
    Pipeline for extracting, validating, and transforming soccer match data.

    This pipeline orchestrates the entire process of:
    1. Loading raw JSON data
    2. Validating data against a schema
    3. Checking data consistency
    4. Normalizing statistics
    5. Standardizing team and competition names
    6. Saving processed data
    """

    def __init__(
        self,
        raw_data_dir: str = "data/raw",
        processed_data_dir: str = "data/processed",
        mapping_file: str = "data/mappings.json",
    ):
        """
        Initialize the extraction pipeline.

        Args:
            raw_data_dir: Directory containing raw data organized by year
            processed_data_dir: Directory to save processed data
            mapping_file: Path to a JSON file containing name mappings
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.mapping_file = mapping_file

        # Initialize components
        self.json_extractor = JsonExtractor(raw_data_dir)
        self.match_extractor = MatchExtractor()
        self.stat_normalizer = StatNormalizer()
        self.schema_validator = SchemaValidator()
        self.consistency_checker = ConsistencyChecker()

        # Initialize team standardizer
        if os.path.exists(mapping_file):
            self.team_standardizer = TeamStandardizer(mapping_file)
        else:
            self.team_standardizer = TeamStandardizer()

        logger.info("Initialized ExtractionPipeline")

    def process_season(self, season: str, save: bool = True) -> Dict[str, Any]:
        """
        Process all matches for a specific season.

        Args:
            season: Season/year to process (e.g., "2017", "2018")
            save: Whether to save the processed data

        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing season: {season}")

        # Load raw data
        try:
            raw_matches = self.json_extractor.load_season_data(season)
            logger.info(f"Loaded {len(raw_matches)} raw matches for season {season}")
        except Exception as e:
            logger.error(f"Error loading raw data for season {season}: {e}")
            return {
                "season": season,
                "success": False,
                "error": str(e),
                "processed_matches": 0,
                "invalid_matches": 0,
                "inconsistent_matches": 0,
            }

        # Validate against schema
        validation_results = self.schema_validator.validate_matches(raw_matches)
        valid_matches = validation_results["valid"]
        invalid_matches = validation_results["invalid"]

        logger.info(
            f"Validated {len(raw_matches)} matches: {len(valid_matches)} valid, {len(invalid_matches)} invalid"
        )

        # Extract team names and competitions
        team_names = self.match_extractor.get_team_names(valid_matches)
        competitions = self.match_extractor.get_competitions(valid_matches)

        # Generate name mappings if they don't exist
        self.team_standardizer.generate_team_mappings(team_names)
        self.team_standardizer.generate_competition_mappings(competitions)

        # Process each valid match
        processed_matches = []
        inconsistent_matches = []

        for match in valid_matches:
            # Check consistency
            is_consistent, inconsistencies = (
                self.consistency_checker.check_match_consistency(match)
            )

            if not is_consistent:
                logger.warning(
                    f"Inconsistent match data (id: {match.get('match_id')}): {inconsistencies}"
                )
                inconsistent_matches.append(
                    {"match": match, "inconsistencies": inconsistencies}
                )
                # We still process inconsistent matches, just warn about them

            # Normalize statistics
            normalized_match = self.stat_normalizer.normalize_match_stats(match)

            # Standardize team and competition names
            standardized_match = self.team_standardizer.standardize_match_data(
                normalized_match
            )

            # Add processing metadata
            standardized_match["_processing_metadata"] = {
                "processed_at": datetime.datetime.now().isoformat(),
                "is_consistent": is_consistent,
                "inconsistencies": inconsistencies if not is_consistent else [],
            }

            processed_matches.append(standardized_match)

        # Save processed data
        if save:
            self.save_processed_data(season, processed_matches)

            # Save mappings
            self.team_standardizer.save_mappings(self.mapping_file)

        # Return processing results
        return {
            "season": season,
            "success": True,
            "processed_matches": len(processed_matches),
            "invalid_matches": len(invalid_matches),
            "inconsistent_matches": len(inconsistent_matches),
            "team_standardizations": len(self.team_standardizer.team_mapping),
            "competition_standardizations": len(
                self.team_standardizer.competition_mapping
            ),
        }

    def process_all_seasons(self, save: bool = True) -> List[Dict[str, Any]]:
        """
        Process all available seasons.

        Args:
            save: Whether to save the processed data

        Returns:
            List of dictionaries with processing results for each season
        """
        seasons = self.json_extractor.get_available_seasons()
        logger.info(f"Processing {len(seasons)} seasons: {seasons}")

        results = []

        for season in seasons:
            result = self.process_season(season, save)
            results.append(result)

        return results

    def save_processed_data(
        self, season: str, processed_matches: List[Dict[str, Any]]
    ) -> None:
        """
        Save processed data to the output directory.

        Args:
            season: Season/year
            processed_matches: List of processed match data dictionaries
        """
        season_dir = self.processed_data_dir / season
        os.makedirs(season_dir, exist_ok=True)

        for match in processed_matches:
            match_id = match.get("match_id")
            if not match_id:
                logger.warning("Match missing ID, skipping save")
                continue

            file_path = season_dir / f"{match_id}.json"

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(match, f, indent=2)

        logger.info(f"Saved {len(processed_matches)} processed matches to {season_dir}")

    def run(
        self, seasons: Optional[List[str]] = None, save: bool = True
    ) -> Dict[str, Any]:
        """
        Run the extraction pipeline.

        Args:
            seasons: List of seasons to process, or None to process all
            save: Whether to save the processed data

        Returns:
            Dictionary with overall processing results
        """
        start_time = datetime.datetime.now()

        if seasons:
            logger.info(f"Running extraction pipeline for seasons: {seasons}")
            results = [self.process_season(season, save) for season in seasons]
        else:
            logger.info("Running extraction pipeline for all seasons")
            results = self.process_all_seasons(save)

        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Compute overall statistics
        total_processed = sum(
            r.get("processed_matches", 0) for r in results if r.get("success", False)
        )
        total_invalid = sum(
            r.get("invalid_matches", 0) for r in results if r.get("success", False)
        )
        total_inconsistent = sum(
            r.get("inconsistent_matches", 0) for r in results if r.get("success", False)
        )

        overall_results = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "seasons_processed": len(results),
            "seasons_succeeded": sum(1 for r in results if r.get("success", False)),
            "seasons_failed": sum(1 for r in results if not r.get("success", False)),
            "total_processed_matches": total_processed,
            "total_invalid_matches": total_invalid,
            "total_inconsistent_matches": total_inconsistent,
            "team_standardizations": len(self.team_standardizer.team_mapping),
            "competition_standardizations": len(
                self.team_standardizer.competition_mapping
            ),
            "season_results": results,
        }

        logger.info(f"Extraction pipeline completed in {duration:.2f} seconds")
        logger.info(
            f"Processed {total_processed} matches across {len(results)} seasons"
        )

        return overall_results
