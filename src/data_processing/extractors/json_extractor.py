# data_processing/extractors/json_extractor.py
import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

logger = logging.getLogger(__name__)


class JsonExtractor:
    """Extracts data from JSON files containing soccer match information."""

    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the JSON extractor.

        Args:
            data_dir: Directory containing raw data organized by year
        """
        self.data_dir = Path(data_dir)
        logger.info(f"Initialized JsonExtractor with data directory: {self.data_dir}")

    def get_available_seasons(self) -> List[str]:
        """Get list of available seasons/years from the data directory."""
        seasons = [d.name for d in self.data_dir.iterdir() if d.is_dir()]
        logger.info(f"Found {len(seasons)} available seasons: {seasons}")
        return seasons

    def load_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a single JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            Dict containing the JSON data

        Raises:
            FileNotFoundError: If the file does not exist
            json.JSONDecodeError: If the file contains invalid JSON
        """
        file_path = Path(file_path)
        logger.debug(f"Loading file: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {file_path}: {e}")
            raise

    def load_season_data(self, season: str) -> List[Dict[str, Any]]:
        """
        Load all match data for a specific season.

        Args:
            season: Season/year to load (e.g., "2017", "2018")

        Returns:
            List of match data dictionaries
        """
        season_dir = self.data_dir / season
        if not season_dir.exists():
            logger.error(f"Season directory does not exist: {season_dir}")
            raise FileNotFoundError(f"Season directory {season_dir} not found")

        match_files = list(season_dir.glob("*.json"))
        logger.info(f"Found {len(match_files)} match files for season {season}")

        matches = []
        for file_path in match_files:
            try:
                match_data = self.load_file(file_path)
                matches.append(match_data)
            except Exception as e:
                logger.warning(f"Error loading match file {file_path}: {e}")
                continue

        logger.info(f"Successfully loaded {len(matches)} matches for season {season}")
        return matches

    def load_all_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load all match data for all available seasons.

        Returns:
            Dictionary mapping season to list of match data
        """
        all_data = {}
        seasons = self.get_available_seasons()

        for season in seasons:
            try:
                season_data = self.load_season_data(season)
                all_data[season] = season_data
                logger.info(f"Loaded {len(season_data)} matches for season {season}")
            except Exception as e:
                logger.error(f"Error loading season {season}: {e}")
                continue

        return all_data
