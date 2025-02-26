from typing import Optional, List, Dict, Any
from pathlib import Path
import json
from datetime import datetime

from src.config.logger import Logger
from src.scraping.exceptions import StorageError


class JsonStorage:
    """Handles persistent storage of scraped data in JSON format."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.logger = Logger(
            name="JsonStorage",
            color="yellow",
            file_output="src/logs/json_storage.log",
        )
        self._ensure_directory_exists()

    def _ensure_directory_exists(self) -> None:
        """Ensure the storage directory exists."""
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            error_msg = f"Failed to create directory {self.base_dir}: {str(e)}"
            self.logger.error(error_msg)
            raise StorageError(str(self.base_dir), "create_directory", str(e))

    def save_match(self, match_data: Dict[str, Any], season: int) -> Path:
        """
        Save match data to a JSON file.

        Args:
            match_data: Dictionary containing match information
            season: Season year

        Returns:
            Path to the saved file
        """
        season_dir = self.base_dir / str(season)
        try:
            season_dir.mkdir(exist_ok=True)

            # Generate filename from match ID
            match_id = match_data.get("match_id", "").split("/")[-1] or str(
                int(datetime.now().timestamp())
            )
            file_path = season_dir / f"{match_id}.json"

            # Add metadata
            match_data["_metadata"] = {
                "scraped_at": datetime.now().isoformat(),
                "season": season,
                "file_path": str(file_path),
            }

            # Save with proper encoding and formatting
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(match_data, f, ensure_ascii=False, indent=2, default=str)

            self.logger.debug(f"Saved match data to {file_path}")
            return file_path

        except Exception as e:
            error_msg = f"Failed to save match to {season_dir}: {str(e)}"
            self.logger.error(error_msg)
            raise StorageError(str(season_dir), "save_match", str(e))

    def load_match(self, match_id: str, season: int) -> Optional[Dict[str, Any]]:
        """
        Load match data from storage.

        Args:
            match_id: Match identifier
            season: Season year

        Returns:
            Dictionary containing match data if found
        """
        file_path = self.base_dir / str(season) / f"{match_id}.json"
        try:
            if not file_path.exists():
                self.logger.debug(f"Match file not found: {file_path}")
                return None

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.logger.debug(f"Loaded match data from {file_path}")
                return data

        except Exception as e:
            error_msg = f"Failed to load match from {file_path}: {str(e)}"
            self.logger.error(error_msg)
            raise StorageError(str(file_path), "load_match", str(e))

    def list_matches(self, season: int) -> List[str]:
        """
        List all match IDs for a given season.

        Args:
            season: Season year

        Returns:
            List of match IDs
        """
        season_dir = self.base_dir / str(season)
        try:
            if not season_dir.exists():
                self.logger.debug(f"Season directory not found: {season_dir}")
                return []

            match_ids = [f.stem for f in season_dir.glob("*.json")]
            self.logger.debug(f"Found {len(match_ids)} matches for season {season}")
            return match_ids

        except Exception as e:
            error_msg = f"Failed to list matches in {season_dir}: {str(e)}"
            self.logger.error(error_msg)
            raise StorageError(str(season_dir), "list_matches", str(e))

    def delete_match(self, match_id: str, season: int) -> bool:
        """
        Delete a match from storage.

        Args:
            match_id: Match identifier
            season: Season year

        Returns:
            True if deletion was successful
        """
        file_path = self.base_dir / str(season) / f"{match_id}.json"
        try:
            if file_path.exists():
                file_path.unlink()
                self.logger.debug(f"Deleted match file: {file_path}")
                return True
            self.logger.debug(f"Match file not found for deletion: {file_path}")
            return False

        except Exception as e:
            error_msg = f"Failed to delete match {file_path}: {str(e)}"
            self.logger.error(error_msg)
            raise StorageError(str(file_path), "delete_match", str(e))
