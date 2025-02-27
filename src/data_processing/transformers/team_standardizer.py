import logging
import re
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

logger = logging.getLogger(__name__)


class TeamStandardizer:
    """Standardizes team and competition names across datasets."""

    def __init__(self, mapping_file: Optional[str] = None):
        """
        Initialize the team standardizer.

        Args:
            mapping_file: Path to a JSON file containing name mappings
        """
        self.team_mapping = {}
        self.competition_mapping = {}

        if mapping_file:
            self.load_mappings(mapping_file)

        logger.info("Initialized TeamStandardizer")

    def load_mappings(self, mapping_file: str) -> None:
        """
        Load team and competition name mappings from a JSON file.

        Args:
            mapping_file: Path to a JSON file containing mappings

        Raises:
            FileNotFoundError: If the mapping file does not exist
            json.JSONDecodeError: If the mapping file contains invalid JSON
        """
        try:
            with open(mapping_file, "r", encoding="utf-8") as f:
                mappings = json.load(f)

                self.team_mapping = mappings.get("teams", {})
                self.competition_mapping = mappings.get("competitions", {})

            logger.info(
                f"Loaded {len(self.team_mapping)} team mappings and {len(self.competition_mapping)} competition mappings"
            )
        except FileNotFoundError:
            logger.warning(f"Mapping file not found: {mapping_file}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in mapping file: {mapping_file}")
            raise

    def save_mappings(self, mapping_file: str) -> None:
        """
        Save team and competition name mappings to a JSON file.

        Args:
            mapping_file: Path to a JSON file to save mappings to
        """
        mappings = {
            "teams": self.team_mapping,
            "competitions": self.competition_mapping,
        }

        os.makedirs(os.path.dirname(mapping_file), exist_ok=True)

        with open(mapping_file, "w", encoding="utf-8") as f:
            json.dump(mappings, f, indent=2)

        logger.info(
            f"Saved {len(self.team_mapping)} team mappings and {len(self.competition_mapping)} competition mappings to {mapping_file}"
        )

    def add_team_mapping(self, original: str, standardized: str) -> None:
        """
        Add a team name mapping.

        Args:
            original: Original team name
            standardized: Standardized team name
        """
        self.team_mapping[original] = standardized
        logger.debug(f"Added team mapping: {original} -> {standardized}")

    def add_competition_mapping(self, original: str, standardized: str) -> None:
        """
        Add a competition name mapping.

        Args:
            original: Original competition name
            standardized: Standardized competition name
        """
        self.competition_mapping[original] = standardized
        logger.debug(f"Added competition mapping: {original} -> {standardized}")

    def standardize_team_name(self, team_name: str) -> str:
        """
        Standardize a team name using the mapping.

        Args:
            team_name: Original team name

        Returns:
            Standardized team name
        """
        return self.team_mapping.get(team_name, team_name)

    def standardize_competition_name(self, competition_name: str) -> str:
        """
        Standardize a competition name using the mapping.

        Args:
            competition_name: Original competition name

        Returns:
            Standardized competition name
        """
        return self.competition_mapping.get(competition_name, competition_name)

    def generate_team_mappings(self, team_names: Set[str]) -> None:
        """
        Generate team name mappings from a set of team names.
        This is a basic implementation that standardizes case and removes suffixes.

        Args:
            team_names: Set of team names to generate mappings for
        """
        for team_name in team_names:
            # Skip if we already have a mapping
            if team_name in self.team_mapping:
                continue

            # Simple standardization: remove FC, CD, etc. and convert to title case
            standardized = re.sub(r"\b(FC|CD|CF|SA|AC|SC|SD)\b", "", team_name)
            standardized = standardized.strip().title()

            if standardized != team_name:
                self.add_team_mapping(team_name, standardized)

    def generate_competition_mappings(self, competition_names: Set[str]) -> None:
        """
        Generate competition name mappings from a set of competition names.
        This is a basic implementation that standardizes formatting.

        Args:
            competition_names: Set of competition names to generate mappings for
        """
        for comp_name in competition_names:
            # Skip if we already have a mapping
            if comp_name in self.competition_mapping:
                continue

            # Extract country and competition type (e.g., "COLOMBIA: PRIMERA A" -> "Colombia Primera A")
            parts = comp_name.split(":")
            if len(parts) == 2:
                country = parts[0].strip().title()
                comp_type = parts[1].strip().title()
                standardized = f"{country} {comp_type}"

                if standardized != comp_name:
                    self.add_competition_mapping(comp_name, standardized)

    def standardize_match_data(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardize team and competition names in match data.

        Args:
            match_data: Dictionary containing match data

        Returns:
            Dictionary with standardized match data
        """
        standardized = match_data.copy()

        # Standardize competition name
        if "competition" in standardized:
            original_competition = standardized["competition"]
            standardized["competition"] = self.standardize_competition_name(
                original_competition
            )
            standardized["original_competition"] = original_competition

        # Standardize team names
        if "home_team" in standardized:
            original_home_team = standardized["home_team"]
            standardized["home_team"] = self.standardize_team_name(original_home_team)
            standardized["original_home_team"] = original_home_team

        if "away_team" in standardized:
            original_away_team = standardized["away_team"]
            standardized["away_team"] = self.standardize_team_name(original_away_team)
            standardized["original_away_team"] = original_away_team

        return standardized
