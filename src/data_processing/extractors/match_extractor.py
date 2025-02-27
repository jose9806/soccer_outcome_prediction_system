import logging
from typing import Dict, List, Optional, Any, Set

logger = logging.getLogger(__name__)


class MatchExtractor:
    """Extracts specific match data and statistics from match dictionaries."""

    def __init__(self):
        """Initialize the match extractor."""
        logger.info("Initialized MatchExtractor")

    def extract_basic_info(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract basic match information.

        Args:
            match_data: Dictionary containing match data

        Returns:
            Dictionary with basic match information
        """
        basic_info = {
            "match_id": match_data.get("match_id"),
            "date": match_data.get("date"),
            "competition": match_data.get("competition"),
            "season": match_data.get("season"),
            "stage": match_data.get("stage"),
            "home_team": match_data.get("home_team"),
            "away_team": match_data.get("away_team"),
            "home_score": match_data.get("home_score"),
            "away_score": match_data.get("away_score"),
            "result": match_data.get("result"),
            "total_goals": match_data.get("total_goals"),
            "both_teams_scored": match_data.get("both_teams_scored"),
            "status": match_data.get("status"),
        }
        return basic_info

    def extract_stats(
        self, match_data: Dict[str, Any], period: str = "full_time"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract statistics for a specific period of a match.

        Args:
            match_data: Dictionary containing match data
            period: Period to extract stats for ("full_time", "first_half", or "second_half")

        Returns:
            Dictionary with match statistics
        """
        period_key_map = {
            "full_time": "full_time_stats",
            "first_half": "first_half_stats",
            "second_half": "second_half_stats",
        }

        period_key = period_key_map.get(period)
        if not period_key or period_key not in match_data:
            logger.warning(f"Period {period} not found in match data")
            return {}

        return match_data[period_key]

    def extract_odds(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract odds information from match data.

        Args:
            match_data: Dictionary containing match data

        Returns:
            Dictionary with odds information
        """
        if "odds" not in match_data:
            logger.warning("Odds information not found in match data")
            return {}

        return match_data["odds"]

    def get_team_names(self, matches: List[Dict[str, Any]]) -> Set[str]:
        """
        Extract all unique team names from a list of matches.

        Args:
            matches: List of match data dictionaries

        Returns:
            Set of unique team names
        """
        team_names = set()

        for match in matches:
            home_team = match.get("home_team")
            away_team = match.get("away_team")

            if home_team:
                team_names.add(home_team)
            if away_team:
                team_names.add(away_team)

        logger.info(f"Extracted {len(team_names)} unique team names")
        return team_names

    def get_competitions(self, matches: List[Dict[str, Any]]) -> Set[str]:
        """
        Extract all unique competition names from a list of matches.

        Args:
            matches: List of match data dictionaries

        Returns:
            Set of unique competition names
        """
        competitions = set()

        for match in matches:
            competition = match.get("competition")
            if competition:
                competitions.add(competition)

        logger.info(f"Extracted {len(competitions)} unique competition names")
        return competitions
