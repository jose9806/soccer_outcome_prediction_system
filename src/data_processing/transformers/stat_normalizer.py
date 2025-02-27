import logging
import numpy as np
from typing import Dict, Any, List, Union, Optional

logger = logging.getLogger(__name__)


class StatNormalizer:
    """Normalizes statistics across matches for consistency and comparability."""

    def __init__(self):
        """Initialize the statistics normalizer."""
        logger.info("Initialized StatNormalizer")

    def normalize_possession(self, possession: Dict[str, int]) -> Dict[str, float]:
        """
        Normalize possession statistics to ensure they sum to 100%.

        Args:
            possession: Dictionary with 'home' and 'away' possession values

        Returns:
            Dictionary with normalized possession values
        """
        home = possession.get("home", 0)
        away = possession.get("away", 0)

        if home == 0 and away == 0:
            logger.warning("Both home and away possession are 0, defaulting to 50/50")
            return {"home": 50.0, "away": 50.0}

        total = home + away

        if total != 100:
            logger.debug(
                f"Normalizing possession: {home}/{away} -> {home*100/total:.1f}/{away*100/total:.1f}"
            )
            return {
                "home": round(home * 100 / total, 1),
                "away": round(away * 100 / total, 1),
            }

        return {"home": float(home), "away": float(away)}

    def normalize_expected_goals(self, xg: Dict[str, Any]) -> Dict[str, float]:
        """
        Normalize expected goals to ensure consistency.

        Args:
            xg: Dictionary with 'home' and 'away' expected goals values

        Returns:
            Dictionary with normalized expected goals values
        """
        home = xg.get("home", 0)
        away = xg.get("away", 0)

        # Convert to float if they're not already
        home = float(home) if home is not None else 0.0
        away = float(away) if away is not None else 0.0

        # Round to 2 decimal places for consistency
        return {"home": round(home, 2), "away": round(away, 2)}

    def normalize_stats(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize all relevant statistics in a stats dictionary.

        Args:
            stats: Dictionary with match statistics

        Returns:
            Dictionary with normalized statistics
        """
        normalized = stats.copy()

        # Normalize possession if present
        if "possession" in normalized:
            normalized["possession"] = self.normalize_possession(
                normalized["possession"]
            )

        # Normalize expected goals if present
        if "expected_goals" in normalized:
            normalized["expected_goals"] = self.normalize_expected_goals(
                normalized["expected_goals"]
            )

        return normalized

    def normalize_match_stats(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize all statistics in a match data dictionary.

        Args:
            match_data: Dictionary with match data

        Returns:
            Dictionary with normalized match data
        """
        normalized = match_data.copy()

        # Normalize full time stats
        if "full_time_stats" in normalized:
            normalized["full_time_stats"] = self.normalize_stats(
                normalized["full_time_stats"]
            )

        # Normalize first half stats
        if "first_half_stats" in normalized:
            normalized["first_half_stats"] = self.normalize_stats(
                normalized["first_half_stats"]
            )

        # Normalize second half stats
        if "second_half_stats" in normalized:
            normalized["second_half_stats"] = self.normalize_stats(
                normalized["second_half_stats"]
            )

        return normalized
