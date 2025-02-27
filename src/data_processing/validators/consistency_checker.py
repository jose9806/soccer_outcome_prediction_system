import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ConsistencyChecker:
    """Checks for consistency and anomalies in match data."""

    def __init__(self):
        """Initialize the consistency checker."""
        logger.info("Initialized ConsistencyChecker")

    def check_score_consistency(self, match_data: Dict[str, Any]) -> bool:
        """
        Check if the score is consistent with the result.

        Args:
            match_data: Dictionary containing match data

        Returns:
            True if consistent, False otherwise
        """
        home_score = match_data.get("home_score")
        away_score = match_data.get("away_score")
        result = match_data.get("result")

        if home_score is None or away_score is None or result is None:
            logger.warning("Missing score or result data, cannot check consistency")
            return False

        consistent = False

        if result == "W" and home_score > away_score:
            consistent = True
        elif result == "L" and home_score < away_score:
            consistent = True
        elif result == "D" and home_score == away_score:
            consistent = True

        if not consistent:
            logger.warning(
                f"Score inconsistent with result: {home_score}-{away_score} (result: {result})"
            )

        return consistent

    def check_total_goals(self, match_data: Dict[str, Any]) -> bool:
        """
        Check if total_goals is consistent with home_score + away_score.

        Args:
            match_data: Dictionary containing match data

        Returns:
            True if consistent, False otherwise
        """
        home_score = match_data.get("home_score")
        away_score = match_data.get("away_score")
        total_goals = match_data.get("total_goals")

        if home_score is None or away_score is None or total_goals is None:
            logger.warning(
                "Missing score or total goals data, cannot check consistency"
            )
            return False

        expected_total = home_score + away_score

        if total_goals != expected_total:
            logger.warning(
                f"Total goals inconsistent: {total_goals} (expected: {expected_total})"
            )
            return False

        return True

    def check_both_teams_scored(self, match_data: Dict[str, Any]) -> bool:
        """
        Check if both_teams_scored is consistent with home_score and away_score.

        Args:
            match_data: Dictionary containing match data

        Returns:
            True if consistent, False otherwise
        """
        home_score = match_data.get("home_score")
        away_score = match_data.get("away_score")
        both_scored = match_data.get("both_teams_scored")

        if home_score is None or away_score is None or both_scored is None:
            logger.warning(
                "Missing score or both_teams_scored data, cannot check consistency"
            )
            return False

        expected_both_scored = home_score > 0 and away_score > 0

        if both_scored != expected_both_scored:
            logger.warning(
                f"both_teams_scored inconsistent: {both_scored} (expected: {expected_both_scored})"
            )
            return False

        return True

    def check_stats_totals(self, match_data: Dict[str, Any]) -> List[str]:
        """
        Check if the stats in each period add up correctly.

        Args:
            match_data: Dictionary containing match data

        Returns:
            List of inconsistency messages, empty if all stats are consistent
        """
        inconsistencies = []

        # Check if full_time_stats is the sum of first_half_stats and second_half_stats
        # for applicable statistics like shots, corners, etc.
        stats_to_check = [
            "shots_on_goal",
            "shots_off_goal",
            "blocked_shots",
            "corner_kicks",
            "offsides",
            "fouls",
            "yellow_cards",
            "red_cards",
        ]

        for stat in stats_to_check:
            try:
                full_home = (
                    match_data.get("full_time_stats", {}).get(stat, {}).get("home")
                )
                full_away = (
                    match_data.get("full_time_stats", {}).get(stat, {}).get("away")
                )

                first_home = (
                    match_data.get("first_half_stats", {}).get(stat, {}).get("home", 0)
                )
                first_away = (
                    match_data.get("first_half_stats", {}).get(stat, {}).get("away", 0)
                )

                second_home = (
                    match_data.get("second_half_stats", {}).get(stat, {}).get("home", 0)
                )
                second_away = (
                    match_data.get("second_half_stats", {}).get(stat, {}).get("away", 0)
                )

                # Skip if any full time stat is missing
                if full_home is None or full_away is None:
                    continue

                # Calculate expected values
                expected_home = first_home + second_home
                expected_away = first_away + second_away

                # Check consistency
                if full_home != expected_home:
                    msg = f"Inconsistent {stat} (home): full={full_home}, first+second={expected_home}"
                    inconsistencies.append(msg)
                    logger.warning(msg)

                if full_away != expected_away:
                    msg = f"Inconsistent {stat} (away): full={full_away}, first+second={expected_away}"
                    inconsistencies.append(msg)
                    logger.warning(msg)
            except Exception as e:
                logger.error(f"Error checking {stat} consistency: {e}")

        return inconsistencies

    def check_match_consistency(
        self, match_data: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Perform all consistency checks on a match.

        Args:
            match_data: Dictionary containing match data

        Returns:
            Tuple of (is_consistent, list_of_inconsistency_messages)
        """
        inconsistencies = []

        # Check score consistency
        if not self.check_score_consistency(match_data):
            inconsistencies.append("Score inconsistent with result")

        # Check total goals
        if not self.check_total_goals(match_data):
            inconsistencies.append("Total goals inconsistent with scores")

        # Check both teams scored
        if not self.check_both_teams_scored(match_data):
            inconsistencies.append("both_teams_scored flag inconsistent with scores")

        # Check stats totals
        stat_inconsistencies = self.check_stats_totals(match_data)
        inconsistencies.extend(stat_inconsistencies)

        return len(inconsistencies) == 0, inconsistencies
