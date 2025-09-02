from typing import Optional, Any
from datetime import datetime
from pathlib import Path
from src.scraping.models.soccer_extraction import Match, MatchStats
from src.config.logging_config import get_logger

logger = get_logger(
    "scraping_validations",
    color="cyan",
    enable_file=True,
    file_path="src/logs/scraping_validations.log",
)


def validate_statistics(stats: MatchStats) -> bool:
    """Validate match statistics."""
    try:
        # Check possession sums to 100%
        home_pos, away_pos = stats.possession
        if abs(home_pos + away_pos - 100) > 0.1:
            logger.error("Invalid possession statistics")
            return False

        # Validate shot statistics
        if any(x < 0 for x in stats.shots_on_goal + stats.shots_off_goal):
            logger.error("Invalid shot statistics")
            return False

        return True

    except Exception as e:
        logger.error(f"Statistics validation error: {str(e)}")
        return False


def validate_odds(odds_list: list) -> bool:
    """Validate betting odds."""
    try:
        for odds in odds_list:
            # Check odds are positive
            if any(x <= 0 for x in [odds.home_win, odds.draw, odds.away_win]):
                logger.error("Invalid odds values")
                return False

            # Check bookmaker name is present
            if not odds.bookmaker:
                logger.error("Missing bookmaker name")
                return False

        return True

    except Exception as e:
        logger.error(f"Odds validation error: {str(e)}")
        return False


def create_directory_if_not_exists(directory_path: Path) -> None:
    """
    Create a directory if it doesn't exist.

    Args:
        directory_path: Path to the directory to create
    """
    directory_path.mkdir(parents=True, exist_ok=True)


def validate_match_data(match_data: Any) -> bool:
    """
    Validate match data to ensure it contains required fields.

    Args:
        match_data: Match data object or dictionary to validate

    Returns:
        True if the data is valid, False otherwise
    """
    if not match_data:
        return False

    # Check for required fields
    required_fields = ["home_team", "away_team", "date"]

    for field in required_fields:
        if isinstance(match_data, dict):
            # Dictionary check
            if field not in match_data or match_data[field] is None:
                return False
        else:
            # Object attribute check
            if not hasattr(match_data, field) or getattr(match_data, field) is None:
                return False

    return True


def parse_date(date_str: str) -> Optional[datetime]:
    """
    Parse a date string into a datetime object.

    Args:
        date_str: Date string to parse

    Returns:
        Datetime object if parsing was successful, None otherwise
    """
    date_formats = [
        "%d.%m.%Y %H:%M",  # E.g., 17.12.2017 19:00
        "%d.%m.%y %H:%M",  # E.g., 17.12.17 19:00
        "%Y-%m-%d %H:%M",  # E.g., 2017-12-17 19:00
    ]

    for date_format in date_formats:
        try:
            return datetime.strptime(date_str.strip(), date_format)
        except ValueError:
            pass

    return None
