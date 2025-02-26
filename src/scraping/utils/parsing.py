from typing import Any, Dict, Optional, Union
from datetime import datetime
import re


def parse_date(date_str: str) -> Optional[datetime]:
    """Parse date string in various formats."""
    formats = ["%d.%m.%Y %H:%M", "%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M"]

    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def parse_score(score_str: str) -> tuple[int, int]:
    """Parse score string into tuple of integers."""
    pattern = r"(\d+)\s*-\s*(\d+)"
    match = re.match(pattern, score_str.strip())
    if match:
        return int(match.group(1)), int(match.group(2))
    raise ValueError(f"Invalid score format: {score_str}")


def parse_percentage(value_str: str) -> float:
    """Parse percentage string into float."""
    try:
        return float(value_str.strip("%"))
    except ValueError:
        raise ValueError(f"Invalid percentage format: {value_str}")


def parse_odds(odds_str: str) -> float:
    """Parse odds string into float."""
    try:
        return float(odds_str.strip())
    except ValueError:
        raise ValueError(f"Invalid odds format: {odds_str}")


def clean_team_name(name: str) -> str:
    """Clean and standardize team names."""
    return re.sub(r"\s+", " ", name.strip())
