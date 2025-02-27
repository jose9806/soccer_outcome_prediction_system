import logging
import jsonschema
from jsonschema.exceptions import ValidationError
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class SchemaValidator:
    """Validates match data against a JSON schema."""

    def __init__(self):
        """Initialize the schema validator with the match data schema."""
        self.match_schema = {
            "type": "object",
            "required": [
                "match_id",
                "date",
                "competition",
                "season",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "status",
            ],
            "properties": {
                "match_id": {"type": "string"},
                "date": {"type": "string", "format": "date-time"},
                "competition": {"type": "string"},
                "season": {"type": "string"},
                "stage": {"type": "string"},
                "home_team": {"type": "string"},
                "away_team": {"type": "string"},
                "home_score": {"type": "integer"},
                "away_score": {"type": "integer"},
                "result": {"type": "string", "enum": ["W", "L", "D"]},
                "total_goals": {"type": "integer"},
                "both_teams_scored": {"type": "boolean"},
                "status": {"type": "string"},
                "full_time_stats": {"type": "object"},
                "first_half_stats": {"type": "object"},
                "second_half_stats": {"type": "object"},
                "odds": {"type": "object"},
            },
        }

        logger.info("Initialized SchemaValidator")

    def validate_match(self, match_data: Dict[str, Any]) -> bool:
        """
        Validate match data against the schema.

        Args:
            match_data: Dictionary containing match data

        Returns:
            True if the match data is valid, False otherwise
        """
        try:
            jsonschema.validate(instance=match_data, schema=self.match_schema)
            return True
        except ValidationError as e:
            logger.warning(f"Match data validation failed: {e}")
            return False

    def validate_matches(
        self, matches: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Validate a list of match data and separate valid and invalid matches.

        Args:
            matches: List of match data dictionaries

        Returns:
            Dictionary with 'valid' and 'invalid' lists of match data
        """
        valid_matches = []
        invalid_matches = []

        for match in matches:
            if self.validate_match(match):
                valid_matches.append(match)
            else:
                invalid_matches.append(match)

        result = {"valid": valid_matches, "invalid": invalid_matches}

        logger.info(
            f"Validated {len(matches)} matches: {len(valid_matches)} valid, {len(invalid_matches)} invalid"
        )
        return result
