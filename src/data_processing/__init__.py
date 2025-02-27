"""
Data processing module for soccer match statistics.

This package contains utilities for loading, validating, normalizing, and standardizing 
soccer match data for use in prediction models and game simulation.
"""

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Import main components for easy access
from .pipelines.extraction_pipeline import ExtractionPipeline
from .extractors.json_extractor import JsonExtractor
from .extractors.match_extractor import MatchExtractor
from .transformers.stat_normalizer import StatNormalizer
from .transformers.team_standardizer import TeamStandardizer
from .validators.schema_validator import SchemaValidator
from .validators.consistency_checker import ConsistencyChecker

__all__ = [
    "ExtractionPipeline",
    "JsonExtractor",
    "MatchExtractor",
    "StatNormalizer",
    "TeamStandardizer",
    "SchemaValidator",
    "ConsistencyChecker",
]
