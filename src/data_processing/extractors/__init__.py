"""
Extractors for loading and parsing soccer match data from various sources.
"""

from .json_extractor import JsonExtractor
from .match_extractor import MatchExtractor

__all__ = ["JsonExtractor", "MatchExtractor"]
