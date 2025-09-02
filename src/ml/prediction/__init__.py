"""
Match prediction module for real-time soccer outcome predictions.

Provides high-level interfaces for predicting match outcomes using
team names as input, automatically generating features and managing
model ensemble predictions.
"""

from .match_predictor import MatchPredictor

__all__ = ['MatchPredictor']