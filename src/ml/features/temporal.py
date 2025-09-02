"""
Temporal feature engineering components for soccer match analysis.

Provides sophisticated time-based features including rolling statistics,
momentum indicators, form analysis, and streak detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from src.config.logging_config import get_logger
from ..core.interfaces import FeatureEngineer
from ..core.exceptions import FeatureEngineeringError


@dataclass
class TemporalConfig:
    """Configuration for temporal feature engineering."""
    rolling_windows: List[int] = None
    form_windows: List[int] = None
    momentum_decay: float = 0.95
    min_matches_required: int = 5
    season_boundary_days: int = 120
    
    def __post_init__(self):
        if self.rolling_windows is None:
            self.rolling_windows = [3, 5, 10, 20]
        if self.form_windows is None:
            self.form_windows = [5, 10, 15]


class RollingStatsCalculator:
    """
    Calculates rolling statistics for team performance.
    
    Following Single Responsibility Principle - focused only on rolling calculations.
    """
    
    def __init__(self, windows: List[int] = None):
        self.windows = windows or [3, 5, 10, 20]
        self.logger = get_logger("RollingStatsCalculator")
    
    def calculate_team_rolling_stats(self, 
                                   team_matches: pd.DataFrame,
                                   team_name: str,
                                   reference_date: datetime) -> Dict[str, float]:
        """
        Calculate rolling statistics for a team up to a reference date.
        
        Args:
            team_matches: DataFrame with team's historical matches
            team_name: Name of the team
            reference_date: Date to calculate stats up to (exclusive)
            
        Returns:
            Dictionary of rolling statistics
        """
        try:
            # Filter matches before reference date
            historical_matches = team_matches[
                team_matches['date'] < reference_date
            ].sort_values('date')
            
            if len(historical_matches) < min(self.windows):
                self.logger.warning(
                    f"Insufficient matches for {team_name}: {len(historical_matches)}"
                )
                return self._get_default_stats()
            
            stats = {}
            
            # Calculate for each rolling window
            for window in self.windows:
                window_matches = historical_matches.tail(window)
                window_stats = self._calculate_window_stats(window_matches, team_name)
                
                # Add window suffix to stat names
                for stat_name, value in window_stats.items():
                    stats[f"{stat_name}_{window}g"] = value
            
            return stats
            
        except Exception as e:
            raise FeatureEngineeringError(
                f"Failed to calculate rolling stats for {team_name}",
                feature_type="rolling_stats",
                error=str(e)
            )
    
    def _calculate_window_stats(self, matches: pd.DataFrame, team_name: str) -> Dict[str, float]:
        """Calculate statistics for a specific window of matches."""
        if matches.empty:
            return self._get_default_stats()
        
        stats = {}
        
        # Determine if team was home or away for each match
        home_mask = matches['home_team'] == team_name
        away_mask = matches['away_team'] == team_name
        
        # Goals scored and conceded
        goals_scored = (
            matches.loc[home_mask, 'home_score'].sum() + 
            matches.loc[away_mask, 'away_score'].sum()
        )
        goals_conceded = (
            matches.loc[home_mask, 'away_score'].sum() + 
            matches.loc[away_mask, 'home_score'].sum()
        )
        
        num_matches = len(matches)
        
        # Basic statistics
        stats['goals_per_game'] = goals_scored / num_matches
        stats['goals_conceded_per_game'] = goals_conceded / num_matches
        stats['goal_difference_per_game'] = (goals_scored - goals_conceded) / num_matches
        
        # Win/Draw/Loss record
        wins = 0
        draws = 0
        losses = 0
        
        for _, match in matches.iterrows():
            if match['home_team'] == team_name:
                if match['home_score'] > match['away_score']:
                    wins += 1
                elif match['home_score'] == match['away_score']:
                    draws += 1
                else:
                    losses += 1
            else:  # away team
                if match['away_score'] > match['home_score']:
                    wins += 1
                elif match['away_score'] == match['home_score']:
                    draws += 1
                else:
                    losses += 1
        
        stats['win_rate'] = wins / num_matches
        stats['draw_rate'] = draws / num_matches
        stats['loss_rate'] = losses / num_matches
        stats['points_per_game'] = (wins * 3 + draws) / num_matches
        
        # Clean sheet and scoring statistics
        clean_sheets = (
            (matches.loc[home_mask, 'away_score'] == 0).sum() +
            (matches.loc[away_mask, 'home_score'] == 0).sum()
        )
        failed_to_score = (
            (matches.loc[home_mask, 'home_score'] == 0).sum() +
            (matches.loc[away_mask, 'away_score'] == 0).sum()
        )
        
        stats['clean_sheet_rate'] = clean_sheets / num_matches
        stats['failed_to_score_rate'] = failed_to_score / num_matches
        
        # High/low scoring games
        total_goals_per_match = matches['home_score'] + matches['away_score']
        stats['over_2_5_rate'] = (total_goals_per_match > 2.5).mean()
        stats['under_1_5_rate'] = (total_goals_per_match < 1.5).mean()
        
        return stats
    
    def _get_default_stats(self) -> Dict[str, float]:
        """Return default statistics when insufficient data."""
        return {
            'goals_per_game': 1.0,
            'goals_conceded_per_game': 1.0,
            'goal_difference_per_game': 0.0,
            'win_rate': 0.33,
            'draw_rate': 0.33,
            'loss_rate': 0.33,
            'points_per_game': 1.0,
            'clean_sheet_rate': 0.25,
            'failed_to_score_rate': 0.25,
            'over_2_5_rate': 0.5,
            'under_1_5_rate': 0.25
        }


class MomentumAnalyzer:
    """
    Analyzes team momentum using exponentially weighted statistics.
    
    Momentum gives more weight to recent matches, capturing current form.
    """
    
    def __init__(self, decay_factor: float = 0.95):
        self.decay_factor = decay_factor
        self.logger = get_logger("MomentumAnalyzer")
    
    def calculate_momentum(self, 
                          team_matches: pd.DataFrame,
                          team_name: str,
                          reference_date: datetime) -> Dict[str, float]:
        """
        Calculate momentum indicators for a team.
        
        Uses exponential decay to weight recent matches more heavily.
        """
        try:
            # Filter and sort matches
            historical_matches = team_matches[
                team_matches['date'] < reference_date
            ].sort_values('date')
            
            if len(historical_matches) < 3:
                return self._get_default_momentum()
            
            # Calculate match results and weights
            results = []
            weights = []
            
            for i, (_, match) in enumerate(historical_matches.iterrows()):
                # Calculate match result (points earned)
                if match['home_team'] == team_name:
                    if match['home_score'] > match['away_score']:
                        result = 3  # Win
                    elif match['home_score'] == match['away_score']:
                        result = 1  # Draw
                    else:
                        result = 0  # Loss
                else:
                    if match['away_score'] > match['home_score']:
                        result = 3  # Win
                    elif match['away_score'] == match['home_score']:
                        result = 1  # Draw
                    else:
                        result = 0  # Loss
                
                results.append(result)
                # More recent matches get higher weight
                weight = self.decay_factor ** (len(historical_matches) - 1 - i)
                weights.append(weight)
            
            # Calculate weighted statistics
            results = np.array(results)
            weights = np.array(weights)
            
            momentum_stats = {
                'momentum_points': np.average(results, weights=weights),
                'momentum_trend': self._calculate_trend(results[-10:]),  # Last 10 matches
                'recent_form_strength': self._calculate_recent_strength(results[-5:]),
                'consistency': self._calculate_consistency(results),
                'volatility': np.std(results[-10:]) if len(results) >= 10 else 1.0
            }
            
            return momentum_stats
            
        except Exception as e:
            raise FeatureEngineeringError(
                f"Failed to calculate momentum for {team_name}",
                feature_type="momentum",
                error=str(e)
            )
    
    def _calculate_trend(self, recent_results: np.ndarray) -> float:
        """Calculate trend in recent results using linear regression slope."""
        if len(recent_results) < 3:
            return 0.0
        
        x = np.arange(len(recent_results))
        # Simple linear regression slope
        slope = np.polyfit(x, recent_results, 1)[0]
        return float(slope)
    
    def _calculate_recent_strength(self, recent_results: np.ndarray) -> float:
        """Calculate strength based on very recent results."""
        if len(recent_results) == 0:
            return 1.0
        
        # Weight recent results more heavily
        weights = np.array([0.5, 0.7, 0.9, 1.0, 1.2][-len(recent_results):])
        return np.average(recent_results, weights=weights) / 3.0  # Normalize to [0,1]
    
    def _calculate_consistency(self, results: np.ndarray) -> float:
        """Calculate consistency (inverse of variance)."""
        if len(results) < 3:
            return 0.5
        
        variance = np.var(results)
        # Higher consistency = lower variance
        return 1.0 / (1.0 + variance)
    
    def _get_default_momentum(self) -> Dict[str, float]:
        """Return default momentum values."""
        return {
            'momentum_points': 1.0,
            'momentum_trend': 0.0,
            'recent_form_strength': 0.5,
            'consistency': 0.5,
            'volatility': 1.0
        }


class FormCalculator:
    """
    Calculates team form over different time periods.
    
    Form is more nuanced than simple win/loss - considers opponent strength,
    margin of victory, and context.
    """
    
    def __init__(self, form_windows: List[int] = None):
        self.form_windows = form_windows or [5, 10, 15]
        self.logger = get_logger("FormCalculator")
    
    def calculate_form(self,
                      team_matches: pd.DataFrame,
                      team_name: str,
                      reference_date: datetime) -> Dict[str, float]:
        """Calculate comprehensive form indicators."""
        try:
            historical_matches = team_matches[
                team_matches['date'] < reference_date
            ].sort_values('date')
            
            if len(historical_matches) < min(self.form_windows):
                return self._get_default_form()
            
            form_stats = {}
            
            for window in self.form_windows:
                recent_matches = historical_matches.tail(window)
                window_form = self._calculate_window_form(recent_matches, team_name)
                
                # Add window suffix
                for stat_name, value in window_form.items():
                    form_stats[f"{stat_name}_{window}f"] = value
            
            return form_stats
            
        except Exception as e:
            raise FeatureEngineeringError(
                f"Failed to calculate form for {team_name}",
                feature_type="form",
                error=str(e)
            )
    
    def _calculate_window_form(self, matches: pd.DataFrame, team_name: str) -> Dict[str, float]:
        """Calculate form for a specific window."""
        if matches.empty:
            return {'form_rating': 0.5}
        
        form_scores = []
        
        for _, match in matches.iterrows():
            score = self._calculate_match_form_score(match, team_name)
            form_scores.append(score)
        
        form_scores = np.array(form_scores)
        
        return {
            'form_rating': np.mean(form_scores),
            'form_trend': self._calculate_trend(form_scores),
            'form_stability': 1.0 - np.std(form_scores),  # Higher = more stable
            'peak_form': np.max(form_scores),
            'worst_form': np.min(form_scores)
        }
    
    def _calculate_match_form_score(self, match: pd.Series, team_name: str) -> float:
        """
        Calculate form score for a single match.
        
        Considers result, goal difference, and opponent strength.
        """
        if match['home_team'] == team_name:
            goals_for = match['home_score']
            goals_against = match['away_score']
            is_home = True
        else:
            goals_for = match['away_score']
            goals_against = match['home_score']
            is_home = False
        
        goal_diff = goals_for - goals_against
        
        # Base score from result
        if goal_diff > 0:
            base_score = 1.0  # Win
        elif goal_diff == 0:
            base_score = 0.5  # Draw
        else:
            base_score = 0.0  # Loss
        
        # Adjust for goal difference (margin of victory/defeat)
        goal_diff_bonus = np.tanh(goal_diff / 3.0) * 0.3  # Max ±0.3 adjustment
        
        # Home advantage adjustment (slightly lower expectations at home)
        home_adjustment = -0.05 if is_home else 0.05
        
        total_score = base_score + goal_diff_bonus + home_adjustment
        
        # Ensure score is between 0 and 1
        return np.clip(total_score, 0.0, 1.0)
    
    def _calculate_trend(self, form_scores: np.ndarray) -> float:
        """Calculate trend in form scores."""
        if len(form_scores) < 3:
            return 0.0
        
        x = np.arange(len(form_scores))
        slope = np.polyfit(x, form_scores, 1)[0]
        return float(slope)
    
    def _get_default_form(self) -> Dict[str, float]:
        """Return default form values."""
        return {
            'form_rating': 0.5,
            'form_trend': 0.0,
            'form_stability': 0.5,
            'peak_form': 0.5,
            'worst_form': 0.5
        }


class StreakDetector:
    """
    Detects and analyzes winning/losing streaks and patterns.
    
    Psychological momentum can be significant in sports.
    """
    
    def __init__(self):
        self.logger = get_logger("StreakDetector")
    
    def detect_streaks(self,
                      team_matches: pd.DataFrame,
                      team_name: str,
                      reference_date: datetime) -> Dict[str, float]:
        """Detect current streaks and historical patterns."""
        try:
            historical_matches = team_matches[
                team_matches['date'] < reference_date
            ].sort_values('date')
            
            if len(historical_matches) < 3:
                return self._get_default_streaks()
            
            # Get match results
            results = []
            for _, match in historical_matches.iterrows():
                result = self._get_match_result(match, team_name)
                results.append(result)
            
            streak_stats = {
                'current_win_streak': self._get_current_streak(results, 'W'),
                'current_unbeaten_streak': self._get_current_unbeaten_streak(results),
                'current_loss_streak': self._get_current_streak(results, 'L'),
                'longest_win_streak': self._get_longest_streak(results, 'W'),
                'longest_unbeaten_streak': self._get_longest_unbeaten_streak(results),
                'streak_volatility': self._calculate_streak_volatility(results),
                'recent_streak_strength': self._calculate_recent_streak_strength(results)
            }
            
            return streak_stats
            
        except Exception as e:
            raise FeatureEngineeringError(
                f"Failed to detect streaks for {team_name}",
                feature_type="streaks",
                error=str(e)
            )
    
    def _get_match_result(self, match: pd.Series, team_name: str) -> str:
        """Get match result from team's perspective."""
        if match['home_team'] == team_name:
            if match['home_score'] > match['away_score']:
                return 'W'
            elif match['home_score'] == match['away_score']:
                return 'D'
            else:
                return 'L'
        else:
            if match['away_score'] > match['home_score']:
                return 'W'
            elif match['away_score'] == match['home_score']:
                return 'D'
            else:
                return 'L'
    
    def _get_current_streak(self, results: List[str], target_result: str) -> int:
        """Get current streak of specific result type."""
        if not results:
            return 0
        
        streak = 0
        for result in reversed(results):
            if result == target_result:
                streak += 1
            else:
                break
        
        return streak
    
    def _get_current_unbeaten_streak(self, results: List[str]) -> int:
        """Get current unbeaten streak (wins + draws)."""
        if not results:
            return 0
        
        streak = 0
        for result in reversed(results):
            if result in ['W', 'D']:
                streak += 1
            else:
                break
        
        return streak
    
    def _get_longest_streak(self, results: List[str], target_result: str) -> int:
        """Get longest streak of specific result type."""
        if not results:
            return 0
        
        max_streak = 0
        current_streak = 0
        
        for result in results:
            if result == target_result:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def _get_longest_unbeaten_streak(self, results: List[str]) -> int:
        """Get longest unbeaten streak."""
        if not results:
            return 0
        
        max_streak = 0
        current_streak = 0
        
        for result in results:
            if result in ['W', 'D']:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def _calculate_streak_volatility(self, results: List[str]) -> float:
        """Calculate how volatile the team's results are."""
        if len(results) < 5:
            return 0.5
        
        # Convert results to numeric
        numeric_results = [2 if r == 'W' else 1 if r == 'D' else 0 for r in results]
        volatility = np.std(numeric_results) / 2.0  # Normalize
        
        return float(volatility)
    
    def _calculate_recent_streak_strength(self, results: List[str]) -> float:
        """Calculate strength of recent streaks."""
        if len(results) < 5:
            return 0.5
        
        recent_results = results[-5:]
        
        # Weight different result types
        weights = {'W': 1.0, 'D': 0.5, 'L': 0.0}
        strength = sum(weights[result] for result in recent_results) / len(recent_results)
        
        return strength
    
    def _get_default_streaks(self) -> Dict[str, float]:
        """Return default streak values."""
        return {
            'current_win_streak': 0,
            'current_unbeaten_streak': 0,
            'current_loss_streak': 0,
            'longest_win_streak': 0,
            'longest_unbeaten_streak': 0,
            'streak_volatility': 0.5,
            'recent_streak_strength': 0.5
        }


class TemporalFeaturesEngine(FeatureEngineer):
    """
    Main engine for temporal feature engineering.
    
    Orchestrates all temporal feature calculators using Facade pattern.
    """
    
    def __init__(self, config: TemporalConfig = None):
        self.config = config or TemporalConfig()
        self.logger = get_logger("TemporalFeaturesEngine")
        
        # Initialize calculators
        self.rolling_calculator = RollingStatsCalculator(self.config.rolling_windows)
        self.momentum_analyzer = MomentumAnalyzer(self.config.momentum_decay)
        self.form_calculator = FormCalculator(self.config.form_windows)
        self.streak_detector = StreakDetector()
        
        self._feature_names = []
        self._feature_importance = {}
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create all temporal features for match data."""
        try:
            self.logger.info(f"Creating temporal features for {len(data)} matches")
            
            # Ensure required columns exist
            required_columns = ['date', 'home_team', 'away_team', 'home_score', 'away_score']
            missing_columns = set(required_columns) - set(data.columns)
            
            if missing_columns:
                raise FeatureEngineeringError(
                    f"Missing required columns: {missing_columns}",
                    feature_type="temporal"
                )
            
            # Convert date column if needed
            if not pd.api.types.is_datetime64_any_dtype(data['date']):
                data['date'] = pd.to_datetime(data['date'])
            
            features_data = data.copy()
            
            # Create temporal features for each match
            for idx, match in data.iterrows():
                match_features = self._create_match_temporal_features(
                    match, data
                )
                
                # Add features to the row
                for feature_name, value in match_features.items():
                    features_data.loc[idx, feature_name] = value
            
            # Update feature names
            temporal_columns = set(features_data.columns) - set(data.columns)
            self._feature_names = sorted(temporal_columns)
            
            self.logger.info(f"Created {len(self._feature_names)} temporal features")
            return features_data
            
        except Exception as e:
            if isinstance(e, FeatureEngineeringError):
                raise
            raise FeatureEngineeringError(
                "Failed to create temporal features",
                feature_type="temporal",
                error=str(e)
            )
    
    def get_feature_names(self) -> List[str]:
        """Return list of temporal feature names."""
        return self._feature_names.copy()
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Return feature importance scores if available."""
        return self._feature_importance.copy() if self._feature_importance else None
    
    def _create_match_temporal_features(self, 
                                      match: pd.Series,
                                      all_data: pd.DataFrame) -> Dict[str, float]:
        """Create temporal features for a single match."""
        home_team = match['home_team']
        away_team = match['away_team']
        match_date = match['date']
        
        # Get historical data for both teams
        home_matches = all_data[
            (all_data['home_team'] == home_team) | (all_data['away_team'] == home_team)
        ]
        away_matches = all_data[
            (all_data['home_team'] == away_team) | (all_data['away_team'] == away_team)
        ]
        
        features = {}
        
        # Calculate features for home team
        home_rolling = self.rolling_calculator.calculate_team_rolling_stats(
            home_matches, home_team, match_date
        )
        home_momentum = self.momentum_analyzer.calculate_momentum(
            home_matches, home_team, match_date
        )
        home_form = self.form_calculator.calculate_form(
            home_matches, home_team, match_date
        )
        home_streaks = self.streak_detector.detect_streaks(
            home_matches, home_team, match_date
        )
        
        # Add home prefix
        for name, value in home_rolling.items():
            features[f"home_{name}"] = value
        for name, value in home_momentum.items():
            features[f"home_{name}"] = value
        for name, value in home_form.items():
            features[f"home_{name}"] = value
        for name, value in home_streaks.items():
            features[f"home_{name}"] = value
        
        # Calculate features for away team
        away_rolling = self.rolling_calculator.calculate_team_rolling_stats(
            away_matches, away_team, match_date
        )
        away_momentum = self.momentum_analyzer.calculate_momentum(
            away_matches, away_team, match_date
        )
        away_form = self.form_calculator.calculate_form(
            away_matches, away_team, match_date
        )
        away_streaks = self.streak_detector.detect_streaks(
            away_matches, away_team, match_date
        )
        
        # Add away prefix
        for name, value in away_rolling.items():
            features[f"away_{name}"] = value
        for name, value in away_momentum.items():
            features[f"away_{name}"] = value
        for name, value in away_form.items():
            features[f"away_{name}"] = value
        for name, value in away_streaks.items():
            features[f"away_{name}"] = value
        
        # Calculate differential features
        features.update(self._calculate_differential_features(
            home_rolling, away_rolling,
            home_momentum, away_momentum,
            home_form, away_form
        ))
        
        return features
    
    def _calculate_differential_features(self,
                                       home_rolling: Dict[str, float],
                                       away_rolling: Dict[str, float],
                                       home_momentum: Dict[str, float],
                                       away_momentum: Dict[str, float],
                                       home_form: Dict[str, float],
                                       away_form: Dict[str, float]) -> Dict[str, float]:
        """Calculate differential features between home and away teams."""
        differentials = {}
        
        # Rolling stats differentials
        for key in home_rolling:
            if key in away_rolling:
                differentials[f"diff_{key}"] = home_rolling[key] - away_rolling[key]
        
        # Momentum differentials
        for key in home_momentum:
            if key in away_momentum:
                differentials[f"diff_{key}"] = home_momentum[key] - away_momentum[key]
        
        # Form differentials (only for matching windows)
        home_form_keys = {k.split('_')[0] + '_' + k.split('_')[1]: k for k in home_form.keys()}
        away_form_keys = {k.split('_')[0] + '_' + k.split('_')[1]: k for k in away_form.keys()}
        
        for base_key in home_form_keys:
            if base_key in away_form_keys:
                home_key = home_form_keys[base_key]
                away_key = away_form_keys[base_key]
                differentials[f"diff_{base_key}"] = home_form[home_key] - away_form[away_key]
        
        return differentials