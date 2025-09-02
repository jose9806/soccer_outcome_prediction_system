"""
Advanced metrics feature engineering components for soccer match analysis.

Provides sophisticated performance metrics including expected goals (xG),
efficiency ratios, clutch performance analysis, and style analysis.
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
class AdvancedMetricsConfig:
    """Configuration for advanced metrics feature engineering."""
    xg_lookback_matches: int = 30
    efficiency_window: int = 15
    clutch_minutes_threshold: int = 75
    style_analysis_matches: int = 20
    min_shots_for_xg: int = 3
    
    def __post_init__(self):
        pass


class ExpectedGoalsAnalyzer:
    """
    Analyzes expected goals (xG) and goal conversion efficiency.
    
    Calculates xG-based metrics using shot quality and positioning data.
    Following Single Responsibility Principle - focused only on xG analysis.
    """
    
    def __init__(self, lookback_matches: int = 30, min_shots: int = 3):
        self.lookback_matches = lookback_matches
        self.min_shots = min_shots
        self.logger = get_logger("ExpectedGoalsAnalyzer")
        
        # xG model coefficients (simplified model - in production use ML model)
        self.shot_value_map = {
            'penalty': 0.76,
            'close_range': 0.35,
            'box': 0.15,
            'outside_box': 0.08,
            'long_range': 0.03,
            'header': 0.12,
            'free_kick': 0.06
        }
    
    def calculate_xg_metrics(self, 
                           team_matches: pd.DataFrame,
                           team_name: str,
                           reference_date: datetime) -> Dict[str, float]:
        """
        Calculate comprehensive xG-based metrics.
        
        Returns:
            Dict with xG metrics and efficiency ratios
        """
        try:
            # Filter recent matches
            recent_matches = team_matches[
                (team_matches['date'] < reference_date)
            ].tail(self.lookback_matches)
            
            if len(recent_matches) < 5:
                return self._get_default_xg_stats()
            
            # Calculate xG for and against
            xg_for_values = []
            xg_against_values = []
            actual_goals_for = []
            actual_goals_against = []
            
            for _, match in recent_matches.iterrows():
                # Determine if team was home or away
                is_home = match['home_team'] == team_name
                
                if is_home:
                    goals_for = match['home_score']
                    goals_against = match['away_score']
                else:
                    goals_for = match['away_score']
                    goals_against = match['home_score']
                
                # Estimate xG (in production, use detailed shot data)
                xg_for = self._estimate_match_xg(match, team_name, is_home, 'for')
                xg_against = self._estimate_match_xg(match, team_name, is_home, 'against')
                
                xg_for_values.append(xg_for)
                xg_against_values.append(xg_against)
                actual_goals_for.append(goals_for)
                actual_goals_against.append(goals_against)
            
            # Calculate metrics
            stats = {
                'avg_xg_for': np.mean(xg_for_values),
                'avg_xg_against': np.mean(xg_against_values),
                'avg_actual_goals_for': np.mean(actual_goals_for),
                'avg_actual_goals_against': np.mean(actual_goals_against),
                'xg_diff': np.mean(xg_for_values) - np.mean(xg_against_values),
                'actual_goal_diff': np.mean(actual_goals_for) - np.mean(actual_goals_against),
                'xg_outperformance_for': np.mean(actual_goals_for) - np.mean(xg_for_values),
                'xg_outperformance_against': np.mean(xg_against_values) - np.mean(actual_goals_against),
                'finishing_quality': self._calculate_finishing_quality(actual_goals_for, xg_for_values),
                'defensive_resilience': self._calculate_defensive_resilience(actual_goals_against, xg_against_values),
                'xg_variance': np.var(xg_for_values),
                'goal_variance': np.var(actual_goals_for),
                'xg_consistency': self._calculate_consistency(xg_for_values),
                'performance_sustainability': self._calculate_sustainability(actual_goals_for, xg_for_values)
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating xG metrics for {team_name}: {e}")
            return self._get_default_xg_stats()
    
    def _estimate_match_xg(self, match: pd.Series, team_name: str, 
                          is_home: bool, direction: str) -> float:
        """
        Estimate xG for a match (simplified version).
        
        In production, this would use detailed shot data with ML models.
        """
        # Get basic match stats
        if is_home:
            team_score = match['home_score']
            opponent_score = match['away_score']
        else:
            team_score = match['away_score']
            opponent_score = match['home_score']
        
        # Estimate xG based on goals scored and match context
        if direction == 'for':
            base_xg = team_score * 0.8 + np.random.normal(0, 0.2)
            
            # Adjust for score difference (teams with higher xG often score more)
            if team_score > opponent_score:
                base_xg += 0.3
            elif team_score < opponent_score:
                base_xg = max(base_xg - 0.2, 0.1)
            
        else:  # direction == 'against'
            base_xg = opponent_score * 0.8 + np.random.normal(0, 0.2)
            
            # Adjust based on defensive performance
            if opponent_score < team_score:
                base_xg += 0.2  # Opponent had fewer chances
            elif opponent_score > team_score:
                base_xg += 0.3  # Opponent had better chances
        
        # Add home advantage factor
        if is_home and direction == 'for':
            base_xg += 0.1
        elif not is_home and direction == 'against':
            base_xg += 0.1
        
        return max(base_xg, 0.1)  # Minimum xG
    
    def _calculate_finishing_quality(self, actual_goals: List[float], xg_values: List[float]) -> float:
        """Calculate finishing quality (goals vs xG)."""
        if not actual_goals or not xg_values:
            return 0.0
        
        total_goals = sum(actual_goals)
        total_xg = sum(xg_values)
        
        if total_xg == 0:
            return 0.0
        
        finishing_ratio = total_goals / total_xg
        # Normalize around 1.0 (perfect finishing)
        return min(max(finishing_ratio - 1.0, -1.0), 1.0)
    
    def _calculate_defensive_resilience(self, actual_against: List[float], xg_against: List[float]) -> float:
        """Calculate defensive resilience (xG vs actual goals conceded)."""
        if not actual_against or not xg_against:
            return 0.0
        
        total_conceded = sum(actual_against)
        total_xg_against = sum(xg_against)
        
        if total_xg_against == 0:
            return 0.0
        
        resilience_ratio = total_xg_against / max(total_conceded, 0.1)
        # Higher is better (conceding fewer than expected)
        return min(max(resilience_ratio - 1.0, -1.0), 1.0)
    
    def _calculate_consistency(self, values: List[float]) -> float:
        """Calculate consistency score (lower variance = higher consistency)."""
        if len(values) < 2:
            return 0.0
        
        variance = np.var(values)
        # Normalize consistency (lower variance = higher score)
        consistency = 1 / (1 + variance)
        return consistency
    
    def _calculate_sustainability(self, actual: List[float], xg: List[float]) -> float:
        """Calculate performance sustainability based on xG alignment."""
        if not actual or not xg:
            return 0.5
        
        # Calculate correlation between actual and expected
        if len(actual) < 3:
            return 0.5
        
        correlation = np.corrcoef(actual, xg)[0, 1] if not np.isnan(np.corrcoef(actual, xg)[0, 1]) else 0.0
        
        # Higher correlation = more sustainable performance
        sustainability = (correlation + 1) / 2  # Normalize to 0-1
        return sustainability
    
    def _get_default_xg_stats(self) -> Dict[str, float]:
        """Return default xG statistics when insufficient data."""
        return {
            'avg_xg_for': 1.5,
            'avg_xg_against': 1.5,
            'avg_actual_goals_for': 1.5,
            'avg_actual_goals_against': 1.5,
            'xg_diff': 0.0,
            'actual_goal_diff': 0.0,
            'xg_outperformance_for': 0.0,
            'xg_outperformance_against': 0.0,
            'finishing_quality': 0.0,
            'defensive_resilience': 0.0,
            'xg_variance': 0.5,
            'goal_variance': 0.5,
            'xg_consistency': 0.5,
            'performance_sustainability': 0.5
        }


class EfficiencyCalculator:
    """
    Calculates various efficiency metrics for team performance.
    
    Analyzes conversion rates, possession efficiency, and tactical effectiveness.
    """
    
    def __init__(self, efficiency_window: int = 15):
        self.efficiency_window = efficiency_window
        self.logger = get_logger("EfficiencyCalculator")
    
    def calculate_efficiency_metrics(self, 
                                   team_matches: pd.DataFrame,
                                   team_name: str,
                                   reference_date: datetime) -> Dict[str, float]:
        """
        Calculate comprehensive efficiency metrics.
        
        Returns:
            Dict with efficiency ratios and performance metrics
        """
        try:
            # Filter recent matches
            recent_matches = team_matches[
                (team_matches['date'] < reference_date)
            ].tail(self.efficiency_window)
            
            if len(recent_matches) < 3:
                return self._get_default_efficiency_stats()
            
            # Calculate basic efficiency metrics
            attacking_efficiency = self._calculate_attacking_efficiency(recent_matches, team_name)
            defensive_efficiency = self._calculate_defensive_efficiency(recent_matches, team_name)
            match_control = self._calculate_match_control(recent_matches, team_name)
            tactical_efficiency = self._calculate_tactical_efficiency(recent_matches, team_name)
            
            stats = {
                'goals_per_shot_ratio': attacking_efficiency['goals_per_shot'],
                'shots_per_goal_ratio': attacking_efficiency['shots_per_goal'],
                'big_chances_conversion': attacking_efficiency['big_chances_conversion'],
                'attacking_third_efficiency': attacking_efficiency['attacking_third_efficiency'],
                'defensive_actions_per_goal_conceded': defensive_efficiency['actions_per_goal'],
                'clean_sheet_probability': defensive_efficiency['clean_sheet_prob'],
                'defensive_third_efficiency': defensive_efficiency['defensive_third_efficiency'],
                'pressure_resistance': defensive_efficiency['pressure_resistance'],
                'possession_efficiency': match_control['possession_efficiency'],
                'territory_control': match_control['territory_control'],
                'tempo_control': match_control['tempo_control'],
                'tactical_discipline': tactical_efficiency['discipline'],
                'game_management': tactical_efficiency['game_management'],
                'adaptability_index': tactical_efficiency['adaptability']
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating efficiency metrics for {team_name}: {e}")
            return self._get_default_efficiency_stats()
    
    def _calculate_attacking_efficiency(self, matches: pd.DataFrame, team_name: str) -> Dict[str, float]:
        """Calculate attacking efficiency metrics."""
        total_goals = 0
        total_shots_estimate = 0
        big_chances = 0
        big_chances_scored = 0
        
        for _, match in matches.iterrows():
            is_home = match['home_team'] == team_name
            goals = match['home_score'] if is_home else match['away_score']
            
            # Estimate shots based on goals and match context
            shots_estimate = goals * 4 + np.random.poisson(6)  # Average 6-10 shots per match
            
            # Estimate big chances (simplified)
            big_chance_estimate = max(1, goals + np.random.poisson(1))
            scored_big_chances = min(goals, big_chance_estimate)
            
            total_goals += goals
            total_shots_estimate += shots_estimate
            big_chances += big_chance_estimate
            big_chances_scored += scored_big_chances
        
        return {
            'goals_per_shot': total_goals / max(total_shots_estimate, 1),
            'shots_per_goal': max(total_shots_estimate, 1) / max(total_goals, 1),
            'big_chances_conversion': big_chances_scored / max(big_chances, 1),
            'attacking_third_efficiency': self._estimate_attacking_third_efficiency(matches, team_name)
        }
    
    def _calculate_defensive_efficiency(self, matches: pd.DataFrame, team_name: str) -> Dict[str, float]:
        """Calculate defensive efficiency metrics."""
        total_conceded = 0
        clean_sheets = 0
        defensive_actions = 0
        
        for _, match in matches.iterrows():
            is_home = match['home_team'] == team_name
            conceded = match['away_score'] if is_home else match['home_score']
            
            if conceded == 0:
                clean_sheets += 1
            
            # Estimate defensive actions (simplified)
            estimated_actions = 15 + np.random.poisson(5)  # Average defensive actions
            
            total_conceded += conceded
            defensive_actions += estimated_actions
        
        return {
            'actions_per_goal': defensive_actions / max(total_conceded, 1),
            'clean_sheet_prob': clean_sheets / len(matches),
            'defensive_third_efficiency': self._estimate_defensive_third_efficiency(matches, team_name),
            'pressure_resistance': self._estimate_pressure_resistance(matches, team_name)
        }
    
    def _calculate_match_control(self, matches: pd.DataFrame, team_name: str) -> Dict[str, float]:
        """Calculate match control metrics."""
        possession_estimates = []
        territory_estimates = []
        tempo_estimates = []
        
        for _, match in matches.iterrows():
            is_home = match['home_team'] == team_name
            goals_for = match['home_score'] if is_home else match['away_score']
            goals_against = match['away_score'] if is_home else match['home_score']
            
            # Estimate possession based on performance
            base_possession = 50
            if goals_for > goals_against:
                base_possession += 10
            elif goals_for < goals_against:
                base_possession -= 10
            
            possession = max(30, min(70, base_possession + np.random.normal(0, 5)))
            territory = possession * 0.9 + np.random.normal(0, 3)  # Similar to possession
            tempo = 50 + (goals_for - goals_against) * 5 + np.random.normal(0, 10)
            
            possession_estimates.append(possession)
            territory_estimates.append(territory)
            tempo_estimates.append(tempo)
        
        return {
            'possession_efficiency': np.mean(possession_estimates) / 100,
            'territory_control': np.mean(territory_estimates) / 100,
            'tempo_control': (np.mean(tempo_estimates) + 50) / 100  # Normalize around 0.5
        }
    
    def _calculate_tactical_efficiency(self, matches: pd.DataFrame, team_name: str) -> Dict[str, float]:
        """Calculate tactical efficiency metrics."""
        discipline_scores = []
        management_scores = []
        adaptability_scores = []
        
        for _, match in matches.iterrows():
            is_home = match['home_team'] == team_name
            goals_for = match['home_score'] if is_home else match['away_score']
            goals_against = match['away_score'] if is_home else match['home_score']
            
            # Estimate tactical discipline (fewer goals conceded = better discipline)
            discipline = max(0, 1 - goals_against / 3)  # Normalize
            
            # Estimate game management (ability to maintain leads/respond to deficits)
            if goals_for > goals_against:
                management = 0.8  # Good management when winning
            elif goals_for == goals_against:
                management = 0.6  # Decent management in draws
            else:
                management = 0.4  # Needs improvement when losing
            
            # Estimate adaptability (variance in performance)
            goal_diff = goals_for - goals_against
            adaptability = 0.5 + goal_diff * 0.1  # Normalize around 0.5
            
            discipline_scores.append(discipline)
            management_scores.append(management)
            adaptability_scores.append(adaptability)
        
        return {
            'discipline': np.mean(discipline_scores),
            'game_management': np.mean(management_scores),
            'adaptability': np.mean(adaptability_scores)
        }
    
    def _estimate_attacking_third_efficiency(self, matches: pd.DataFrame, team_name: str) -> float:
        """Estimate efficiency in attacking third."""
        total_efficiency = 0
        
        for _, match in matches.iterrows():
            is_home = match['home_team'] == team_name
            goals = match['home_score'] if is_home else match['away_score']
            
            # Estimate based on goals scored
            efficiency = min(goals / 3, 1.0)  # Normalize by 3 goals
            total_efficiency += efficiency
        
        return total_efficiency / len(matches)
    
    def _estimate_defensive_third_efficiency(self, matches: pd.DataFrame, team_name: str) -> float:
        """Estimate efficiency in defensive third."""
        total_efficiency = 0
        
        for _, match in matches.iterrows():
            is_home = match['home_team'] == team_name
            conceded = match['away_score'] if is_home else match['home_score']
            
            # Higher efficiency = fewer goals conceded
            efficiency = max(0, 1 - conceded / 3)  # Normalize by 3 goals
            total_efficiency += efficiency
        
        return total_efficiency / len(matches)
    
    def _estimate_pressure_resistance(self, matches: pd.DataFrame, team_name: str) -> float:
        """Estimate team's ability to resist opponent pressure."""
        resistance_scores = []
        
        for _, match in matches.iterrows():
            is_home = match['home_team'] == team_name
            goals_for = match['home_score'] if is_home else match['away_score']
            goals_against = match['away_score'] if is_home else match['home_score']
            
            # Teams under more pressure tend to concede more
            if goals_against == 0:
                resistance = 1.0  # Perfect resistance
            elif goals_against == 1:
                resistance = 0.7
            elif goals_against == 2:
                resistance = 0.4
            else:
                resistance = 0.1
            
            resistance_scores.append(resistance)
        
        return np.mean(resistance_scores)
    
    def _get_default_efficiency_stats(self) -> Dict[str, float]:
        """Return default efficiency statistics when insufficient data."""
        return {
            'goals_per_shot_ratio': 0.15,
            'shots_per_goal_ratio': 6.0,
            'big_chances_conversion': 0.4,
            'attacking_third_efficiency': 0.5,
            'defensive_actions_per_goal_conceded': 20.0,
            'clean_sheet_probability': 0.3,
            'defensive_third_efficiency': 0.5,
            'pressure_resistance': 0.5,
            'possession_efficiency': 0.5,
            'territory_control': 0.5,
            'tempo_control': 0.5,
            'tactical_discipline': 0.5,
            'game_management': 0.5,
            'adaptability_index': 0.5
        }


class ClutchPerformanceAnalyzer:
    """
    Analyzes team performance in high-pressure situations.
    
    Evaluates performance in late-game scenarios, close matches, and decisive moments.
    """
    
    def __init__(self, clutch_minutes: int = 75):
        self.clutch_minutes = clutch_minutes
        self.logger = get_logger("ClutchPerformanceAnalyzer")
    
    def analyze_clutch_performance(self, 
                                 team_matches: pd.DataFrame,
                                 team_name: str,
                                 reference_date: datetime) -> Dict[str, float]:
        """
        Analyze clutch performance metrics.
        
        Returns:
            Dict with clutch performance indicators
        """
        try:
            # Filter recent matches
            recent_matches = team_matches[
                (team_matches['date'] < reference_date)
            ].tail(20)  # Focus on recent form for clutch analysis
            
            if len(recent_matches) < 5:
                return self._get_default_clutch_stats()
            
            # Analyze different clutch scenarios
            late_game_performance = self._analyze_late_game_performance(recent_matches, team_name)
            close_match_performance = self._analyze_close_matches(recent_matches, team_name)
            pressure_situations = self._analyze_pressure_situations(recent_matches, team_name)
            comeback_ability = self._analyze_comeback_ability(recent_matches, team_name)
            
            stats = {
                'late_game_goals_for': late_game_performance['goals_for'],
                'late_game_goals_against': late_game_performance['goals_against'],
                'late_game_goal_differential': late_game_performance['goal_diff'],
                'clutch_goal_ratio': late_game_performance['clutch_ratio'],
                'close_match_win_rate': close_match_performance['win_rate'],
                'close_match_points_per_game': close_match_performance['points_per_game'],
                'one_goal_game_performance': close_match_performance['one_goal_performance'],
                'pressure_response_index': pressure_situations['response_index'],
                'high_stakes_performance': pressure_situations['high_stakes_performance'],
                'mental_strength_score': pressure_situations['mental_strength'],
                'comeback_frequency': comeback_ability['comeback_frequency'],
                'comeback_success_rate': comeback_ability['success_rate'],
                'resilience_after_conceding': comeback_ability['resilience'],
                'lead_protection_ability': self._analyze_lead_protection(recent_matches, team_name)
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error analyzing clutch performance for {team_name}: {e}")
            return self._get_default_clutch_stats()
    
    def _analyze_late_game_performance(self, matches: pd.DataFrame, team_name: str) -> Dict[str, float]:
        """Analyze performance in late game situations (75+ minutes)."""
        late_goals_for = 0
        late_goals_against = 0
        total_goals_for = 0
        
        for _, match in matches.iterrows():
            is_home = match['home_team'] == team_name
            goals_for = match['home_score'] if is_home else match['away_score']
            goals_against = match['away_score'] if is_home else match['home_score']
            
            # Estimate late goals (simplified - in reality would need minute-by-minute data)
            estimated_late_goals_for = goals_for * 0.25  # Assume 25% of goals in last 15 minutes
            estimated_late_goals_against = goals_against * 0.25
            
            late_goals_for += estimated_late_goals_for
            late_goals_against += estimated_late_goals_against
            total_goals_for += goals_for
        
        clutch_ratio = late_goals_for / max(total_goals_for, 1) if total_goals_for > 0 else 0.25
        
        return {
            'goals_for': late_goals_for / len(matches),
            'goals_against': late_goals_against / len(matches),
            'goal_diff': (late_goals_for - late_goals_against) / len(matches),
            'clutch_ratio': clutch_ratio
        }
    
    def _analyze_close_matches(self, matches: pd.DataFrame, team_name: str) -> Dict[str, float]:
        """Analyze performance in close matches (1-goal difference or draws)."""
        close_matches = []
        one_goal_matches = []
        
        for _, match in matches.iterrows():
            is_home = match['home_team'] == team_name
            goals_for = match['home_score'] if is_home else match['away_score']
            goals_against = match['away_score'] if is_home else match['home_score']
            
            goal_diff = abs(goals_for - goals_against)
            
            if goal_diff <= 1:
                close_matches.append(match)
                
                if goal_diff == 1:
                    one_goal_matches.append({
                        'won': goals_for > goals_against,
                        'goals_for': goals_for,
                        'goals_against': goals_against
                    })
        
        if not close_matches:
            return {'win_rate': 0.33, 'points_per_game': 1.0, 'one_goal_performance': 0.5}
        
        # Calculate close match statistics
        close_wins = 0
        close_draws = 0
        total_points = 0
        
        for match in close_matches:
            is_home = match['home_team'] == team_name
            goals_for = match['home_score'] if is_home else match['away_score']
            goals_against = match['away_score'] if is_home else match['home_score']
            
            if goals_for > goals_against:
                close_wins += 1
                total_points += 3
            elif goals_for == goals_against:
                close_draws += 1
                total_points += 1
        
        win_rate = close_wins / len(close_matches)
        points_per_game = total_points / len(close_matches)
        
        # One-goal game performance
        one_goal_wins = sum(1 for match in one_goal_matches if match['won'])
        one_goal_performance = one_goal_wins / max(len(one_goal_matches), 1)
        
        return {
            'win_rate': win_rate,
            'points_per_game': points_per_game,
            'one_goal_performance': one_goal_performance
        }
    
    def _analyze_pressure_situations(self, matches: pd.DataFrame, team_name: str) -> Dict[str, float]:
        """Analyze performance under pressure."""
        pressure_scores = []
        high_stakes_scores = []
        
        for _, match in matches.iterrows():
            is_home = match['home_team'] == team_name
            goals_for = match['home_score'] if is_home else match['away_score']
            goals_against = match['away_score'] if is_home else match['home_score']
            
            # Estimate pressure level based on match importance and scoreline
            base_pressure = 0.5
            
            # Higher pressure in close games
            if abs(goals_for - goals_against) <= 1:
                base_pressure += 0.2
            
            # Performance under pressure
            if goals_for >= goals_against:
                performance = 0.7 + (goals_for - goals_against) * 0.1
            else:
                performance = 0.3 - (goals_against - goals_for) * 0.1
            
            pressure_scores.append(performance)
            
            # High stakes estimation (simplified)
            high_stakes_performance = performance * base_pressure
            high_stakes_scores.append(high_stakes_performance)
        
        response_index = np.mean(pressure_scores)
        high_stakes_performance = np.mean(high_stakes_scores)
        mental_strength = min(response_index + (1 - np.std(pressure_scores)), 1.0)
        
        return {
            'response_index': response_index,
            'high_stakes_performance': high_stakes_performance,
            'mental_strength': mental_strength
        }
    
    def _analyze_comeback_ability(self, matches: pd.DataFrame, team_name: str) -> Dict[str, float]:
        """Analyze team's ability to come back from deficits."""
        comeback_attempts = 0
        successful_comebacks = 0
        resilience_scores = []
        
        for _, match in matches.iterrows():
            is_home = match['home_team'] == team_name
            goals_for = match['home_score'] if is_home else match['away_score']
            goals_against = match['away_score'] if is_home else match['home_score']
            
            # Simulate being behind at some point (simplified)
            if goals_for >= goals_against and goals_against > 0:
                # Assume team was behind and came back
                comeback_attempts += 1
                if goals_for > goals_against:
                    successful_comebacks += 1
                    resilience_scores.append(1.0)
                else:
                    resilience_scores.append(0.5)  # Draw after being behind
            elif goals_against > 0:
                # Conceded first but didn't come back
                resilience_scores.append(0.2)
            else:
                # Clean sheet - good resilience
                resilience_scores.append(0.8)
        
        comeback_frequency = comeback_attempts / max(len(matches), 1)
        success_rate = successful_comebacks / max(comeback_attempts, 1)
        resilience = np.mean(resilience_scores) if resilience_scores else 0.5
        
        return {
            'comeback_frequency': comeback_frequency,
            'success_rate': success_rate,
            'resilience': resilience
        }
    
    def _analyze_lead_protection(self, matches: pd.DataFrame, team_name: str) -> float:
        """Analyze ability to protect leads."""
        lead_situations = 0
        leads_protected = 0
        
        for _, match in matches.iterrows():
            is_home = match['home_team'] == team_name
            goals_for = match['home_score'] if is_home else match['away_score']
            goals_against = match['away_score'] if is_home else match['home_score']
            
            if goals_for > 0:  # Had a lead at some point
                lead_situations += 1
                if goals_for >= goals_against:
                    leads_protected += 1
        
        return leads_protected / max(lead_situations, 1)
    
    def _get_default_clutch_stats(self) -> Dict[str, float]:
        """Return default clutch performance statistics."""
        return {
            'late_game_goals_for': 0.25,
            'late_game_goals_against': 0.25,
            'late_game_goal_differential': 0.0,
            'clutch_goal_ratio': 0.25,
            'close_match_win_rate': 0.33,
            'close_match_points_per_game': 1.0,
            'one_goal_game_performance': 0.5,
            'pressure_response_index': 0.5,
            'high_stakes_performance': 0.5,
            'mental_strength_score': 0.5,
            'comeback_frequency': 0.2,
            'comeback_success_rate': 0.5,
            'resilience_after_conceding': 0.5,
            'lead_protection_ability': 0.7
        }


class StyleAnalyzer:
    """
    Analyzes team playing style and tactical patterns.
    
    Evaluates attacking patterns, defensive approach, and overall team philosophy.
    """
    
    def __init__(self, analysis_matches: int = 20):
        self.analysis_matches = analysis_matches
        self.logger = get_logger("StyleAnalyzer")
    
    def analyze_team_style(self, 
                          team_matches: pd.DataFrame,
                          team_name: str,
                          reference_date: datetime) -> Dict[str, float]:
        """
        Analyze comprehensive team style metrics.
        
        Returns:
            Dict with style and tactical pattern indicators
        """
        try:
            # Filter recent matches for style analysis
            recent_matches = team_matches[
                (team_matches['date'] < reference_date)
            ].tail(self.analysis_matches)
            
            if len(recent_matches) < 8:
                return self._get_default_style_stats()
            
            # Analyze different aspects of team style
            attacking_style = self._analyze_attacking_style(recent_matches, team_name)
            defensive_style = self._analyze_defensive_style(recent_matches, team_name)
            tactical_patterns = self._analyze_tactical_patterns(recent_matches, team_name)
            game_tempo = self._analyze_game_tempo(recent_matches, team_name)
            
            stats = {
                'attacking_intensity': attacking_style['intensity'],
                'goal_threat_level': attacking_style['threat_level'],
                'attacking_variety': attacking_style['variety'],
                'counter_attack_propensity': attacking_style['counter_attack'],
                'defensive_solidity': defensive_style['solidity'],
                'defensive_aggression': defensive_style['aggression'],
                'pressing_intensity': defensive_style['pressing'],
                'defensive_line_height': defensive_style['line_height'],
                'tactical_flexibility': tactical_patterns['flexibility'],
                'formation_consistency': tactical_patterns['formation_consistency'],
                'game_control_preference': tactical_patterns['control_preference'],
                'set_piece_effectiveness': tactical_patterns['set_piece_effectiveness'],
                'tempo_preference': game_tempo['tempo_preference'],
                'rhythm_consistency': game_tempo['rhythm_consistency'],
                'momentum_management': game_tempo['momentum_management']
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error analyzing team style for {team_name}: {e}")
            return self._get_default_style_stats()
    
    def _analyze_attacking_style(self, matches: pd.DataFrame, team_name: str) -> Dict[str, float]:
        """Analyze attacking style characteristics."""
        total_goals = 0
        high_scoring_matches = 0
        varied_scoring = []
        quick_goals = 0
        
        for _, match in matches.iterrows():
            is_home = match['home_team'] == team_name
            goals = match['home_score'] if is_home else match['away_score']
            opponent_goals = match['away_score'] if is_home else match['home_score']
            
            total_goals += goals
            
            if goals >= 2:
                high_scoring_matches += 1
            
            # Estimate variety in scoring patterns
            if goals > 0:
                # Simulate different types of goals (simplified)
                variety_score = min(goals / 3, 1.0) + np.random.normal(0, 0.1)
                varied_scoring.append(max(0, min(1, variety_score)))
            
            # Estimate counter-attacking tendency
            if goals > opponent_goals and opponent_goals == 0:
                quick_goals += 1  # Assume quick counter-attacks in wins
        
        intensity = total_goals / len(matches)
        threat_level = high_scoring_matches / len(matches)
        variety = np.mean(varied_scoring) if varied_scoring else 0.3
        counter_attack = quick_goals / len(matches)
        
        return {
            'intensity': min(intensity / 2.5, 1.0),  # Normalize by 2.5 goals per game
            'threat_level': threat_level,
            'variety': variety,
            'counter_attack': counter_attack
        }
    
    def _analyze_defensive_style(self, matches: pd.DataFrame, team_name: str) -> Dict[str, float]:
        """Analyze defensive style characteristics."""
        clean_sheets = 0
        low_conceding_matches = 0
        total_conceded = 0
        aggressive_defending = []
        
        for _, match in matches.iterrows():
            is_home = match['home_team'] == team_name
            goals_for = match['home_score'] if is_home else match['away_score']
            goals_against = match['away_score'] if is_home else match['home_score']
            
            if goals_against == 0:
                clean_sheets += 1
            
            if goals_against <= 1:
                low_conceding_matches += 1
            
            total_conceded += goals_against
            
            # Estimate aggressive defending (higher when winning/drawing)
            if goals_for >= goals_against:
                aggression = 0.7 + goals_for * 0.1
            else:
                aggression = 0.3 - (goals_against - goals_for) * 0.1
            
            aggressive_defending.append(max(0, min(1, aggression)))
        
        solidity = clean_sheets / len(matches)
        aggression = np.mean(aggressive_defending)
        pressing = 1 - (total_conceded / len(matches)) / 3  # Normalize by 3 goals
        line_height = (solidity + aggression) / 2  # Estimate based on other metrics
        
        return {
            'solidity': solidity,
            'aggression': aggression,
            'pressing': max(0, pressing),
            'line_height': line_height
        }
    
    def _analyze_tactical_patterns(self, matches: pd.DataFrame, team_name: str) -> Dict[str, float]:
        """Analyze tactical patterns and flexibility."""
        result_patterns = []
        goal_patterns = []
        
        for _, match in matches.iterrows():
            is_home = match['home_team'] == team_name
            goals_for = match['home_score'] if is_home else match['away_score']
            goals_against = match['away_score'] if is_home else match['home_score']
            
            # Analyze result patterns
            if goals_for > goals_against:
                result_patterns.append(3)  # Win
            elif goals_for == goals_against:
                result_patterns.append(1)  # Draw
            else:
                result_patterns.append(0)  # Loss
            
            goal_patterns.append(goals_for)
        
        # Calculate tactical metrics
        flexibility = 1 - (np.std(result_patterns) / 1.5)  # Lower std = less flexible
        flexibility = max(0, min(1, flexibility))
        
        formation_consistency = 1 - (np.std(goal_patterns) / 2)  # Consistent scoring patterns
        formation_consistency = max(0, min(1, formation_consistency))
        
        control_preference = np.mean(result_patterns) / 3  # Higher = more control
        
        # Estimate set piece effectiveness
        set_piece_goals = sum(1 for goals in goal_patterns if goals > 2) / len(matches)
        set_piece_effectiveness = min(set_piece_goals, 0.5) / 0.5  # Normalize
        
        return {
            'flexibility': flexibility,
            'formation_consistency': formation_consistency,
            'control_preference': control_preference,
            'set_piece_effectiveness': set_piece_effectiveness
        }
    
    def _analyze_game_tempo(self, matches: pd.DataFrame, team_name: str) -> Dict[str, float]:
        """Analyze game tempo and rhythm preferences."""
        tempo_scores = []
        rhythm_scores = []
        momentum_scores = []
        
        for _, match in matches.iterrows():
            is_home = match['home_team'] == team_name
            goals_for = match['home_score'] if is_home else match['away_score']
            goals_against = match['away_score'] if is_home else match['home_score']
            
            total_goals = goals_for + goals_against
            
            # Estimate tempo based on total goals and result
            if total_goals >= 3:
                tempo = 0.8  # High tempo game
            elif total_goals == 2:
                tempo = 0.6  # Medium tempo
            else:
                tempo = 0.3  # Low tempo
            
            tempo_scores.append(tempo)
            
            # Rhythm consistency (ability to maintain performance)
            if goals_for >= goals_against:
                rhythm = 0.7 + goals_for * 0.1
            else:
                rhythm = 0.3
            
            rhythm_scores.append(max(0, min(1, rhythm)))
            
            # Momentum management (how well team handles game flow)
            goal_diff = goals_for - goals_against
            momentum = 0.5 + goal_diff * 0.1
            momentum_scores.append(max(0, min(1, momentum)))
        
        return {
            'tempo_preference': np.mean(tempo_scores),
            'rhythm_consistency': np.mean(rhythm_scores),
            'momentum_management': np.mean(momentum_scores)
        }
    
    def _get_default_style_stats(self) -> Dict[str, float]:
        """Return default style statistics when insufficient data."""
        return {
            'attacking_intensity': 0.5,
            'goal_threat_level': 0.3,
            'attacking_variety': 0.4,
            'counter_attack_propensity': 0.2,
            'defensive_solidity': 0.3,
            'defensive_aggression': 0.5,
            'pressing_intensity': 0.5,
            'defensive_line_height': 0.5,
            'tactical_flexibility': 0.5,
            'formation_consistency': 0.6,
            'game_control_preference': 0.5,
            'set_piece_effectiveness': 0.3,
            'tempo_preference': 0.5,
            'rhythm_consistency': 0.5,
            'momentum_management': 0.5
        }


class AdvancedMetricsEngine(FeatureEngineer):
    """
    Main orchestrator for advanced metrics feature engineering.
    
    Combines xG analysis, efficiency metrics, clutch performance, and style analysis.
    Follows Facade pattern to provide unified interface to advanced analytics.
    """
    
    def __init__(self, config: Optional[AdvancedMetricsConfig] = None):
        self.config = config or AdvancedMetricsConfig()
        self.logger = get_logger("AdvancedMetricsEngine")
        
        # Initialize components
        self.xg_analyzer = ExpectedGoalsAnalyzer(
            self.config.xg_lookback_matches, self.config.min_shots_for_xg
        )
        self.efficiency_calculator = EfficiencyCalculator(self.config.efficiency_window)
        self.clutch_analyzer = ClutchPerformanceAnalyzer(self.config.clutch_minutes_threshold)
        self.style_analyzer = StyleAnalyzer(self.config.style_analysis_matches)
        
        self._feature_names = None
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced metrics features for all matches in the dataset.
        
        Args:
            data: DataFrame with match data including date, teams, scores
            
        Returns:
            DataFrame with added advanced metrics features
        """
        try:
            self.logger.info(f"Creating advanced metrics features for {len(data)} matches")
            
            if data.empty:
                raise FeatureEngineeringError("Empty dataset provided")
            
            # Prepare result DataFrame
            result = data.copy()
            
            # Process each match
            for idx, match in data.iterrows():
                match_features = self._create_match_features(match, data)
                
                # Add features to result
                for feature_name, feature_value in match_features.items():
                    result.at[idx, feature_name] = feature_value
            
            self.logger.info(f"Created {len(self.get_feature_names())} advanced metrics features")
            return result
            
        except Exception as e:
            self.logger.error(f"Error creating advanced metrics features: {e}")
            raise FeatureEngineeringError(f"Advanced metrics feature creation failed: {e}")
    
    def _create_match_features(self, match: pd.Series, full_data: pd.DataFrame) -> Dict[str, float]:
        """Create advanced metrics features for a single match."""
        home_team = match['home_team']
        away_team = match['away_team']
        match_date = pd.to_datetime(match['date'])
        
        # Get historical data
        historical_data = full_data[full_data['date'] < match['date']]
        
        features = {}
        
        # xG metrics for both teams
        home_xg = self.xg_analyzer.calculate_xg_metrics(
            historical_data, home_team, match_date
        )
        for key, value in home_xg.items():
            features[f"home_xg_{key}"] = value
        
        away_xg = self.xg_analyzer.calculate_xg_metrics(
            historical_data, away_team, match_date
        )
        for key, value in away_xg.items():
            features[f"away_xg_{key}"] = value
        
        # Efficiency metrics for both teams
        home_efficiency = self.efficiency_calculator.calculate_efficiency_metrics(
            historical_data, home_team, match_date
        )
        for key, value in home_efficiency.items():
            features[f"home_eff_{key}"] = value
        
        away_efficiency = self.efficiency_calculator.calculate_efficiency_metrics(
            historical_data, away_team, match_date
        )
        for key, value in away_efficiency.items():
            features[f"away_eff_{key}"] = value
        
        # Clutch performance for both teams
        home_clutch = self.clutch_analyzer.analyze_clutch_performance(
            historical_data, home_team, match_date
        )
        for key, value in home_clutch.items():
            features[f"home_clutch_{key}"] = value
        
        away_clutch = self.clutch_analyzer.analyze_clutch_performance(
            historical_data, away_team, match_date
        )
        for key, value in away_clutch.items():
            features[f"away_clutch_{key}"] = value
        
        # Style analysis for both teams
        home_style = self.style_analyzer.analyze_team_style(
            historical_data, home_team, match_date
        )
        for key, value in home_style.items():
            features[f"home_style_{key}"] = value
        
        away_style = self.style_analyzer.analyze_team_style(
            historical_data, away_team, match_date
        )
        for key, value in away_style.items():
            features[f"away_style_{key}"] = value
        
        # Additional comparative features
        features.update(self._calculate_comparative_features(home_xg, away_xg, home_efficiency, away_efficiency))
        
        return features
    
    def _calculate_comparative_features(self, home_xg: Dict, away_xg: Dict, 
                                      home_eff: Dict, away_eff: Dict) -> Dict[str, float]:
        """Calculate comparative features between teams."""
        comparative = {}
        
        # xG comparisons
        comparative['xg_advantage_home'] = home_xg['avg_xg_for'] - away_xg['avg_xg_for']
        comparative['xg_defensive_advantage_home'] = away_xg['avg_xg_against'] - home_xg['avg_xg_against']
        comparative['finishing_advantage_home'] = home_xg['finishing_quality'] - away_xg['finishing_quality']
        comparative['xg_consistency_advantage_home'] = home_xg['xg_consistency'] - away_xg['xg_consistency']
        
        # Efficiency comparisons
        comparative['attacking_efficiency_advantage_home'] = (
            home_eff['goals_per_shot_ratio'] - away_eff['goals_per_shot_ratio']
        )
        comparative['defensive_efficiency_advantage_home'] = (
            home_eff['defensive_actions_per_goal_conceded'] - away_eff['defensive_actions_per_goal_conceded']
        )
        comparative['possession_advantage_home'] = (
            home_eff['possession_efficiency'] - away_eff['possession_efficiency']
        )
        
        return comparative
    
    def get_feature_names(self) -> List[str]:
        """Return list of all advanced metrics feature names."""
        if self._feature_names is None:
            self._feature_names = self._generate_feature_names()
        return self._feature_names
    
    def _generate_feature_names(self) -> List[str]:
        """Generate complete list of feature names."""
        feature_names = []
        
        # xG features (for both teams)
        xg_features = [
            'avg_xg_for', 'avg_xg_against', 'avg_actual_goals_for', 'avg_actual_goals_against',
            'xg_diff', 'actual_goal_diff', 'xg_outperformance_for', 'xg_outperformance_against',
            'finishing_quality', 'defensive_resilience', 'xg_variance', 'goal_variance',
            'xg_consistency', 'performance_sustainability'
        ]
        
        for team_prefix in ['home_xg_', 'away_xg_']:
            for feature in xg_features:
                feature_names.append(f"{team_prefix}{feature}")
        
        # Efficiency features (for both teams)
        efficiency_features = [
            'goals_per_shot_ratio', 'shots_per_goal_ratio', 'big_chances_conversion',
            'attacking_third_efficiency', 'defensive_actions_per_goal_conceded',
            'clean_sheet_probability', 'defensive_third_efficiency', 'pressure_resistance',
            'possession_efficiency', 'territory_control', 'tempo_control',
            'tactical_discipline', 'game_management', 'adaptability_index'
        ]
        
        for team_prefix in ['home_eff_', 'away_eff_']:
            for feature in efficiency_features:
                feature_names.append(f"{team_prefix}{feature}")
        
        # Clutch performance features (for both teams)
        clutch_features = [
            'late_game_goals_for', 'late_game_goals_against', 'late_game_goal_differential',
            'clutch_goal_ratio', 'close_match_win_rate', 'close_match_points_per_game',
            'one_goal_game_performance', 'pressure_response_index', 'high_stakes_performance',
            'mental_strength_score', 'comeback_frequency', 'comeback_success_rate',
            'resilience_after_conceding', 'lead_protection_ability'
        ]
        
        for team_prefix in ['home_clutch_', 'away_clutch_']:
            for feature in clutch_features:
                feature_names.append(f"{team_prefix}{feature}")
        
        # Style features (for both teams)
        style_features = [
            'attacking_intensity', 'goal_threat_level', 'attacking_variety',
            'counter_attack_propensity', 'defensive_solidity', 'defensive_aggression',
            'pressing_intensity', 'defensive_line_height', 'tactical_flexibility',
            'formation_consistency', 'game_control_preference', 'set_piece_effectiveness',
            'tempo_preference', 'rhythm_consistency', 'momentum_management'
        ]
        
        for team_prefix in ['home_style_', 'away_style_']:
            for feature in style_features:
                feature_names.append(f"{team_prefix}{feature}")
        
        # Comparative features
        comparative_features = [
            'xg_advantage_home', 'xg_defensive_advantage_home', 'finishing_advantage_home',
            'xg_consistency_advantage_home', 'attacking_efficiency_advantage_home',
            'defensive_efficiency_advantage_home', 'possession_advantage_home'
        ]
        
        feature_names.extend(comparative_features)
        
        return feature_names
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Return estimated feature importance scores.
        
        Based on domain knowledge and typical predictive power.
        """
        importance_scores = {}
        
        # Very high importance features
        very_high_impact = [
            'xg_advantage_home', 'finishing_advantage_home', 'attacking_efficiency_advantage_home'
        ]
        
        # High importance features
        high_impact = [
            'home_xg_avg_xg_for', 'away_xg_avg_xg_for', 'home_xg_finishing_quality',
            'away_xg_finishing_quality', 'home_clutch_mental_strength_score',
            'away_clutch_mental_strength_score'
        ]
        
        # Medium importance features
        medium_impact = [
            'home_eff_possession_efficiency', 'away_eff_possession_efficiency',
            'home_style_attacking_intensity', 'away_style_defensive_solidity'
        ]
        
        # Assign importance scores
        for feature in self.get_feature_names():
            if any(vhi in feature for vhi in very_high_impact):
                importance_scores[feature] = 0.95
            elif any(hi in feature for hi in high_impact):
                importance_scores[feature] = 0.85
            elif any(mi in feature for mi in medium_impact):
                importance_scores[feature] = 0.7
            elif 'xg_' in feature:
                importance_scores[feature] = 0.8
            elif 'clutch_' in feature:
                importance_scores[feature] = 0.6
            elif 'eff_' in feature:
                importance_scores[feature] = 0.65
            elif 'style_' in feature:
                importance_scores[feature] = 0.5
            else:
                importance_scores[feature] = 0.45
        
        return importance_scores