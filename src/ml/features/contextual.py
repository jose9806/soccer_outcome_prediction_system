"""
Contextual feature engineering components for soccer match analysis.

Provides context-aware features including home advantage, team rivalries,
travel fatigue, and seasonal patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import math

from src.config.logging_config import get_logger
from ..core.interfaces import FeatureEngineer
from ..core.exceptions import FeatureEngineeringError


@dataclass
class ContextualConfig:
    """Configuration for contextual feature engineering."""
    home_advantage_lookback: int = 50
    rivalry_threshold: int = 10
    travel_distance_threshold: float = 500.0
    seasonal_months_lookback: int = 12
    derby_keywords: List[str] = None
    
    def __post_init__(self):
        if self.derby_keywords is None:
            self.derby_keywords = ['real', 'atletico', 'barcelona', 'madrid', 'sevilla', 'valencia']


class HomeAdvantageCalculator:
    """
    Calculates home advantage statistics for teams.
    
    Following Single Responsibility Principle - focused only on home advantage analysis.
    """
    
    def __init__(self, lookback_matches: int = 50):
        self.lookback_matches = lookback_matches
        self.logger = get_logger("HomeAdvantageCalculator")
    
    def calculate_home_advantage(self, 
                               team_matches: pd.DataFrame,
                               team_name: str,
                               reference_date: datetime) -> Dict[str, float]:
        """
        Calculate comprehensive home advantage metrics.
        
        Returns:
            Dict with home advantage statistics
        """
        try:
            # Filter matches before reference date
            historical_matches = team_matches[
                (team_matches['date'] < reference_date)
            ].tail(self.lookback_matches)
            
            if len(historical_matches) < 5:
                return self._get_default_home_stats()
            
            # Separate home and away matches
            home_matches = historical_matches[
                historical_matches['home_team'] == team_name
            ]
            away_matches = historical_matches[
                historical_matches['away_team'] == team_name
            ]
            
            if len(home_matches) == 0 or len(away_matches) == 0:
                return self._get_default_home_stats()
            
            # Calculate basic stats
            home_stats = self._calculate_basic_stats(home_matches, team_name, 'home')
            away_stats = self._calculate_basic_stats(away_matches, team_name, 'away')
            
            # Calculate advantage differentials
            stats = {
                'home_advantage_win_rate': home_stats['win_rate'] - away_stats['win_rate'],
                'home_advantage_goals_for': home_stats['avg_goals_for'] - away_stats['avg_goals_for'],
                'home_advantage_goals_against': away_stats['avg_goals_against'] - home_stats['avg_goals_against'],
                'home_advantage_goal_diff': (home_stats['avg_goals_for'] - home_stats['avg_goals_against']) - 
                                          (away_stats['avg_goals_for'] - away_stats['avg_goals_against']),
                'home_matches_ratio': len(home_matches) / len(historical_matches),
                'home_form_strength': self._calculate_form_strength(home_matches, team_name, 'home'),
                'away_form_strength': self._calculate_form_strength(away_matches, team_name, 'away'),
                'venue_consistency': self._calculate_venue_consistency(historical_matches, team_name)
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating home advantage for {team_name}: {e}")
            return self._get_default_home_stats()
    
    def _calculate_basic_stats(self, matches: pd.DataFrame, team_name: str, venue: str) -> Dict[str, float]:
        """Calculate basic statistics for home or away matches."""
        if venue == 'home':
            goals_for = matches['home_score'].mean()
            goals_against = matches['away_score'].mean()
            wins = (matches['home_score'] > matches['away_score']).mean()
        else:
            goals_for = matches['away_score'].mean()
            goals_against = matches['home_score'].mean()
            wins = (matches['away_score'] > matches['home_score']).mean()
        
        return {
            'avg_goals_for': goals_for,
            'avg_goals_against': goals_against,
            'win_rate': wins
        }
    
    def _calculate_form_strength(self, matches: pd.DataFrame, team_name: str, venue: str) -> float:
        """Calculate weighted form strength with recent matches having more impact."""
        if len(matches) == 0:
            return 0.0
        
        # Sort by date (most recent first)
        sorted_matches = matches.sort_values('date', ascending=False)
        
        weights = np.exp(-np.arange(len(sorted_matches)) * 0.1)  # Exponential decay
        
        if venue == 'home':
            results = (sorted_matches['home_score'] > sorted_matches['away_score']).astype(float)
            goal_diff = sorted_matches['home_score'] - sorted_matches['away_score']
        else:
            results = (sorted_matches['away_score'] > sorted_matches['home_score']).astype(float)
            goal_diff = sorted_matches['away_score'] - sorted_matches['home_score']
        
        # Combine win/loss with goal difference
        strength = np.average(results + goal_diff * 0.1, weights=weights)
        return strength
    
    def _calculate_venue_consistency(self, matches: pd.DataFrame, team_name: str) -> float:
        """Calculate how consistent team performance is between venues."""
        home_matches = matches[matches['home_team'] == team_name]
        away_matches = matches[matches['away_team'] == team_name]
        
        if len(home_matches) == 0 or len(away_matches) == 0:
            return 0.0
        
        home_goals = home_matches['home_score'] - home_matches['away_score']
        away_goals = away_matches['away_score'] - away_matches['home_score']
        
        # Lower variance indicates higher consistency
        home_variance = home_goals.var()
        away_variance = away_goals.var()
        
        # Return normalized consistency score (higher = more consistent)
        consistency = 1 / (1 + abs(home_variance - away_variance))
        return consistency
    
    def _get_default_home_stats(self) -> Dict[str, float]:
        """Return default statistics when insufficient data."""
        return {
            'home_advantage_win_rate': 0.0,
            'home_advantage_goals_for': 0.0,
            'home_advantage_goals_against': 0.0,
            'home_advantage_goal_diff': 0.0,
            'home_matches_ratio': 0.5,
            'home_form_strength': 0.0,
            'away_form_strength': 0.0,
            'venue_consistency': 0.5
        }


class RivalryAnalyzer:
    """
    Analyzes historical rivalry patterns between teams.
    
    Identifies meaningful rivalries and their impact on performance.
    """
    
    def __init__(self, min_matches_for_rivalry: int = 10, derby_keywords: List[str] = None):
        self.min_matches_for_rivalry = min_matches_for_rivalry
        self.derby_keywords = derby_keywords or ['real', 'atletico', 'barcelona', 'madrid', 'sevilla']
        self.logger = get_logger("RivalryAnalyzer")
        self._rivalry_cache = {}
    
    def analyze_rivalry(self, 
                       home_team: str, 
                       away_team: str,
                       historical_data: pd.DataFrame,
                       reference_date: datetime) -> Dict[str, float]:
        """
        Analyze rivalry between two teams.
        
        Returns:
            Dict with rivalry metrics and head-to-head statistics
        """
        try:
            cache_key = f"{home_team}_{away_team}_{reference_date.date()}"
            if cache_key in self._rivalry_cache:
                return self._rivalry_cache[cache_key]
            
            # Filter head-to-head matches before reference date
            h2h_matches = historical_data[
                (((historical_data['home_team'] == home_team) & 
                  (historical_data['away_team'] == away_team)) |
                 ((historical_data['home_team'] == away_team) & 
                  (historical_data['away_team'] == home_team))) &
                (historical_data['date'] < reference_date)
            ].sort_values('date')
            
            if len(h2h_matches) < 3:
                stats = self._get_default_rivalry_stats()
            else:
                stats = self._calculate_rivalry_metrics(h2h_matches, home_team, away_team)
            
            # Add derby detection
            stats['is_derby'] = self._detect_derby(home_team, away_team)
            stats['rivalry_intensity'] = self._calculate_rivalry_intensity(
                h2h_matches, home_team, away_team, stats['is_derby']
            )
            
            self._rivalry_cache[cache_key] = stats
            return stats
            
        except Exception as e:
            self.logger.error(f"Error analyzing rivalry {home_team} vs {away_team}: {e}")
            return self._get_default_rivalry_stats()
    
    def _calculate_rivalry_metrics(self, h2h_matches: pd.DataFrame, 
                                 home_team: str, away_team: str) -> Dict[str, float]:
        """Calculate detailed rivalry metrics from head-to-head matches."""
        total_matches = len(h2h_matches)
        
        # Overall head-to-head record
        team1_wins = len(h2h_matches[
            ((h2h_matches['home_team'] == home_team) & 
             (h2h_matches['home_score'] > h2h_matches['away_score'])) |
            ((h2h_matches['away_team'] == home_team) & 
             (h2h_matches['away_score'] > h2h_matches['home_score']))
        ])
        
        team2_wins = len(h2h_matches[
            ((h2h_matches['home_team'] == away_team) & 
             (h2h_matches['home_score'] > h2h_matches['away_score'])) |
            ((h2h_matches['away_team'] == away_team) & 
             (h2h_matches['away_score'] > h2h_matches['home_score']))
        ])
        
        draws = total_matches - team1_wins - team2_wins
        
        # Recent form (last 5 matches)
        recent_matches = h2h_matches.tail(min(5, len(h2h_matches)))
        recent_team1_wins = len(recent_matches[
            ((recent_matches['home_team'] == home_team) & 
             (recent_matches['home_score'] > recent_matches['away_score'])) |
            ((recent_matches['away_team'] == home_team) & 
             (recent_matches['away_score'] > recent_matches['home_score']))
        ])
        
        # Goal statistics
        team1_goals = 0
        team2_goals = 0
        
        for _, match in h2h_matches.iterrows():
            if match['home_team'] == home_team:
                team1_goals += match['home_score']
                team2_goals += match['away_score']
            else:
                team1_goals += match['away_score']
                team2_goals += match['home_score']
        
        # Calculate competitive balance (how evenly matched teams are)
        balance = 1 - abs(team1_wins - team2_wins) / total_matches
        
        # Calculate match volatility (how unpredictable results are)
        goal_differences = []
        for _, match in h2h_matches.iterrows():
            goal_diff = abs(match['home_score'] - match['away_score'])
            goal_differences.append(goal_diff)
        
        volatility = np.std(goal_differences) if goal_differences else 0
        
        return {
            'h2h_total_matches': total_matches,
            'h2h_home_team_wins': team1_wins,
            'h2h_away_team_wins': team2_wins,
            'h2h_draws': draws,
            'h2h_home_team_win_rate': team1_wins / total_matches,
            'h2h_away_team_win_rate': team2_wins / total_matches,
            'h2h_draw_rate': draws / total_matches,
            'h2h_recent_form_home': recent_team1_wins / len(recent_matches),
            'h2h_avg_goals_home_team': team1_goals / total_matches,
            'h2h_avg_goals_away_team': team2_goals / total_matches,
            'h2h_competitive_balance': balance,
            'h2h_match_volatility': volatility
        }
    
    def _detect_derby(self, home_team: str, away_team: str) -> float:
        """Detect if match is a derby based on team names."""
        home_lower = home_team.lower()
        away_lower = away_team.lower()
        
        # Check for common city/region keywords
        derby_score = 0.0
        for keyword in self.derby_keywords:
            if keyword in home_lower and keyword in away_lower:
                derby_score += 1.0
        
        # Normalize derby score
        return min(derby_score / len(self.derby_keywords), 1.0)
    
    def _calculate_rivalry_intensity(self, h2h_matches: pd.DataFrame, 
                                   home_team: str, away_team: str, is_derby: float) -> float:
        """Calculate overall rivalry intensity score."""
        if len(h2h_matches) == 0:
            return 0.0
        
        # Base intensity from number of matches
        match_intensity = min(len(h2h_matches) / 20, 1.0)
        
        # Recent frequency (matches in last 3 years)
        recent_cutoff = datetime.now() - timedelta(days=3*365)
        recent_matches = h2h_matches[h2h_matches['date'] > recent_cutoff]
        frequency_intensity = min(len(recent_matches) / 6, 1.0)
        
        # Derby bonus
        derby_bonus = is_derby * 0.5
        
        # Competitive balance (closer matches = higher intensity)
        if len(h2h_matches) > 0:
            goal_diffs = abs(h2h_matches['home_score'] - h2h_matches['away_score'])
            avg_goal_diff = goal_diffs.mean()
            balance_intensity = max(0, 1 - avg_goal_diff / 3)  # Closer matches score higher
        else:
            balance_intensity = 0.0
        
        # Combine factors
        intensity = (match_intensity * 0.3 + 
                    frequency_intensity * 0.3 + 
                    balance_intensity * 0.2 + 
                    derby_bonus)
        
        return min(intensity, 1.0)
    
    def _get_default_rivalry_stats(self) -> Dict[str, float]:
        """Return default rivalry statistics when insufficient data."""
        return {
            'h2h_total_matches': 0,
            'h2h_home_team_wins': 0,
            'h2h_away_team_wins': 0,
            'h2h_draws': 0,
            'h2h_home_team_win_rate': 0.33,
            'h2h_away_team_win_rate': 0.33,
            'h2h_draw_rate': 0.33,
            'h2h_recent_form_home': 0.33,
            'h2h_avg_goals_home_team': 1.5,
            'h2h_avg_goals_away_team': 1.5,
            'h2h_competitive_balance': 0.5,
            'h2h_match_volatility': 1.0,
            'is_derby': 0.0,
            'rivalry_intensity': 0.0
        }


class TravelFatigueCalculator:
    """
    Calculates travel fatigue impact on team performance.
    
    Considers distance traveled, time between matches, and historical impact.
    """
    
    def __init__(self, distance_threshold: float = 500.0):
        self.distance_threshold = distance_threshold
        self.logger = get_logger("TravelFatigueCalculator")
        # Simplified city coordinates (in real implementation, use proper geo data)
        self.city_coordinates = {
            'madrid': (40.4168, -3.7038),
            'barcelona': (41.3851, 2.1734),
            'valencia': (39.4699, -0.3763),
            'sevilla': (37.3891, -5.9845),
            'bilbao': (43.2627, -2.9253),
            'vigo': (42.2406, -8.7207)
        }
    
    def calculate_travel_impact(self, 
                              team_name: str,
                              home_team: str,
                              away_team: str,
                              previous_matches: pd.DataFrame,
                              match_date: datetime) -> Dict[str, float]:
        """
        Calculate travel fatigue impact for a team.
        
        Returns:
            Dict with travel-related features
        """
        try:
            # Get team's recent matches
            recent_matches = previous_matches[
                ((previous_matches['home_team'] == team_name) | 
                 (previous_matches['away_team'] == team_name)) &
                (previous_matches['date'] < match_date)
            ].sort_values('date').tail(5)
            
            if len(recent_matches) == 0:
                return self._get_default_travel_stats()
            
            # Calculate travel distance
            current_distance = self._estimate_travel_distance(team_name, home_team, away_team)
            
            # Calculate days since last match
            last_match_date = recent_matches.iloc[-1]['date']
            days_rest = (match_date - last_match_date).days
            
            # Analyze recent travel pattern
            recent_distances = []
            total_recent_travel = 0
            
            for i, match in recent_matches.iterrows():
                if match['home_team'] == team_name:
                    # Home match - minimal travel
                    distance = 0
                else:
                    # Away match - estimate distance
                    distance = self._estimate_travel_distance(team_name, match['home_team'], match['away_team'])
                
                recent_distances.append(distance)
                total_recent_travel += distance
            
            # Calculate travel fatigue metrics
            stats = {
                'current_travel_distance': current_distance,
                'is_long_distance_travel': float(current_distance > self.distance_threshold),
                'days_since_last_match': min(days_rest, 14),  # Cap at 14 days
                'rest_adequacy': self._calculate_rest_adequacy(days_rest, current_distance),
                'recent_total_travel': total_recent_travel,
                'avg_recent_travel': np.mean(recent_distances) if recent_distances else 0,
                'travel_frequency': sum(1 for d in recent_distances if d > self.distance_threshold) / len(recent_distances),
                'cumulative_fatigue': self._calculate_cumulative_fatigue(recent_distances, recent_matches, match_date)
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating travel impact for {team_name}: {e}")
            return self._get_default_travel_stats()
    
    def _estimate_travel_distance(self, team_name: str, home_team: str, away_team: str) -> float:
        """Estimate travel distance based on team locations."""
        # Simplified distance calculation
        team_city = self._get_team_city(team_name)
        match_city = self._get_team_city(home_team)
        
        if team_city == match_city:
            return 0.0  # Home match
        
        if team_city in self.city_coordinates and match_city in self.city_coordinates:
            # Calculate approximate distance using Haversine formula
            lat1, lon1 = self.city_coordinates[team_city]
            lat2, lon2 = self.city_coordinates[match_city]
            
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = (math.sin(dlat/2)**2 + 
                 math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
                 math.sin(dlon/2)**2)
            distance = 2 * math.asin(math.sqrt(a)) * 6371  # Earth radius in km
            
            return distance
        
        # Default estimate if cities not found
        return 300.0
    
    def _get_team_city(self, team_name: str) -> str:
        """Extract city from team name (simplified)."""
        team_lower = team_name.lower()
        
        for city in self.city_coordinates.keys():
            if city in team_lower:
                return city
        
        # Default to madrid if no match found
        return 'madrid'
    
    def _calculate_rest_adequacy(self, days_rest: int, travel_distance: float) -> float:
        """Calculate how adequate the rest period is given travel distance."""
        # Base rest requirement
        base_rest_needed = 2
        
        # Additional rest needed for long distance travel
        if travel_distance > self.distance_threshold:
            additional_rest = travel_distance / 1000  # 1 day per 1000km
            total_rest_needed = base_rest_needed + additional_rest
        else:
            total_rest_needed = base_rest_needed
        
        # Calculate adequacy ratio
        adequacy = min(days_rest / total_rest_needed, 1.0)
        return adequacy
    
    def _calculate_cumulative_fatigue(self, recent_distances: List[float], 
                                    recent_matches: pd.DataFrame, 
                                    match_date: datetime) -> float:
        """Calculate cumulative fatigue based on recent travel and match frequency."""
        if len(recent_distances) == 0:
            return 0.0
        
        fatigue = 0.0
        decay_factor = 0.8  # Fatigue decays over time
        
        for i, (_, match) in enumerate(recent_matches.iterrows()):
            days_ago = (match_date - match['date']).days
            distance = recent_distances[i]
            
            # Distance-based fatigue
            distance_fatigue = distance / 1000  # Normalize by 1000km
            
            # Time decay
            time_weight = decay_factor ** (days_ago / 7)  # Weekly decay
            
            fatigue += distance_fatigue * time_weight
        
        # Normalize fatigue score
        return min(fatigue, 1.0)
    
    def _get_default_travel_stats(self) -> Dict[str, float]:
        """Return default travel statistics when insufficient data."""
        return {
            'current_travel_distance': 0.0,
            'is_long_distance_travel': 0.0,
            'days_since_last_match': 7.0,
            'rest_adequacy': 1.0,
            'recent_total_travel': 0.0,
            'avg_recent_travel': 0.0,
            'travel_frequency': 0.0,
            'cumulative_fatigue': 0.0
        }


class SeasonalAdjuster:
    """
    Adjusts features based on seasonal patterns and trends.
    
    Accounts for season progression, weather effects, and motivation changes.
    """
    
    def __init__(self, months_lookback: int = 12):
        self.months_lookback = months_lookback
        self.logger = get_logger("SeasonalAdjuster")
    
    def calculate_seasonal_features(self, 
                                  team_name: str,
                                  match_date: datetime,
                                  season_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate seasonal adjustment features.
        
        Returns:
            Dict with seasonal features and adjustments
        """
        try:
            # Filter data for the team within lookback period
            cutoff_date = match_date - timedelta(days=self.months_lookback * 30)
            team_matches = season_data[
                ((season_data['home_team'] == team_name) | 
                 (season_data['away_team'] == team_name)) &
                (season_data['date'] >= cutoff_date) &
                (season_data['date'] < match_date)
            ].sort_values('date')
            
            if len(team_matches) < 10:
                return self._get_default_seasonal_stats(match_date)
            
            # Calculate various seasonal features
            stats = {
                'season_progress': self._calculate_season_progress(match_date),
                'month_of_year': match_date.month,
                'is_winter_period': float(match_date.month in [12, 1, 2]),
                'is_spring_period': float(match_date.month in [3, 4, 5]),
                'is_summer_period': float(match_date.month in [6, 7, 8]),
                'is_autumn_period': float(match_date.month in [9, 10, 11]),
                'christmas_period': float(match_date.month == 12 and match_date.day > 15),
                'new_year_period': float(match_date.month == 1 and match_date.day < 15),
                'end_season_pressure': self._calculate_end_season_pressure(match_date, team_matches),
                'seasonal_form_trend': self._calculate_seasonal_trend(team_matches, team_name),
                'motivation_index': self._calculate_motivation_index(match_date, team_matches, team_name)
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating seasonal features for {team_name}: {e}")
            return self._get_default_seasonal_stats(match_date)
    
    def _calculate_season_progress(self, match_date: datetime) -> float:
        """Calculate how far into the season we are (0-1)."""
        # Assume season runs from August to May
        if match_date.month >= 8:
            # Current season started in August of same year
            season_start = datetime(match_date.year, 8, 1)
            season_end = datetime(match_date.year + 1, 5, 31)
        else:
            # Current season started in August of previous year
            season_start = datetime(match_date.year - 1, 8, 1)
            season_end = datetime(match_date.year, 5, 31)
        
        season_length = (season_end - season_start).days
        current_progress = (match_date - season_start).days
        
        return min(max(current_progress / season_length, 0.0), 1.0)
    
    def _calculate_end_season_pressure(self, match_date: datetime, team_matches: pd.DataFrame) -> float:
        """Calculate pressure based on end-of-season situation."""
        season_progress = self._calculate_season_progress(match_date)
        
        # More pressure towards end of season
        base_pressure = season_progress ** 2
        
        # Additional pressure if team has been inconsistent recently
        if len(team_matches) > 0:
            recent_matches = team_matches.tail(10)
            recent_results = []
            
            for _, match in recent_matches.iterrows():
                if match['home_team'] == team_matches.iloc[0]['home_team'] or match['home_team'] == team_matches.iloc[0]['away_team']:
                    # Determine if this team won, lost, or drew
                    if match['home_team'] in [team_matches.iloc[0]['home_team'], team_matches.iloc[0]['away_team']]:
                        home_team = match['home_team']
                        if home_team == team_matches.iloc[0]['home_team'] or home_team == team_matches.iloc[0]['away_team']:
                            if match['home_score'] > match['away_score']:
                                result = 3 if home_team == team_matches.iloc[0]['home_team'] or home_team == team_matches.iloc[0]['away_team'] else 0
                            elif match['home_score'] < match['away_score']:
                                result = 0 if home_team == team_matches.iloc[0]['home_team'] or home_team == team_matches.iloc[0]['away_team'] else 3
                            else:
                                result = 1
                            recent_results.append(result)
            
            if recent_results:
                result_variance = np.var(recent_results)
                inconsistency_pressure = result_variance / 9  # Normalize (max variance is 9)
                base_pressure += inconsistency_pressure * 0.5
        
        return min(base_pressure, 1.0)
    
    def _calculate_seasonal_trend(self, team_matches: pd.DataFrame, team_name: str) -> float:
        """Calculate team's performance trend over the season."""
        if len(team_matches) < 6:
            return 0.0
        
        # Split matches into early and late season
        mid_point = len(team_matches) // 2
        early_matches = team_matches.iloc[:mid_point]
        late_matches = team_matches.iloc[mid_point:]
        
        # Calculate average performance for each period
        early_performance = self._calculate_average_performance(early_matches, team_name)
        late_performance = self._calculate_average_performance(late_matches, team_name)
        
        # Return trend (positive = improving, negative = declining)
        trend = late_performance - early_performance
        return max(-1.0, min(1.0, trend))  # Clamp between -1 and 1
    
    def _calculate_average_performance(self, matches: pd.DataFrame, team_name: str) -> float:
        """Calculate average performance score for a set of matches."""
        if len(matches) == 0:
            return 0.0
        
        total_points = 0
        total_goal_diff = 0
        
        for _, match in matches.iterrows():
            if match['home_team'] == team_name:
                # Playing at home
                if match['home_score'] > match['away_score']:
                    points = 3
                elif match['home_score'] == match['away_score']:
                    points = 1
                else:
                    points = 0
                goal_diff = match['home_score'] - match['away_score']
            else:
                # Playing away
                if match['away_score'] > match['home_score']:
                    points = 3
                elif match['away_score'] == match['home_score']:
                    points = 1
                else:
                    points = 0
                goal_diff = match['away_score'] - match['home_score']
            
            total_points += points
            total_goal_diff += goal_diff
        
        # Combine points and goal difference for performance score
        avg_points = total_points / len(matches)
        avg_goal_diff = total_goal_diff / len(matches)
        
        # Normalize to 0-1 scale
        performance = (avg_points + avg_goal_diff + 3) / 6  # Adding 3 to handle negative goal diff
        return max(0.0, min(1.0, performance))
    
    def _calculate_motivation_index(self, match_date: datetime, 
                                  team_matches: pd.DataFrame, 
                                  team_name: str) -> float:
        """Calculate team motivation based on various factors."""
        motivation = 0.5  # Base motivation
        
        # Season timing factors
        season_progress = self._calculate_season_progress(match_date)
        
        # Higher motivation at season start and end
        if season_progress < 0.2:
            motivation += 0.2  # Early season enthusiasm
        elif season_progress > 0.8:
            motivation += 0.3  # End season push
        
        # Recent performance impact
        if len(team_matches) >= 5:
            recent_matches = team_matches.tail(5)
            recent_performance = self._calculate_average_performance(recent_matches, team_name)
            
            # Lower performance might increase motivation (need to improve)
            if recent_performance < 0.3:
                motivation += 0.2
            elif recent_performance > 0.7:
                motivation += 0.1  # Confidence boost
        
        # Holiday period adjustments
        if match_date.month == 12 and match_date.day > 20:
            motivation -= 0.1  # Christmas distraction
        elif match_date.month == 1 and match_date.day < 10:
            motivation -= 0.1  # New year period
        
        return max(0.0, min(1.0, motivation))
    
    def _get_default_seasonal_stats(self, match_date: datetime) -> Dict[str, float]:
        """Return default seasonal statistics when insufficient data."""
        return {
            'season_progress': self._calculate_season_progress(match_date),
            'month_of_year': match_date.month,
            'is_winter_period': float(match_date.month in [12, 1, 2]),
            'is_spring_period': float(match_date.month in [3, 4, 5]),
            'is_summer_period': float(match_date.month in [6, 7, 8]),
            'is_autumn_period': float(match_date.month in [9, 10, 11]),
            'christmas_period': float(match_date.month == 12 and match_date.day > 15),
            'new_year_period': float(match_date.month == 1 and match_date.day < 15),
            'end_season_pressure': 0.5,
            'seasonal_form_trend': 0.0,
            'motivation_index': 0.5
        }


class ContextualFeaturesEngine(FeatureEngineer):
    """
    Main orchestrator for contextual feature engineering.
    
    Combines all contextual analyzers to create comprehensive features.
    Follows Facade pattern to provide simple interface to complex subsystem.
    """
    
    def __init__(self, config: Optional[ContextualConfig] = None):
        self.config = config or ContextualConfig()
        self.logger = get_logger("ContextualFeaturesEngine")
        
        # Initialize components
        self.home_calculator = HomeAdvantageCalculator(self.config.home_advantage_lookback)
        self.rivalry_analyzer = RivalryAnalyzer(self.config.rivalry_threshold, self.config.derby_keywords)
        self.travel_calculator = TravelFatigueCalculator(self.config.travel_distance_threshold)
        self.seasonal_adjuster = SeasonalAdjuster(self.config.seasonal_months_lookback)
        
        self._feature_names = None
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create contextual features for all matches in the dataset.
        
        Args:
            data: DataFrame with match data including date, teams, scores
            
        Returns:
            DataFrame with added contextual features
        """
        try:
            self.logger.info(f"Creating contextual features for {len(data)} matches")
            
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
            
            self.logger.info(f"Created {len(self.get_feature_names())} contextual features")
            return result
            
        except Exception as e:
            self.logger.error(f"Error creating contextual features: {e}")
            raise FeatureEngineeringError(f"Contextual feature creation failed: {e}")
    
    def _create_match_features(self, match: pd.Series, full_data: pd.DataFrame) -> Dict[str, float]:
        """Create contextual features for a single match."""
        home_team = match['home_team']
        away_team = match['away_team']
        match_date = pd.to_datetime(match['date'])
        
        # Get historical data (all matches before this one)
        historical_data = full_data[full_data['date'] < match['date']]
        
        features = {}
        
        # Home advantage features
        home_adv = self.home_calculator.calculate_home_advantage(
            historical_data, home_team, match_date
        )
        for key, value in home_adv.items():
            features[f"home_{key}"] = value
        
        away_adv = self.home_calculator.calculate_home_advantage(
            historical_data, away_team, match_date
        )
        for key, value in away_adv.items():
            features[f"away_{key}"] = value
        
        # Rivalry features
        rivalry = self.rivalry_analyzer.analyze_rivalry(
            home_team, away_team, historical_data, match_date
        )
        features.update(rivalry)
        
        # Travel fatigue features
        home_travel = self.travel_calculator.calculate_travel_impact(
            home_team, home_team, away_team, historical_data, match_date
        )
        for key, value in home_travel.items():
            features[f"home_travel_{key}"] = value
        
        away_travel = self.travel_calculator.calculate_travel_impact(
            away_team, home_team, away_team, historical_data, match_date
        )
        for key, value in away_travel.items():
            features[f"away_travel_{key}"] = value
        
        # Seasonal features
        home_seasonal = self.seasonal_adjuster.calculate_seasonal_features(
            home_team, match_date, historical_data
        )
        for key, value in home_seasonal.items():
            features[f"home_seasonal_{key}"] = value
        
        away_seasonal = self.seasonal_adjuster.calculate_seasonal_features(
            away_team, match_date, historical_data
        )
        for key, value in away_seasonal.items():
            features[f"away_seasonal_{key}"] = value
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Return list of all contextual feature names."""
        if self._feature_names is None:
            self._feature_names = self._generate_feature_names()
        return self._feature_names
    
    def _generate_feature_names(self) -> List[str]:
        """Generate complete list of feature names."""
        feature_names = []
        
        # Home advantage features (for both teams)
        home_adv_features = [
            'home_advantage_win_rate', 'home_advantage_goals_for', 'home_advantage_goals_against',
            'home_advantage_goal_diff', 'home_matches_ratio', 'home_form_strength',
            'away_form_strength', 'venue_consistency'
        ]
        
        for team_prefix in ['home_', 'away_']:
            for feature in home_adv_features:
                feature_names.append(f"{team_prefix}{feature}")
        
        # Rivalry features
        rivalry_features = [
            'h2h_total_matches', 'h2h_home_team_wins', 'h2h_away_team_wins', 'h2h_draws',
            'h2h_home_team_win_rate', 'h2h_away_team_win_rate', 'h2h_draw_rate',
            'h2h_recent_form_home', 'h2h_avg_goals_home_team', 'h2h_avg_goals_away_team',
            'h2h_competitive_balance', 'h2h_match_volatility', 'is_derby', 'rivalry_intensity'
        ]
        feature_names.extend(rivalry_features)
        
        # Travel features (for both teams)
        travel_features = [
            'current_travel_distance', 'is_long_distance_travel', 'days_since_last_match',
            'rest_adequacy', 'recent_total_travel', 'avg_recent_travel',
            'travel_frequency', 'cumulative_fatigue'
        ]
        
        for team_prefix in ['home_travel_', 'away_travel_']:
            for feature in travel_features:
                feature_names.append(f"{team_prefix}{feature}")
        
        # Seasonal features (for both teams)
        seasonal_features = [
            'season_progress', 'month_of_year', 'is_winter_period', 'is_spring_period',
            'is_summer_period', 'is_autumn_period', 'christmas_period', 'new_year_period',
            'end_season_pressure', 'seasonal_form_trend', 'motivation_index'
        ]
        
        for team_prefix in ['home_seasonal_', 'away_seasonal_']:
            for feature in seasonal_features:
                feature_names.append(f"{team_prefix}{feature}")
        
        return feature_names
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Return estimated feature importance scores.
        
        Based on domain knowledge and typical impact on match outcomes.
        """
        importance_scores = {}
        
        # High importance features
        high_impact = [
            'home_home_advantage_win_rate', 'away_home_advantage_win_rate',
            'h2h_home_team_win_rate', 'h2h_away_team_win_rate',
            'rivalry_intensity', 'is_derby'
        ]
        
        # Medium importance features
        medium_impact = [
            'home_home_form_strength', 'away_away_form_strength',
            'home_travel_cumulative_fatigue', 'away_travel_cumulative_fatigue',
            'home_seasonal_motivation_index', 'away_seasonal_motivation_index'
        ]
        
        # Lower importance features
        low_impact = [
            'home_travel_current_travel_distance', 'away_travel_current_travel_distance',
            'home_seasonal_month_of_year', 'away_seasonal_month_of_year'
        ]
        
        # Assign importance scores
        for feature in self.get_feature_names():
            if any(high_feat in feature for high_feat in high_impact):
                importance_scores[feature] = 0.8
            elif any(med_feat in feature for med_feat in medium_impact):
                importance_scores[feature] = 0.6
            elif any(low_feat in feature for low_feat in low_impact):
                importance_scores[feature] = 0.3
            else:
                importance_scores[feature] = 0.4  # Default importance
        
        return importance_scores