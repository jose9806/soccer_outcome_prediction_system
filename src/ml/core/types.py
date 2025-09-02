"""
Type definitions and data classes for the ML betting system.

Provides strong typing throughout the system to improve code reliability
and enable better IDE support and static analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
from enum import Enum
import numpy as np


class MatchOutcome(str, Enum):
    """Possible match outcomes."""
    HOME_WIN = "1"
    DRAW = "X" 
    AWAY_WIN = "2"


class BetMarket(str, Enum):
    """Available betting markets."""
    MATCH_WINNER = "1x2"
    OVER_UNDER = "over_under"
    BOTH_TEAMS_SCORE = "btts"
    ASIAN_HANDICAP = "handicap"
    CORRECT_SCORE = "correct_score"


@dataclass(frozen=True)
class TeamInfo:
    """Information about a team."""
    name: str
    id: str
    country: str
    league: str


@dataclass
class MatchData:
    """Raw match data from scraping."""
    match_id: str
    date: datetime
    home_team: TeamInfo
    away_team: TeamInfo
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    status: str = "SCHEDULED"
    competition: str = ""
    season: str = ""
    stage: str = ""
    
    # Statistics
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    # Odds data
    odds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Context
    context: Dict[str, Any] = field(default_factory=dict)

    @property
    def outcome(self) -> Optional[MatchOutcome]:
        """Determine match outcome from scores."""
        if self.home_score is None or self.away_score is None:
            return None
        
        if self.home_score > self.away_score:
            return MatchOutcome.HOME_WIN
        elif self.home_score < self.away_score:
            return MatchOutcome.AWAY_WIN
        else:
            return MatchOutcome.DRAW

    @property
    def total_goals(self) -> Optional[int]:
        """Calculate total goals scored."""
        if self.home_score is None or self.away_score is None:
            return None
        return self.home_score + self.away_score


@dataclass
class MatchFeatures:
    """Engineered features for ML models."""
    match_id: str
    
    # Team strength features
    home_strength: float
    away_strength: float
    strength_diff: float
    
    # Form features  
    home_form: float
    away_form: float
    
    # Head-to-head features
    h2h_home_wins: int
    h2h_draws: int
    h2h_away_wins: int
    h2h_avg_goals: float
    
    # Context features
    home_advantage: float
    rest_days_home: int
    rest_days_away: int
    
    # Advanced features
    features: Dict[str, float] = field(default_factory=dict)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML models."""
        base_features = [
            self.home_strength, self.away_strength, self.strength_diff,
            self.home_form, self.away_form,
            self.h2h_home_wins, self.h2h_draws, self.h2h_away_wins, self.h2h_avg_goals,
            self.home_advantage, self.rest_days_home, self.rest_days_away
        ]
        
        additional_features = list(self.features.values())
        return np.array(base_features + additional_features)
    
    def get_feature_names(self) -> List[str]:
        """Get ordered list of feature names."""
        base_names = [
            'home_strength', 'away_strength', 'strength_diff',
            'home_form', 'away_form', 
            'h2h_home_wins', 'h2h_draws', 'h2h_away_wins', 'h2h_avg_goals',
            'home_advantage', 'rest_days_home', 'rest_days_away'
        ]
        
        additional_names = list(self.features.keys())
        return base_names + additional_names


@dataclass
class Prediction:
    """Model prediction for a match."""
    match_id: str
    model_name: str
    timestamp: datetime
    
    # Primary predictions
    probabilities: Dict[MatchOutcome, float]
    predicted_outcome: MatchOutcome
    confidence: float
    
    # Additional predictions
    expected_goals_home: Optional[float] = None
    expected_goals_away: Optional[float] = None
    over_under_2_5: Optional[float] = None
    both_teams_score: Optional[float] = None
    
    # Model metadata
    model_version: str = "1.0"
    features_used: List[str] = field(default_factory=list)
    
    @property
    def max_probability(self) -> float:
        """Return highest probability among outcomes."""
        return max(self.probabilities.values())
    
    @property
    def entropy(self) -> float:
        """Calculate prediction entropy (uncertainty measure)."""
        probs = np.array(list(self.probabilities.values()))
        probs = probs[probs > 0]  # Avoid log(0)
        return -np.sum(probs * np.log2(probs))


@dataclass
class ModelMetrics:
    """Performance metrics for ML models."""
    accuracy: float
    precision: Dict[MatchOutcome, float]
    recall: Dict[MatchOutcome, float]
    f1_score: Dict[MatchOutcome, float]
    
    # Probabilistic metrics
    brier_score: float
    log_loss: float
    roc_auc: float
    
    # Betting-specific metrics
    roi: Optional[float] = None
    profit: Optional[float] = None
    num_bets: Optional[int] = None
    win_rate: Optional[float] = None
    
    # Additional metrics
    confusion_matrix: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None


@dataclass 
class BacktestResult:
    """Results from a backtesting simulation."""
    start_date: datetime
    end_date: datetime
    initial_bankroll: float
    final_bankroll: float
    
    # Performance metrics
    total_return: float
    roi: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Betting statistics
    total_bets: int
    winning_bets: int
    losing_bets: int
    win_rate: float
    
    # Detailed results
    daily_pnl: List[Tuple[datetime, float]]
    bet_history: List[Dict[str, Any]]
    
    # Risk metrics
    var_95: float  # Value at Risk at 95% confidence
    expected_shortfall: float
    
    @property
    def profit(self) -> float:
        """Calculate total profit/loss."""
        return self.final_bankroll - self.initial_bankroll
    
    @property
    def average_bet_size(self) -> float:
        """Calculate average bet size."""
        if self.total_bets == 0:
            return 0.0
        total_stakes = sum(bet['stake'] for bet in self.bet_history)
        return total_stakes / self.total_bets


@dataclass
class BetRecommendation:
    """Betting recommendation from the system."""
    match_id: str
    prediction: Prediction
    market: BetMarket
    selection: str
    recommended_odds: float
    available_odds: Dict[str, float]  # bookmaker -> odds
    
    # Betting decision
    should_bet: bool
    stake: float
    expected_value: float
    
    # Risk assessment
    confidence_level: float
    risk_rating: str  # LOW, MEDIUM, HIGH
    
    # Rationale
    reasoning: List[str] = field(default_factory=list)