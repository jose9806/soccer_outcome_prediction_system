from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Tuple, Union


@dataclass
class MatchStats:
    # Original stats
    expected_goals: Optional[Tuple[float, float]] = None
    possession: Optional[Tuple[int, int]] = None
    shots_on_goal: Optional[Tuple[int, int]] = None
    shots_off_goal: Optional[Tuple[int, int]] = None
    blocked_shots: Optional[Tuple[int, int]] = None
    big_chances: Optional[Tuple[int, int]] = None
    corner_kicks: Optional[Tuple[int, int]] = None
    shots_inside_box: Optional[Tuple[int, int]] = None
    shots_outside_box: Optional[Tuple[int, int]] = None

    # Additional stats from image
    goal_attempts: Optional[Tuple[int, int]] = None
    goalkeeper_saves: Optional[Tuple[int, int]] = None
    offsides: Optional[Tuple[int, int]] = None
    fouls: Optional[Tuple[int, int]] = None
    yellow_cards: Optional[Tuple[int, int]] = None
    red_cards: Optional[Tuple[int, int]] = None


class OddsType(Enum):
    """Enumeration of different betting market types."""

    MATCH_WINNER = "1X2"  # Home win, Draw, Away win
    OVER_UNDER = "Over/Under"  # Over/Under a specific goal count
    BOTH_TEAMS_TO_SCORE = "BTTS"  # Both teams to score (Yes/No)
    ASIAN_HANDICAP = "Asian Handicap"  # Adjusted score lines with handicaps
    DOUBLE_CHANCE = "Double Chance"  # Two possible outcomes combined
    DRAW_NO_BET = "Draw No Bet"  # Home or Away with Draw void
    CORRECT_SCORE = "Correct Score"  # Exact final score
    HALFTIME_FULLTIME = "HT/FT"  # Result at half time and full time
    ODD_EVEN = "Odd/Even"  # Total goals odd or even


@dataclass
class BaseOdds:
    """Base class for all odds types."""

    bookmaker: str
    timestamp: datetime
    period: str  # Full Time, 1st Half, 2nd Half
    market_type: OddsType


@dataclass
class MatchWinnerOdds(BaseOdds):
    """1X2 odds - Home win, Draw, Away win."""

    home_win: float
    draw: float
    away_win: float


@dataclass
class OverUnderOdds(BaseOdds):
    """Over/Under odds for total goals."""

    total: float
    over: float
    under: float


@dataclass
class BothTeamsToScoreOdds(BaseOdds):
    """Both Teams To Score odds."""

    yes: float
    no: float


@dataclass
class AsianHandicapOdds(BaseOdds):
    """Asian Handicap odds."""

    handicap: float
    home: float
    away: float


@dataclass
class DoubleChanceOdds(BaseOdds):
    """Double Chance odds."""

    home_or_draw: float
    draw_or_away: float
    home_or_away: float


@dataclass
class DrawNoBetOdds(BaseOdds):
    """Draw No Bet odds."""

    home: float
    away: float


@dataclass
class CorrectScoreOdds(BaseOdds):
    """Correct Score odds."""

    score: str  # Format: "2-1", "1-1", etc.
    odds: float


@dataclass
class HalftimeFulltimeOdds(BaseOdds):
    """Half Time/Full Time odds."""

    result_combination: str  # Format: "Home/Home", "Draw/Away", etc.
    odds: float


@dataclass
class OddEvenOdds(BaseOdds):
    """Odd/Even total goals odds."""

    odd: float
    even: float


# Type alias for all odds types
OddsVariant = Union[
    MatchWinnerOdds,
    OverUnderOdds,
    BothTeamsToScoreOdds,
    AsianHandicapOdds,
    DoubleChanceOdds,
    DrawNoBetOdds,
    CorrectScoreOdds,
    HalftimeFulltimeOdds,
    OddEvenOdds,
]


@dataclass
class MatchOdds:
    """Container for all odds data for a match."""

    match_id: str
    match_winner_odds: List[MatchWinnerOdds] = field(default_factory=list)
    over_under_odds: List[OverUnderOdds] = field(default_factory=list)
    btts_odds: List[BothTeamsToScoreOdds] = field(default_factory=list)
    asian_handicap_odds: List[AsianHandicapOdds] = field(default_factory=list)
    double_chance_odds: List[DoubleChanceOdds] = field(default_factory=list)
    draw_no_bet_odds: List[DrawNoBetOdds] = field(default_factory=list)
    correct_score_odds: List[CorrectScoreOdds] = field(default_factory=list)
    halftime_fulltime_odds: List[HalftimeFulltimeOdds] = field(default_factory=list)
    odd_even_odds: List[OddEvenOdds] = field(default_factory=list)

    def add_odds(self, odds: OddsVariant) -> None:
        """Add an odds instance to the appropriate list based on its type."""
        if isinstance(odds, MatchWinnerOdds):
            self.match_winner_odds.append(odds)
        elif isinstance(odds, OverUnderOdds):
            self.over_under_odds.append(odds)
        elif isinstance(odds, BothTeamsToScoreOdds):
            self.btts_odds.append(odds)
        elif isinstance(odds, AsianHandicapOdds):
            self.asian_handicap_odds.append(odds)
        elif isinstance(odds, DoubleChanceOdds):
            self.double_chance_odds.append(odds)
        elif isinstance(odds, DrawNoBetOdds):
            self.draw_no_bet_odds.append(odds)
        elif isinstance(odds, CorrectScoreOdds):
            self.correct_score_odds.append(odds)
        elif isinstance(odds, HalftimeFulltimeOdds):
            self.halftime_fulltime_odds.append(odds)
        elif isinstance(odds, OddEvenOdds):
            self.odd_even_odds.append(odds)


from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

from src.scraping.models.soccer_extraction import MatchStats, MatchOdds


@dataclass
class Match:
    """
    Represents a soccer match with all related data including basic information,
    statistics for different periods, and comprehensive betting odds.
    """

    # Basic match identification
    match_id: str

    # Match metadata
    date: datetime
    competition: str
    season: str
    stage: str

    # Team and score information
    home_team: str
    away_team: str
    home_score: int
    away_score: int

    # Match statistics for different periods
    first_half_stats: Optional[MatchStats] = None
    second_half_stats: Optional[MatchStats] = None
    full_time_stats: Optional[MatchStats] = None

    # Comprehensive odds data
    odds: Optional[MatchOdds] = None

    # For backward compatibility
    odds_history: List[Any] = field(default_factory=list)

    # Additional metadata that might be scraped
    venue: Optional[str] = None
    referee: Optional[str] = None
    attendance: Optional[int] = None
    weather: Optional[str] = None

    # Match status information
    status: Optional[str] = None  # e.g., "Finished", "Postponed", "Cancelled"

    def __post_init__(self):
        """Initialize additional derived properties after initialization."""
        # Handle legacy odds_history conversion if needed
        if self.odds_history and not self.odds:
            # If we have old-style odds but no new comprehensive odds,
            # we can convert them to the new format if needed
            pass

    @property
    def result(self) -> str:
        """Return the match result from the home team's perspective (W/D/L)."""
        if self.home_score > self.away_score:
            return "W"
        elif self.home_score == self.away_score:
            return "D"
        else:
            return "L"

    @property
    def total_goals(self) -> int:
        """Return the total number of goals scored in the match."""
        return self.home_score + self.away_score

    @property
    def both_teams_scored(self) -> bool:
        """Return True if both teams scored at least one goal."""
        return self.home_score > 0 and self.away_score > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Match object to a dictionary for serialization."""
        result = {
            "match_id": self.match_id,
            "date": self.date.isoformat(),
            "competition": self.competition,
            "season": self.season,
            "stage": self.stage,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_score": self.home_score,
            "away_score": self.away_score,
            "result": self.result,
            "total_goals": self.total_goals,
            "both_teams_scored": self.both_teams_scored,
        }

        # Add optional fields if they exist
        if self.venue:
            result["venue"] = self.venue
        if self.referee:
            result["referee"] = self.referee
        if self.attendance:
            result["attendance"] = self.attendance
        if self.weather:
            result["weather"] = self.weather
        if self.status:
            result["status"] = self.status

        # Add statistics if available
        if self.full_time_stats:
            result["full_time_stats"] = self._stats_to_dict(self.full_time_stats)
        if self.first_half_stats:
            result["first_half_stats"] = self._stats_to_dict(self.first_half_stats)
        if self.second_half_stats:
            result["second_half_stats"] = self._stats_to_dict(self.second_half_stats)

        # Add odds information if available
        if self.odds:
            result["odds"] = self._odds_to_dict(self.odds)

        return result

    def _stats_to_dict(self, stats: MatchStats) -> Dict[str, Any]:
        """Convert a MatchStats object to a dictionary."""
        # Start with an empty dictionary
        stats_dict = {}

        # Add each non-None attribute
        if stats.possession is not None:
            stats_dict["possession"] = {
                "home": stats.possession[0],
                "away": stats.possession[1],
            }
        if stats.shots_on_goal is not None:
            stats_dict["shots_on_goal"] = {
                "home": stats.shots_on_goal[0],
                "away": stats.shots_on_goal[1],
            }
        if stats.shots_off_goal is not None:
            stats_dict["shots_off_goal"] = {
                "home": stats.shots_off_goal[0],
                "away": stats.shots_off_goal[1],
            }
        if stats.blocked_shots is not None:
            stats_dict["blocked_shots"] = {
                "home": stats.blocked_shots[0],
                "away": stats.blocked_shots[1],
            }
        if stats.corner_kicks is not None:
            stats_dict["corner_kicks"] = {
                "home": stats.corner_kicks[0],
                "away": stats.corner_kicks[1],
            }
        if stats.goal_attempts is not None:
            stats_dict["goal_attempts"] = {
                "home": stats.goal_attempts[0],
                "away": stats.goal_attempts[1],
            }
        if stats.goalkeeper_saves is not None:
            stats_dict["goalkeeper_saves"] = {
                "home": stats.goalkeeper_saves[0],
                "away": stats.goalkeeper_saves[1],
            }
        if stats.offsides is not None:
            stats_dict["offsides"] = {
                "home": stats.offsides[0],
                "away": stats.offsides[1],
            }
        if stats.fouls is not None:
            stats_dict["fouls"] = {"home": stats.fouls[0], "away": stats.fouls[1]}
        if stats.yellow_cards is not None:
            stats_dict["yellow_cards"] = {
                "home": stats.yellow_cards[0],
                "away": stats.yellow_cards[1],
            }
        if stats.red_cards is not None:
            stats_dict["red_cards"] = {
                "home": stats.red_cards[0],
                "away": stats.red_cards[1],
            }

        # Add additional stats that might be available
        if stats.expected_goals is not None:
            stats_dict["expected_goals"] = {
                "home": stats.expected_goals[0],
                "away": stats.expected_goals[1],
            }
        if stats.big_chances is not None:
            stats_dict["big_chances"] = {
                "home": stats.big_chances[0],
                "away": stats.big_chances[1],
            }
        if stats.shots_inside_box is not None:
            stats_dict["shots_inside_box"] = {
                "home": stats.shots_inside_box[0],
                "away": stats.shots_inside_box[1],
            }
        if stats.shots_outside_box is not None:
            stats_dict["shots_outside_box"] = {
                "home": stats.shots_outside_box[0],
                "away": stats.shots_outside_box[1],
            }

        return stats_dict

    def _odds_to_dict(self, odds: MatchOdds) -> Dict[str, Any]:
        """Convert a MatchOdds object to a dictionary."""
        odds_dict: Dict[str, Any] = {
            "match_id": odds.match_id,
        }

        # Add each type of odds if they exist
        if odds.match_winner_odds:
            odds_dict["match_winner"] = [
                {
                    "bookmaker": o.bookmaker,
                    "timestamp": o.timestamp.isoformat(),
                    "period": o.period,
                    "home_win": o.home_win,
                    "draw": o.draw,
                    "away_win": o.away_win,
                }
                for o in odds.match_winner_odds
            ]

        if odds.over_under_odds:
            odds_dict["over_under"] = [
                {
                    "bookmaker": o.bookmaker,
                    "timestamp": o.timestamp.isoformat(),
                    "period": o.period,
                    "total": o.total,
                    "over": o.over,
                    "under": o.under,
                }
                for o in odds.over_under_odds
            ]

        if odds.btts_odds:
            odds_dict["both_teams_to_score"] = [
                {
                    "bookmaker": o.bookmaker,
                    "timestamp": o.timestamp.isoformat(),
                    "period": o.period,
                    "yes": o.yes,
                    "no": o.no,
                }
                for o in odds.btts_odds
            ]

        # Add other odds types similarly
        if odds.asian_handicap_odds:
            odds_dict["asian_handicap"] = [
                {
                    "bookmaker": o.bookmaker,
                    "timestamp": o.timestamp.isoformat(),
                    "period": o.period,
                    "handicap": o.handicap,
                    "home": o.home,
                    "away": o.away,
                }
                for o in odds.asian_handicap_odds
            ]

        if odds.double_chance_odds:
            odds_dict["double_chance"] = [
                {
                    "bookmaker": o.bookmaker,
                    "timestamp": o.timestamp.isoformat(),
                    "period": o.period,
                    "home_or_draw": o.home_or_draw,
                    "draw_or_away": o.draw_or_away,
                    "home_or_away": o.home_or_away,
                }
                for o in odds.double_chance_odds
            ]

        if odds.draw_no_bet_odds:
            odds_dict["draw_no_bet"] = [
                {
                    "bookmaker": o.bookmaker,
                    "timestamp": o.timestamp.isoformat(),
                    "period": o.period,
                    "home": o.home,
                    "away": o.away,
                }
                for o in odds.draw_no_bet_odds
            ]

        if odds.correct_score_odds:
            odds_dict["correct_score"] = [
                {
                    "bookmaker": o.bookmaker,
                    "timestamp": o.timestamp.isoformat(),
                    "period": o.period,
                    "score": o.score,
                    "odds": o.odds,
                }
                for o in odds.correct_score_odds
            ]

        if odds.halftime_fulltime_odds:
            odds_dict["halftime_fulltime"] = [
                {
                    "bookmaker": o.bookmaker,
                    "timestamp": o.timestamp.isoformat(),
                    "period": o.period,
                    "result_combination": o.result_combination,
                    "odds": o.odds,
                }
                for o in odds.halftime_fulltime_odds
            ]

        if odds.odd_even_odds:
            odds_dict["odd_even"] = [
                {
                    "bookmaker": o.bookmaker,
                    "timestamp": o.timestamp.isoformat(),
                    "period": o.period,
                    "odd": o.odd,
                    "even": o.even,
                }
                for o in odds.odd_even_odds
            ]

        return odds_dict


@dataclass
class Season:
    """Represents a football season with its associated matches."""

    year: int
    match_urls: List[str]
    total_matches: int
    tournaments: List[Dict[str, str]] = field(default_factory=list)
