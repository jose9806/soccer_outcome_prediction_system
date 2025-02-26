import logging
from typing import Dict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ScrapingConfig:
    # Base URLs
    BASE_URL: str = "https://www.soccer24.com"
    COLOMBIA_URL: str = f"{BASE_URL}/colombia/primera-a"
    ARCHIVE_URL: str = f"{COLOMBIA_URL}/archive"

    # Timing configurations
    PAGE_LOAD_TIMEOUT: int = 30
    ELEMENT_WAIT_TIMEOUT: int = 10
    REQUEST_DELAY: float = 2.0
    LOG_LEVEL: int = logging.DEBUG

    # Selenium selectors using default_factory for mutable defaults
    SELECTORS: Dict[str, str] = field(
        default_factory=lambda: {
            # Archive page selectors
            "archive_row": "div.archive__row",
            "archive_text": "a.archive__text.archive__text--clickable",
            "archive_season": "div.archive__season",
            "archive_winner": "div.archive__winner",
            "show_more_button": "a.event_more.event_more--static",
            # Match selectors
            "match_link": "a[href*='/match/']",
            "match_date": "div.duelParticipant__startTime, .eventTime, .eventDate",
            "home_team": ".participant__participantName, .duelParticipant__home .participant__participantName",
            "away_team": ".participant__participantName, .duelParticipant__away .participant__participantName",
            "score": ".detailScore__wrapper, .event__scores",
            "match_status": ".detailScore__status, .event__stage--block",
            "competition": ".tournamentHeader__country, .event__title--type, .tournamentHeaderDescription",
            "season": ".tournamentHeader__season",
            # Cookie consent
            "cookie_consent": "button[aria-label='Accept cookies'], button.css-47sehv, button#onetrust-accept-btn-handler",
            "stats_tab": "a[href*='match-statistics'], button[class*='wcl-tab'], div[class*='wcl-tabs'] button, .tabs-wrapper a:nth-child(2), a.filterOverTab, a[title='Statistics'], a[href*='#/match-summary/match-statistics']",
            # Remove jQuery-style :contains() and use direct attribute selectors where possible
            "possession_stat": "div[title='Ball Possession'], .stat__categoryName, strong[data-testid^='wcl-scores-simpleText']",
            "goal_attempts": "div[title='Goal Attempts'], .stat__row, strong[data-testid^='wcl-scores-simpleText']",
            "shots_on_goal": "div[title='Shots on Goal'], .stat__row, strong[data-testid^='wcl-scores-simpleText']",
            "shots_off_goal": "div[title='Shots off Goal'], .stat__row, strong[data-testid^='wcl-scores-simpleText']",
            "blocked_shots": "div[title='Blocked Shots'], .stat__row, strong[data-testid^='wcl-scores-simpleText']",
            "corner_kicks": "div[title='Corner Kicks'], .stat__row, strong[data-testid^='wcl-scores-simpleText']",
            "goalkeeper_saves": "div[title='Goalkeeper Saves'], .stat__row, strong[data-testid^='wcl-scores-simpleText']",
            "offsides": "div[title='Offsides'], .stat__row, strong[data-testid^='wcl-scores-simpleText']",
            "fouls": "div[title='Fouls'], .stat__row, strong[data-testid^='wcl-scores-simpleText']",
            "yellow_cards": "div[title='Yellow Cards'], .stat__row, strong[data-testid^='wcl-scores-simpleText']",
            "red_cards": "div[title='Red Cards'], .stat__row, strong[data-testid^='wcl-scores-simpleText']",
            # Additional stats selectors
            "big_chances": "div[title='Big Chances'], .stat__row, strong[data-testid^='wcl-scores-simpleText']",
            "shots_inside_box": "div[title='Shots inside box'], .stat__row, strong[data-testid^='wcl-scores-simpleText']",
            "shots_outside_box": "div[title='Shots outside box'], .stat__row, strong[data-testid^='wcl-scores-simpleText']",
            "expected_goals": "div[title='Expected goals (xG)'], .stat__row, strong[data-testid^='wcl-scores-simpleText']",
            "wcl_stats_container": "div[data-testid='wcl-statistics'], div[class*='wcl-row_'], div[class*='-row_'], div[class='section']",
            # The key issue is with this selector - we need to target both direct text elements and strong elements
            "wcl_stats_label": "strong[data-testid^='wcl-scores-simpleText'], strong[class*='wcl-simpleText_'], div[class*='wcl-category_'], div[class*='-category_']",
            "wcl_home_value": "div[class*='wcl-homeValue_'], div[class*='-homeValue_'], div[class*='wcl-value_'][class*='_IyoQw']",
            "wcl_away_value": "div[class*='wcl-awayValue_'], div[class*='-awayValue_'], div[class*='wcl-value_'][class*='_rQvxS']",
            # Odds tab and navigation
            "odds_tab": "a[href*='odds-comparison'], a[href*='1x2-odds'], .tabs-wrapper a:nth-child(3), a[title='1X2']",
            "bookmaker_rows": "tr.oddsTab__tableRow, .ui-table__row, tr",
            "bookmaker_name": ".oddsCell__bookmakerPart, .teamHeader__logo, .bookmaker__name, .bookmaker img, .oddsCell__bookmakerPart",
            "odds_values": "td.oddsCell__odd, .ui-table__cell:not(.bookmaker), .oddsCell__odd, a.oddsCell__odd",
            # Market selectors for different betting types
            "market_tabs": ".filterOverTab, a[title], a.wcl-tabs_jy59b, div.wcl-tabs_jy59b",
            "period_tabs": ".wcl-tab_y-fEC, .event__tab, .oddsTabOption, div[role='tab']",
            # Specific market selectors
            "1x2_tab": "a[title='1X2'], a[href*='1x2-odds'], .filterOverTab:contains('1X2')",
            "over_under_tab": "a[title='Over/Under'], a[href*='over-under'], .filterOverTab:contains('Over/Under')",
            "btts_tab": "a[title='Both teams to score'], a[href*='both-teams-to-score'], .filterOverTab:contains('Both teams to score')",
            "asian_handicap_tab": "a[title='Asian handicap'], a[href*='asian-handicap'], .filterOverTab:contains('Asian handicap')",
            "double_chance_tab": "a[title='Double chance'], a[href*='double-chance'], .filterOverTab:contains('Double chance')",
            "draw_no_bet_tab": "a[title='Draw no bet'], a[href*='draw-no-bet'], .filterOverTab:contains('Draw no bet')",
            "correct_score_tab": "a[title='Correct score'], a[href*='correct-score'], .filterOverTab:contains('Correct score')",
            "ht_ft_tab": "a[title='Half Time/Full Time'], a[href*='ht-ft'], .filterOverTab:contains('Half Time/Full Time')",
            "odd_even_tab": "a[title='Odd/Even'], a[href*='odd-even'], .filterOverTab:contains('Odd/Even')",
        }
    )

    # Storage configuration
    DATA_DIR: Path = Path("data/raw")
    LOG_DIR: Path = Path("src/logs")

    # Rate limiting
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 5.0

    def __post_init__(self):
        # Ensure data directory exists
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Ensure logs directory exists
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
