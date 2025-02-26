import time
import re
from typing import List, Sequence, Any
from datetime import datetime
from urllib.parse import urlparse

from selenium.webdriver.common.by import By

from selenium.common.exceptions import (
    TimeoutException,
)

from src.config.logger import Logger
from src.scraping.models.soccer_extraction import (
    MatchOdds,
    OddsType,
    MatchWinnerOdds,
    OverUnderOdds,
    BothTeamsToScoreOdds,
    CorrectScoreOdds,
    OddEvenOdds,
    OddsVariant,
)
from src.scraping.scrapers.base import BaseScraper


class OddsScraper(BaseScraper):
    """Streamlined scraper for collecting betting odds from specific markets."""

    def __init__(self, driver, config, **kwargs):
        super().__init__(driver, config)

        self.logger = Logger(
            name="OddsScraper",
            color="magenta",
            level=(self.config.LOG_LEVEL if hasattr(self.config, "LOG_LEVEL") else 20),
            file_output="src/logs/odds_scraper.log",
        )
        self.logger.info("OddsScraper initialized")

        self.market_mapping = {
            "1X2": OddsType.MATCH_WINNER,
            "Over/Under": OddsType.OVER_UNDER,
            "Both teams to score": OddsType.BOTH_TEAMS_TO_SCORE,
            "Correct score": OddsType.CORRECT_SCORE,
            "Odd/Even": OddsType.ODD_EVEN,
        }

        # Markets we're interested in
        self.target_markets = [
            "1X2",
            "Over/Under",
            "Both teams to score",
            "Correct score",
            "Odd/Even",
        ]

        self.fast_mode = kwargs.get("fast_odds", False)

        # Market limit for testing
        self.market_limit = None

    def set_fast_mode(self, fast_mode: bool):
        """Set fast mode (only scrape 1X2 market)"""
        self.fast_mode = fast_mode
        self.logger.info(f"Fast mode set to: {fast_mode}")

    def set_market_limit(self, limit: int):
        """Set market limit for testing"""
        self.market_limit = limit
        self.logger.info(f"Market limit set to: {limit}")

    def scrape(self, match_url: str) -> MatchOdds:
        """
        Scrape betting odds from the specified markets with improved reliability.

        Args:
            match_url: URL of the match to scrape

        Returns:
            MatchOdds object containing all scraped odds data
        """
        self.logger.add_context(url=match_url)
        self.logger.info("Starting odds scraping process")

        # Extract match ID from URL for tracking
        match_id = self._extract_match_id(match_url)
        match_odds = MatchOdds(match_id=match_id)

        try:
            # Navigate to the match page if needed
            current_url = self.driver.current_url
            if not current_url == match_url and not "#/odds-comparison" in current_url:
                self.logger.debug(f"Navigating to match URL")
                self.driver.get(match_url)
                time.sleep(self.config.REQUEST_DELAY)

            # Determining if we need to click the odds tab
            click_odds_tab = True
            if "#/odds-comparison" in self.driver.current_url:
                click_odds_tab = False
                self.logger.debug("Already on odds page, skipping tab click")

            # Click on odds tab if needed
            if click_odds_tab:
                try:
                    self.logger.debug("Clicking on odds tab")
                    # Try multiple selectors for odds tab
                    odds_tab_selectors = [
                        self.config.SELECTORS["odds_tab"],
                        "a[href*='odds-comparison']",
                        "a[href*='1x2-odds']",
                        ".tabs__tab:nth-child(2)",
                        "//a[contains(text(), 'Odds')]",
                        "//a[contains(@href, 'odds')]",
                    ]

                    clicked = False
                    for selector in odds_tab_selectors:
                        try:
                            if isinstance(selector, str) and selector.startswith("//"):
                                elements = self.driver.find_elements(By.XPATH, selector)
                            else:
                                elements = self.driver.find_elements(
                                    By.CSS_SELECTOR, selector
                                )

                            if elements:
                                elements[0].click()
                                self.logger.debug(
                                    f"Clicked odds tab using selector: {selector}"
                                )
                                clicked = True
                                time.sleep(1.5)
                                break
                        except Exception:
                            continue

                    if not clicked:
                        # Try JavaScript click as last resort
                        try:
                            script = "var links = document.querySelectorAll('a'); for (var i = 0; i < links.length; i++) { if (links[i].href.includes('odds') || links[i].textContent.includes('Odds')) { links[i].click(); return true; } }; return false;"
                            result = self.driver.execute_script(script)
                            if result:
                                self.logger.debug("Clicked odds tab using JavaScript")
                                time.sleep(1.5)
                                clicked = True
                        except Exception as e:
                            self.logger.debug(f"JavaScript click failed: {str(e)}")

                    if not clicked:
                        # Try direct navigation to odds URL
                        try:
                            odds_url = match_url.replace(
                                "#/match-summary",
                                "#/odds-comparison/1x2-odds/full-time",
                            )
                            self.driver.get(odds_url)
                            self.logger.debug(
                                f"Navigated directly to odds URL: {odds_url}"
                            )
                            time.sleep(1.5)
                        except Exception as e:
                            self.logger.warning(
                                f"Direct odds navigation failed: {str(e)}"
                            )
                            raise TimeoutException(
                                "Odds tab not found and direct navigation failed"
                            )

                except TimeoutException:
                    self.logger.warning(
                        "Odds tab not found or not clickable, odds might not be available"
                    )
                    return match_odds

            # Identify available betting markets
            if self.fast_mode:
                markets = ["1X2"]
                self.logger.info("Fast mode enabled, only scraping 1X2 market")
            else:
                try:
                    markets = self._identify_available_markets()

                    # Apply market limit if set
                    if self.market_limit is not None and self.market_limit > 0:
                        markets = markets[: self.market_limit]

                    self.logger.info(
                        f"Found {len(markets)} available betting markets",
                        extra={"markets": markets},
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to identify markets: {str(e)}")
                    markets = ["1X2"]  # Default to 1X2 if identification fails

            # Process each market sequentially
            for market_name in markets:
                try:
                    self.logger.debug(f"Scraping {market_name} market")

                    # Navigate to the market tab
                    self._navigate_to_market(market_name)
                    time.sleep(1.5)  # Extended wait for better reliability

                    # Determine the odds type from the market name
                    odds_type = self._determine_odds_type(market_name)

                    # Extract available betting periods (Full Time, 1st Half, etc.)
                    periods = self._extract_betting_periods()

                    # Simplified period handling - for most odds, Full Time is sufficient
                    # and more reliable than attempting to extract all periods
                    main_period = periods[0] if periods else "Full Time"

                    try:
                        self.logger.debug(
                            f"Extracting {main_period} odds for {market_name}"
                        )

                        # Extract odds for the main period
                        odds_list = self._extract_odds_for_market(
                            odds_type, main_period
                        )

                        # Add odds to match_odds container
                        for odds in odds_list:
                            match_odds.add_odds(odds)

                        self.logger.info(
                            f"Extracted {len(odds_list)} {main_period} odds entries for {market_name}",
                            extra={
                                "bookmakers": [odds.bookmaker for odds in odds_list[:3]]
                            },  # Just show first 3
                        )

                        # If we've successfully extracted main period odds and there are additional periods,
                        # attempt to extract those as well
                        if len(odds_list) > 0 and len(periods) > 1:
                            for period in periods[1:]:
                                try:
                                    self.logger.debug(
                                        f"Extracting {period} odds for {market_name}"
                                    )
                                    self._select_betting_period(period)
                                    time.sleep(1)

                                    period_odds_list = self._extract_odds_for_market(
                                        odds_type, period
                                    )

                                    # Add odds to match_odds container
                                    for odds in period_odds_list:
                                        match_odds.add_odds(odds)

                                    self.logger.info(
                                        f"Extracted {len(period_odds_list)} {period} odds entries for {market_name}"
                                    )
                                except Exception as e:
                                    self.logger.warning(
                                        f"Failed to extract {period} odds for {market_name}",
                                        extra={"error": str(e)},
                                    )
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to extract odds for {market_name}",
                            extra={
                                "error_type": type(e).__name__,
                                "error_details": str(e),
                            },
                        )

                except Exception as e:
                    self.logger.warning(
                        f"Failed to scrape {market_name} market",
                        extra={"error_type": type(e).__name__, "error_details": str(e)},
                    )
                    continue

            # Log summary of collected odds
            self._log_odds_summary(match_odds)
            return match_odds

        except Exception as e:
            self.logger.error(
                f"Failed to scrape match odds",
                extra={"error_type": type(e).__name__, "error_details": str(e)},
            )
            return match_odds

    def _extract_match_id(self, url: str) -> str:
        """Extract match ID from URL."""
        try:
            # Parse match ID from URL format like: https://www.soccer24.com/match/nZHLD80c/#/match-summary
            match_path = urlparse(url).path
            match_id = match_path.split("/match/")[1].split("/")[0]
            return match_id
        except Exception as e:
            self.logger.warning(
                f"Failed to extract match ID from URL", extra={"error": str(e)}
            )
            return url.split("/")[-1]

    def _identify_available_markets(self) -> List[str]:
        """
        Identify all available betting markets from the tabs.

        Returns:
            List of market names
        """
        markets = []
        try:
            # Find market tab elements - try multiple selectors
            market_selectors = [
                "a[title]:not([title=''])",
                ".filterOverTab",
                "a.wcl-tabs_jy59b",
                "div.wcl-tabs_jy59b a",
                "div[role='tablist'] a",
            ]

            for selector in market_selectors:
                market_tabs = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if market_tabs:
                    # Extract market names from tab texts
                    for tab in market_tabs:
                        try:
                            # Try to get the title attribute first
                            market_name = tab.get_attribute("title")

                            # If no title, try the text content
                            if not market_name:
                                market_name = tab.text.strip()

                            if market_name and market_name not in [
                                "MATCH",
                                "ODDS",
                                "H2H",
                                "DRAW",
                            ]:
                                # Check if this market is one we care about
                                if any(
                                    target.lower() in market_name.lower()
                                    for target in self.target_markets
                                ):
                                    if market_name not in markets:
                                        markets.append(market_name)
                        except:
                            continue

            # Try to find markets using the href attribute if we didn't find enough
            if len(markets) < len(self.target_markets):
                # Look for hrefs containing market names
                href_patterns = {
                    "1X2": "/odds-comparison/1x2",
                    "Over/Under": "/odds-comparison/over-under",
                    "Both teams to score": "/odds-comparison/both-teams-to-score",
                    "Correct score": "/odds-comparison/correct-score",
                    "Odd/Even": "/odds-comparison/odd-even",
                }

                for market_name, href_pattern in href_patterns.items():
                    if any(m.lower() == market_name.lower() for m in markets):
                        continue  # Already found this market

                    try:
                        elements = self.driver.find_elements(
                            By.CSS_SELECTOR, f"a[href*='{href_pattern}']"
                        )
                        if elements:
                            markets.append(market_name)
                    except:
                        pass

            # Always include 1X2 (match winner) if not already in the list
            if not any(m for m in markets if "1X2" in m or "1x2" in m.lower()):
                markets.insert(0, "1X2")

            # Ensure we have the exact market names we want
            cleaned_markets = []
            for target in self.target_markets:
                # Find the best match in our collected markets
                matched = False
                for m in markets:
                    if target.lower() in m.lower():
                        cleaned_markets.append(m)
                        matched = True
                        break

                # If no match found, add the target name directly
                if not matched and not self.fast_mode:
                    cleaned_markets.append(target)

            return cleaned_markets

        except Exception as e:
            self.logger.warning(
                f"Failed to identify available markets",
                extra={"error_type": type(e).__name__, "error_details": str(e)},
            )
            # Return at least 1X2 as default
            return ["1X2"]

    def _navigate_to_market(self, market_name: str) -> None:
        """
        Navigate to the specified betting market tab with enhanced reliability.

        Args:
            market_name: Name of the market to navigate to
        """
        try:
            # Direct URL navigation - most reliable method
            current_url = self.driver.current_url
            base_url = current_url.split("#")[0]

            # Map market names to URL fragments
            market_urls = {
                "1X2": f"{base_url}#/odds-comparison/1x2-odds/full-time",
                "Over/Under": f"{base_url}#/odds-comparison/over-under/full-time",
                "Both teams to score": f"{base_url}#/odds-comparison/both-teams-to-score/full-time",
                "Correct score": f"{base_url}#/odds-comparison/correct-score/full-time",
                "Odd/Even": f"{base_url}#/odds-comparison/odd-even/full-time",
            }

            # Find best match for the market name
            market_key = None
            for key in market_urls.keys():
                if (
                    key.lower() in market_name.lower()
                    or market_name.lower() in key.lower()
                ):
                    market_key = key
                    break

            if market_key:
                # Navigate directly to the market URL
                self.logger.debug(f"Navigating directly to {market_key} market URL")
                self.driver.get(market_urls[market_key])
                time.sleep(2)  # Wait for page to load
                return

            # Fallback to clicking methods if direct navigation fails

            # Try using the href attribute which is more reliable than text
            url_fragments = {
                "1X2": "/1x2-odds/",
                "Over/Under": "/over-under/",
                "Both teams to score": "/both-teams-to-score/",
                "Correct score": "/correct-score/",
                "Odd/Even": "/odd-even/",
            }

            for key, fragment in url_fragments.items():
                if (
                    key.lower() in market_name.lower()
                    or market_name.lower() in key.lower()
                ):
                    try:
                        selector = f"a[href*='{fragment}']"
                        elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        if elements:
                            elements[0].click()
                            self.logger.debug(
                                f"Clicked on {market_name} market tab using href: {fragment}"
                            )
                            time.sleep(1.5)
                            return
                    except Exception:
                        continue

            # If all else fails, log the failure
            self.logger.warning(f"Could not navigate to market: {market_name}")

        except Exception as e:
            self.logger.warning(
                f"Failed to navigate to market",
                extra={
                    "market": market_name,
                    "error_type": type(e).__name__,
                    "error_details": str(e),
                },
            )

    def _determine_odds_type(self, market_name: str) -> OddsType:
        """
        Determine the OddsType from the market name.

        Args:
            market_name: Name of the betting market

        Returns:
            Corresponding OddsType enum value
        """
        # Clean market name for better matching
        market_name_lower = market_name.lower()

        # Check for exact matches first
        for key, value in self.market_mapping.items():
            if key.lower() == market_name_lower:
                return value

        # If no exact match, check for partial matches
        if "1x2" in market_name_lower:
            return OddsType.MATCH_WINNER
        elif "over/under" in market_name_lower or "o/u" in market_name_lower:
            return OddsType.OVER_UNDER
        elif "both teams to score" in market_name_lower or "btts" in market_name_lower:
            return OddsType.BOTH_TEAMS_TO_SCORE
        elif "correct score" in market_name_lower:
            return OddsType.CORRECT_SCORE
        elif "odd/even" in market_name_lower or "odd even" in market_name_lower:
            return OddsType.ODD_EVEN

        # Default to match winner if no match found
        self.logger.warning(
            f"Could not determine odds type for market: {market_name}, defaulting to Match Winner"
        )
        return OddsType.MATCH_WINNER

    def _extract_betting_periods(self) -> List[str]:
        """
        Extract available betting periods with improved reliability.
        Simplified to focus on commonly found periods.

        Returns:
            List of period names
        """
        # Start with just Full Time as our default
        periods = ["Full Time"]

        try:
            # Check URL for period indicators
            url = self.driver.current_url
            if "/full-time" in url.lower():
                periods = ["Full Time"]
                return periods
            elif "/1st-half" in url.lower():
                periods = ["1st Half"]
                return periods
            elif "/2nd-half" in url.lower():
                periods = ["2nd Half"]
                return periods

            # Look for period tabs using direct DOM inspection
            period_elements = []

            # Try multiple selectors
            period_selectors = [
                "li[class*='period']",
                ".wcl-tab_y-fEC",
                ".event__tab",
                ".oddsTabOption",
                "div[role='tab']",
                "a[href*='full-time']",
                "a[href*='1st-half']",
                "a[href*='2nd-half']",
            ]

            for selector in period_selectors:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    period_elements = elements
                    self.logger.debug(
                        f"Found {len(elements)} period elements using selector: {selector}"
                    )
                    break

            # Process period elements if found
            if period_elements:
                # Clear default and build new list
                periods = []

                for element in period_elements:
                    period_text = element.text.strip()
                    if period_text:
                        # Convert to standard period names
                        if (
                            "full" in period_text.lower()
                            or "match" in period_text.lower()
                            or "ft" == period_text.lower()
                        ):
                            periods.append("Full Time")
                        elif (
                            "1st" in period_text.lower()
                            or "first" in period_text.lower()
                            or "1h" == period_text.lower()
                        ):
                            periods.append("1st Half")
                        elif (
                            "2nd" in period_text.lower()
                            or "second" in period_text.lower()
                            or "2h" == period_text.lower()
                        ):
                            periods.append("2nd Half")
                        else:
                            periods.append(period_text)

            # Filter to remove duplicates and ensure periods is non-empty
            periods = list(dict.fromkeys(periods))
            if not periods:
                periods = ["Full Time"]

            self.logger.debug(f"Extracted betting periods: {periods}")
            return periods

        except Exception as e:
            self.logger.warning(
                f"Failed to extract betting periods",
                extra={"error_type": type(e).__name__, "error_details": str(e)},
            )
            return ["Full Time"]  # Default to Full Time only

    def _select_betting_period(self, period: str) -> None:
        """
        Select a specific betting period tab.

        Args:
            period: Period to select ("Full Time", "1st Half", "2nd Half")
        """
        try:
            # Map periods to possible button texts
            period_mapping = {
                "Full Time": ["FULL TIME", "Match", "MATCH", "FT"],
                "1st Half": ["1ST HALF", "First Half", "FIRST HALF", "1H"],
                "2nd Half": ["2ND HALF", "Second Half", "SECOND HALF", "2H"],
            }

            possible_texts = period_mapping.get(period, [period])

            # Try to click the period tab using multiple approaches
            for text in possible_texts:
                # Try direct text match with XPath
                xpath = f"//*[text()='{text}' or contains(text(), '{text}')]"
                try:
                    elements = self.driver.find_elements(By.XPATH, xpath)
                    for element in elements:
                        try:
                            # Check if this element is visible and looks like a tab
                            if element.is_displayed():
                                element.click()
                                time.sleep(0.5)
                                self.logger.debug(
                                    f"Selected period {period} using text: {text}"
                                )
                                return
                        except:
                            continue
                except:
                    pass

                # Try with CSS selectors for common period tab classes
                selectors = [
                    f".wcl-tab_y-fEC:contains('{text}')",
                    f".event__tab:contains('{text}')",
                    f"div[role='tab']:contains('{text}')",
                ]

                for selector in selectors:
                    try:
                        elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        if elements and elements[0].is_displayed():
                            elements[0].click()
                            time.sleep(0.5)
                            self.logger.debug(
                                f"Selected period {period} using selector: {selector}"
                            )
                            return
                    except:
                        continue

            # Try JavaScript click as last resort
            for text in possible_texts:
                try:
                    script = f"var elements = document.querySelectorAll('*'); for (var i = 0; i < elements.length; i++) {{ if (elements[i].textContent.includes('{text}')) {{ elements[i].click(); return true; }} }}; return false;"
                    clicked = self.driver.execute_script(script)
                    if clicked:
                        time.sleep(0.5)
                        self.logger.debug(
                            f"Selected period {period} using JavaScript with text: {text}"
                        )
                        return
                except:
                    pass

            self.logger.warning(f"Could not find tab for period: {period}")

        except Exception as e:
            self.logger.warning(
                f"Failed to select betting period",
                extra={
                    "period": period,
                    "error_type": type(e).__name__,
                    "error_details": str(e),
                },
            )

    def _extract_odds_for_market(
        self, odds_type: OddsType, period: str
    ) -> Sequence[OddsVariant]:
        """
        Extract odds for a specific market and period.

        Args:
            odds_type: Type of odds to extract
            period: Betting period ("Full Time", "1st Half", "2nd Half")

        Returns:
            List of odds objects of the appropriate type
        """
        # Call the appropriate extraction method based on odds_type
        if odds_type == OddsType.MATCH_WINNER:
            return self._extract_match_winner_odds(period)
        elif odds_type == OddsType.OVER_UNDER:
            return self._extract_over_under_odds(period)
        elif odds_type == OddsType.BOTH_TEAMS_TO_SCORE:
            return self._extract_btts_odds(period)
        elif odds_type == OddsType.CORRECT_SCORE:
            return self._extract_correct_score_odds(period)
        elif odds_type == OddsType.ODD_EVEN:
            return self._extract_odd_even_odds(period)
        else:
            self.logger.warning(f"Unknown odds type: {odds_type}")
            return []

    def _extract_match_winner_odds(self, period: str) -> List[MatchWinnerOdds]:
        """
        Extract 1X2 (match winner) odds with improved DOM targeting.

        Args:
            period: Betting period

        Returns:
            List of MatchWinnerOdds objects
        """
        odds_list = []

        try:
            # First check if we're on the right page/tab
            url = self.driver.current_url
            if "/1x2-odds/" not in url and "/match-summary/" not in url:
                self.logger.debug(f"Not on 1X2 odds page. Current URL: {url}")
                return odds_list

            # Find all bookmaker rows using multiple potential selectors
            bookmaker_selectors = [
                "tr.oddsTab__tableRow",
                ".ui-table__row",
                "tr",
                ".oddsCell__container",
                ".ui-table_body tr",
            ]

            bookmaker_rows = []
            for selector in bookmaker_selectors:
                rows = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if rows:
                    bookmaker_rows = rows
                    self.logger.debug(
                        f"Found {len(rows)} bookmaker rows using selector: {selector}"
                    )
                    break

            # If we still don't have rows, try a broader approach
            if not bookmaker_rows:
                # Look for table body first
                table_bodies = self.driver.find_elements(
                    By.CSS_SELECTOR, "tbody, .ui-table__body"
                )
                if table_bodies:
                    for tbody in table_bodies:
                        rows = tbody.find_elements(By.TAG_NAME, "tr")
                        if rows:
                            bookmaker_rows = rows
                            self.logger.debug(
                                f"Found {len(rows)} bookmaker rows in table body"
                            )
                            break

            # Debug info about the rows found
            if bookmaker_rows:
                self.logger.debug(f"Processing {len(bookmaker_rows)} bookmaker rows")
            else:
                self.logger.warning("No bookmaker rows found")

                # Last resort: try to capture any visible odds on the page
                try:
                    # Look for odds cells directly
                    odds_cells = self.driver.find_elements(
                        By.CSS_SELECTOR,
                        ".oddsCell__odd, a[class*='odd'], td:not(.bookmaker)",
                    )

                    if odds_cells and len(odds_cells) >= 3:
                        # Group odds by bookmaker if possible
                        home_win = self._parse_odds_value(odds_cells[0].text)
                        draw = self._parse_odds_value(odds_cells[1].text)
                        away_win = self._parse_odds_value(odds_cells[2].text)

                        if home_win > 0 and draw > 0 and away_win > 0:
                            odds = MatchWinnerOdds(
                                bookmaker="Unknown Bookmaker",
                                timestamp=datetime.now(),
                                period=period,
                                market_type=OddsType.MATCH_WINNER,
                                home_win=home_win,
                                draw=draw,
                                away_win=away_win,
                            )
                            odds_list.append(odds)
                            self.logger.debug(
                                f"Created fallback odds entry: {home_win}/{draw}/{away_win}"
                            )
                except Exception as e:
                    self.logger.debug(f"Fallback odds extraction failed: {str(e)}")

                return odds_list

            # Process each row to extract bookmaker and odds
            for row in bookmaker_rows:
                try:
                    # Skip header rows and empty rows
                    row_text = row.text.strip()
                    if (
                        not row_text
                        or row_text.upper() == "BOOKMAKER"
                        or "CS" in row_text
                    ):
                        continue

                    # Extract bookmaker name using multiple techniques
                    bookmaker = self._extract_bookmaker_name(row)

                    # Skip if we couldn't find a valid bookmaker
                    if bookmaker in ["BOOKMAKER", "Unknown Bookmaker", ""]:
                        # Try to extract any logo alt text
                        try:
                            img = row.find_element(By.TAG_NAME, "img")
                            bookmaker = img.get_attribute("alt") or "Unknown Bookmaker"
                        except:
                            # Check if the row contains bet365, which is a common bookmaker
                            if "bet365" in row_text.lower():
                                bookmaker = "bet365"
                            elif "unibet" in row_text.lower():
                                bookmaker = "Unibet"
                            elif "william" in row_text.lower():
                                bookmaker = "William Hill"
                            else:
                                continue  # Skip this row if we can't identify the bookmaker

                    # Extract odds values (home, draw, away) - try multiple methods
                    home_win, draw, away_win = 0, 0, 0

                    # Method 1: Use odds_values selector
                    try:
                        odds_cells = row.find_elements(
                            By.CSS_SELECTOR, self.config.SELECTORS["odds_values"]
                        )
                        if len(odds_cells) >= 3:
                            home_win = self._parse_odds_value(odds_cells[0].text)
                            draw = self._parse_odds_value(odds_cells[1].text)
                            away_win = self._parse_odds_value(odds_cells[2].text)
                    except Exception:
                        pass

                    # Method 2: Use any td elements if method 1 fails
                    if home_win == 0 or draw == 0 or away_win == 0:
                        try:
                            cells = row.find_elements(By.TAG_NAME, "td")
                            # Skip the first cell (bookmaker) and get the next three
                            if len(cells) >= 4:
                                home_win = self._parse_odds_value(cells[1].text)
                                draw = self._parse_odds_value(cells[2].text)
                                away_win = self._parse_odds_value(cells[3].text)
                        except Exception:
                            pass

                    # Method 3: Use any elements with odds class
                    if home_win == 0 or draw == 0 or away_win == 0:
                        try:
                            odd_elements = row.find_elements(
                                By.CSS_SELECTOR,
                                "[class*='odd'], [class*='oddsCell'], .oddsCell__odd, a.oddsCell__odd",
                            )
                            if len(odd_elements) >= 3:
                                home_win = self._parse_odds_value(odd_elements[0].text)
                                draw = self._parse_odds_value(odd_elements[1].text)
                                away_win = self._parse_odds_value(odd_elements[2].text)
                        except Exception:
                            pass

                    # If we have valid odds, create a MatchWinnerOdds object
                    if home_win > 0 and draw > 0 and away_win > 0:
                        odds = MatchWinnerOdds(
                            bookmaker=bookmaker,
                            timestamp=datetime.now(),
                            period=period,
                            market_type=OddsType.MATCH_WINNER,
                            home_win=home_win,
                            draw=draw,
                            away_win=away_win,
                        )
                        odds_list.append(odds)
                        self.logger.debug(
                            f"Extracted 1X2 odds for {bookmaker}",
                            extra={"home": home_win, "draw": draw, "away": away_win},
                        )

                except Exception as e:
                    self.logger.debug(
                        f"Failed to extract match winner odds for a row",
                        extra={"error_type": type(e).__name__, "error_details": str(e)},
                    )
                    continue

            return odds_list

        except Exception as e:
            self.logger.warning(
                f"Failed to extract match winner odds",
                extra={"error_type": type(e).__name__, "error_details": str(e)},
            )
            return []

    def _extract_over_under_odds(self, period: str) -> List[OverUnderOdds]:
        """
        Extract Over/Under odds.

        Args:
            period: Betting period

        Returns:
            List of OverUnderOdds objects
        """
        odds_list = []

        try:
            # Find all bookmaker rows
            bookmaker_rows = self.driver.find_elements(
                By.CSS_SELECTOR, self.config.SELECTORS["bookmaker_rows"]
            )

            # First, determine the structure of the table
            # Sometimes each row has a total, other times all rows share a total
            table_structure = "unknown"
            shared_total = None

            # Try to find table headers that might contain total values
            header_elements = self.driver.find_elements(By.CSS_SELECTOR, "th, td.total")
            for header in header_elements:
                header_text = header.text.strip()
                # Look for a total value in headers
                if "TOTAL" in header_text.upper():
                    table_structure = "shared_totals"
                    break
                # Look for OVER/UNDER headers which might indicate totals in column headers
                elif "OVER" in header_text.upper() or "UNDER" in header_text.upper():
                    # Check if the header contains a number (the total)
                    match = re.search(r"(\d+\.?\d*)", header_text)
                    if match:
                        shared_total = float(match.group(1))
                        table_structure = "column_totals"
                        break

            # Process each row based on the identified structure
            for row in bookmaker_rows:
                try:
                    # Extract bookmaker name
                    bookmaker = self._extract_bookmaker_name(row)

                    # Skip header rows and rows without a valid bookmaker
                    if bookmaker in [
                        "BOOKMAKER",
                        "TOTAL",
                        "",
                        "Unknown Bookmaker",
                    ] and not row.get_attribute("class"):
                        continue

                    # Set default total in case we can't find one
                    total = shared_total or 2.5

                    # Extract total value based on table structure
                    if table_structure == "shared_totals":
                        # Look for total cells in this row
                        total_cells = row.find_elements(
                            By.CSS_SELECTOR, "td:first-child, td.total"
                        )
                        for cell in total_cells:
                            cell_text = cell.text.strip()
                            match = re.search(r"(\d+\.?\d*)", cell_text)
                            if match:
                                total = float(match.group(1))
                                break

                    # Extract odds values (over, under)
                    odds_cells = row.find_elements(
                        By.CSS_SELECTOR, self.config.SELECTORS["odds_values"]
                    )

                    # Sometimes over/under values are in different row structures
                    if len(odds_cells) >= 2:
                        # Determine which is over and which is under based on context
                        over_idx = 0
                        under_idx = 1

                        # If we have classes or headers that indicate which is which
                        over_candidates = row.find_elements(
                            By.CSS_SELECTOR, ".over, [class*='over']"
                        )
                        under_candidates = row.find_elements(
                            By.CSS_SELECTOR, ".under, [class*='under']"
                        )

                        if over_candidates and under_candidates:
                            # We have specific elements for over and under
                            over = self._parse_odds_value(over_candidates[0].text)
                            under = self._parse_odds_value(under_candidates[0].text)
                        else:
                            # Standard structure - just use the odds cells
                            over = self._parse_odds_value(odds_cells[over_idx].text)
                            under = self._parse_odds_value(odds_cells[under_idx].text)

                        # Skip rows where both values are 0 (invalid rows)
                        if over == 0 and under == 0:
                            continue

                        # Create OverUnderOdds object
                        odds = OverUnderOdds(
                            bookmaker=bookmaker,
                            timestamp=datetime.now(),
                            period=period,
                            market_type=OddsType.OVER_UNDER,
                            total=total,
                            over=over,
                            under=under,
                        )

                        odds_list.append(odds)

                except Exception as e:
                    self.logger.debug(
                        f"Failed to extract over/under odds for a row",
                        extra={"error_type": type(e).__name__, "error_details": str(e)},
                    )
                    continue

            return odds_list

        except Exception as e:
            self.logger.warning(
                f"Failed to extract over/under odds",
                extra={"error_type": type(e).__name__, "error_details": str(e)},
            )
            return []

    def _extract_btts_odds(self, period: str) -> List[BothTeamsToScoreOdds]:
        """
        Extract Both Teams To Score odds.

        Args:
            period: Betting period

        Returns:
            List of BothTeamsToScoreOdds objects
        """
        odds_list = []

        try:
            # Find all bookmaker rows
            bookmaker_rows = self.driver.find_elements(
                By.CSS_SELECTOR, self.config.SELECTORS["bookmaker_rows"]
            )

            for row in bookmaker_rows:
                try:
                    # Extract bookmaker name
                    bookmaker = self._extract_bookmaker_name(row)

                    # Skip header rows and rows without a valid bookmaker
                    if bookmaker in [
                        "BOOKMAKER",
                        "",
                        "Unknown Bookmaker",
                    ] and not row.get_attribute("class"):
                        continue

                    # Extract odds values (yes, no)
                    odds_cells = row.find_elements(
                        By.CSS_SELECTOR, self.config.SELECTORS["odds_values"]
                    )

                    if len(odds_cells) >= 2:
                        yes = self._parse_odds_value(odds_cells[0].text)
                        no = self._parse_odds_value(odds_cells[1].text)

                        # Skip rows where both values are 0 (invalid rows)
                        if yes == 0 and no == 0:
                            continue

                        # Create BothTeamsToScoreOdds object
                        odds = BothTeamsToScoreOdds(
                            bookmaker=bookmaker,
                            timestamp=datetime.now(),
                            period=period,
                            market_type=OddsType.BOTH_TEAMS_TO_SCORE,
                            yes=yes,
                            no=no,
                        )

                        odds_list.append(odds)

                except Exception as e:
                    self.logger.debug(
                        f"Failed to extract BTTS odds for a row",
                        extra={"error_type": type(e).__name__, "error_details": str(e)},
                    )
                    continue

            return odds_list

        except Exception as e:
            self.logger.warning(
                f"Failed to extract BTTS odds",
                extra={"error_type": type(e).__name__, "error_details": str(e)},
            )
            return []

    def _extract_correct_score_odds(self, period: str) -> List[CorrectScoreOdds]:
        """
        Extract Correct Score odds.

        Args:
            period: Betting period

        Returns:
            List of CorrectScoreOdds objects
        """
        odds_list = []

        try:
            # This is a complex table structure - typically showing scores and odds

            # Find column headers first to identify score mapping
            headers = self.driver.find_elements(By.CSS_SELECTOR, "th")
            score_map = {}

            # Try to extract scores from headers
            for idx, header in enumerate(headers):
                header_text = header.text.strip()
                # Scores often look like "1-0" or similar
                if re.match(r"\d+-\d+", header_text):
                    score_map[idx] = header_text

            # If we couldn't find scores in headers, check for CS column
            if not score_map:
                cs_headers = self.driver.find_elements(
                    By.CSS_SELECTOR, "th:contains('CS'), td:contains('CS')"
                )
                if cs_headers:
                    # We likely have a layout with scores in the first column
                    score_map = {"column_1": True}

            # Find all bookmaker rows
            bookmaker_rows = self.driver.find_elements(
                By.CSS_SELECTOR, self.config.SELECTORS["bookmaker_rows"]
            )

            for row in bookmaker_rows:
                try:
                    # Extract bookmaker name
                    bookmaker = self._extract_bookmaker_name(row)

                    # Skip header rows and rows without a valid bookmaker
                    if bookmaker in [
                        "BOOKMAKER",
                        "CS",
                        "",
                        "Unknown Bookmaker",
                    ] and not row.get_attribute("class"):
                        continue

                    # Get all cells in the row
                    cells = row.find_elements(By.CSS_SELECTOR, "td")

                    # Handle different table structures
                    if score_map and "column_1" in score_map:
                        # Score is in first column, odds in second
                        if len(cells) >= 2:
                            score = cells[0].text.strip()
                            odds_value = self._parse_odds_value(cells[1].text)

                            # Make sure we have a valid score and odds
                            if re.match(r"\d+-\d+", score) and odds_value > 0:
                                odds = CorrectScoreOdds(
                                    bookmaker=bookmaker,
                                    timestamp=datetime.now(),
                                    period=period,
                                    market_type=OddsType.CORRECT_SCORE,
                                    score=score,
                                    odds=odds_value,
                                )
                                odds_list.append(odds)
                    else:
                        # Each score has its own column, with headers defining the scores
                        odds_cells = row.find_elements(
                            By.CSS_SELECTOR, self.config.SELECTORS["odds_values"]
                        )

                        # Match odds cells with scores from the header mapping
                        for idx, odds_cell in enumerate(odds_cells):
                            if idx in score_map:
                                score = score_map[idx]
                                odds_value = self._parse_odds_value(odds_cell.text)

                                # Only add if we have valid odds
                                if odds_value > 0:
                                    odds = CorrectScoreOdds(
                                        bookmaker=bookmaker,
                                        timestamp=datetime.now(),
                                        period=period,
                                        market_type=OddsType.CORRECT_SCORE,
                                        score=score,
                                        odds=odds_value,
                                    )
                                    odds_list.append(odds)

                except Exception as e:
                    self.logger.debug(
                        f"Failed to extract correct score odds for a row",
                        extra={"error_type": type(e).__name__, "error_details": str(e)},
                    )
                    continue

            # If we couldn't extract anything through the main methods, try an alternative approach
            if not odds_list:
                # Find rows that have a CS column
                cs_rows = self.driver.find_elements(
                    By.XPATH, "//tr[.//td[contains(text(), '-')]]"
                )

                for row in cs_rows:
                    try:
                        # Extract bookmaker name
                        bookmaker = self._extract_bookmaker_name(row)

                        # Look for a score pattern in the row
                        row_text = row.text
                        # Find all score patterns (e.g., "1-0", "2-1")
                        score_matches = re.findall(r"(\d+-\d+)", row_text)

                        # Find all odds values
                        odds_cells = row.find_elements(
                            By.CSS_SELECTOR, self.config.SELECTORS["odds_values"]
                        )

                        if score_matches and odds_cells:
                            score = score_matches[0]  # Use the first score found
                            odds_value = self._parse_odds_value(odds_cells[0].text)

                            if odds_value > 0:
                                odds = CorrectScoreOdds(
                                    bookmaker=bookmaker,
                                    timestamp=datetime.now(),
                                    period=period,
                                    market_type=OddsType.CORRECT_SCORE,
                                    score=score,
                                    odds=odds_value,
                                )
                                odds_list.append(odds)
                    except Exception:
                        continue

            return odds_list

        except Exception as e:
            self.logger.warning(
                f"Failed to extract correct score odds",
                extra={"error_type": type(e).__name__, "error_details": str(e)},
            )
            return []

    def _extract_odd_even_odds(self, period: str) -> List[OddEvenOdds]:
        """
        Extract Odd/Even odds.

        Args:
            period: Betting period

        Returns:
            List of OddEvenOdds objects
        """
        odds_list = []

        try:
            # Find all bookmaker rows
            bookmaker_rows = self.driver.find_elements(
                By.CSS_SELECTOR, self.config.SELECTORS["bookmaker_rows"]
            )

            for row in bookmaker_rows:
                try:
                    # Extract bookmaker name
                    bookmaker = self._extract_bookmaker_name(row)

                    # Skip header rows and rows without a valid bookmaker
                    if bookmaker in [
                        "BOOKMAKER",
                        "",
                        "Unknown Bookmaker",
                    ] and not row.get_attribute("class"):
                        continue

                    # Extract odds values
                    odds_cells = row.find_elements(
                        By.CSS_SELECTOR, self.config.SELECTORS["odds_values"]
                    )

                    if len(odds_cells) >= 2:
                        odd = self._parse_odds_value(odds_cells[0].text)
                        even = self._parse_odds_value(odds_cells[1].text)

                        # Skip rows where both values are 0 (invalid rows)
                        if odd == 0 and even == 0:
                            continue

                        # Create OddEvenOdds object
                        odds = OddEvenOdds(
                            bookmaker=bookmaker,
                            timestamp=datetime.now(),
                            period=period,
                            market_type=OddsType.ODD_EVEN,
                            odd=odd,
                            even=even,
                        )

                        odds_list.append(odds)

                except Exception as e:
                    self.logger.debug(
                        f"Failed to extract odd/even odds for a row",
                        extra={"error_type": type(e).__name__, "error_details": str(e)},
                    )
                    continue

            return odds_list

        except Exception as e:
            self.logger.warning(
                f"Failed to extract odd/even odds",
                extra={"error_type": type(e).__name__, "error_details": str(e)},
            )
            return []

    def _extract_bookmaker_name(self, row: Any) -> str:
        """
        Extract bookmaker name from a row.

        Args:
            row: WebElement representing a bookmaker row

        Returns:
            Bookmaker name string
        """
        try:
            # Try different selectors for bookmaker name
            for selector in [
                self.config.SELECTORS["bookmaker_name"],
                ".oddsCell__bookmakerPart",
                ".bookmaker",
                "td:first-child",
                ".bookmaker img",
                "img[alt]",
            ]:
                try:
                    element = row.find_element(By.CSS_SELECTOR, selector)
                    bookmaker = element.text.strip()

                    # If text is empty, try to get from image alt
                    if not bookmaker:
                        try:
                            img = element.find_element(By.TAG_NAME, "img")
                            bookmaker = img.get_attribute("alt")
                        except:
                            pass

                    if bookmaker:
                        return bookmaker
                except:
                    continue

            # Fallback: try to get first cell if it's not numeric
            try:
                first_cell = row.find_element(By.CSS_SELECTOR, "td:first-child")
                text = first_cell.text.strip()
                # Check if it's not numeric (to avoid mistaking odds for bookmaker name)
                if text and not re.match(r"^\d+\.?\d*$", text):
                    return text
            except:
                pass

            return "Unknown Bookmaker"

        except Exception as e:
            self.logger.debug(
                f"Failed to extract bookmaker name", extra={"error": str(e)}
            )
            return "Unknown Bookmaker"

    def _parse_odds_value(self, odds_text: str) -> float:
        """
        Parse a string odds value into a float.

        Args:
            odds_text: String representation of odds

        Returns:
            Float value of odds
        """
        try:
            # Remove any non-numeric characters except decimal point
            odds_text = odds_text.strip()

            # Skip empty or no odds indicators
            if not odds_text or odds_text.upper() in ["N/A", "-", ""]:
                return 0.0

            # Handle European decimal format (e.g., "2,10")
            odds_text = odds_text.replace(",", ".")

            # Extract the numeric value using regex
            match = re.search(r"(\d+\.?\d*)", odds_text)
            if match:
                return float(match.group(1))

            # If no match is found, try direct conversion
            return float(odds_text)

        except (ValueError, AttributeError) as e:
            self.logger.debug(
                f"Failed to parse odds value: {odds_text}", extra={"error": str(e)}
            )
            return 0.0

    def _log_odds_summary(self, match_odds: MatchOdds) -> None:
        """
        Log summary of collected odds data.

        Args:
            match_odds: MatchOdds object with collected data
        """
        summary = {
            "match_id": match_odds.match_id,
            "match_winner_odds_count": len(match_odds.match_winner_odds),
            "over_under_odds_count": len(match_odds.over_under_odds),
            "btts_odds_count": len(match_odds.btts_odds),
            "correct_score_odds_count": len(match_odds.correct_score_odds),
            "odd_even_odds_count": len(match_odds.odd_even_odds),
        }

        # Calculate total number of odds entries
        total_odds = sum(
            count for market, count in summary.items() if market != "match_id"
        )
        summary["total_odds_entries"] = total_odds

        self.logger.info(f"Odds scraping completed", extra=summary)
