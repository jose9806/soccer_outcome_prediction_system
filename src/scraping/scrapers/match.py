import time
import re
from typing import Tuple, Dict, Any, Optional
from datetime import datetime
from urllib.parse import urlparse

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
)

from src.config.logging_config import get_logger
from src.scraping.models.soccer_extraction import Match, MatchOdds
from src.scraping.scrapers.base import BaseScraper
from src.scraping.scrapers.stats import StatsScraper
from src.scraping.scrapers.odds import OddsScraper


class MatchScraper(BaseScraper):
    """Scraper for collecting match-level data."""

    def __init__(self, driver, config, **kwargs):
        super().__init__(driver, config)

        self.logger = get_logger(
            name="MatchScraper",
            color="green",
            level=(self.config.LOG_LEVEL if hasattr(self.config, "LOG_LEVEL") else 20),
            enable_file=True,
            file_path="src/logs/match_scraper.log",
        )
        self.logger.info("MatchScraper initialized")

        # Pass kwargs to child scrapers as needed
        self.stats_scraper = StatsScraper(driver, config, **kwargs)
        self.odds_scraper = OddsScraper(driver, config, **kwargs)

        self.scrape_stats = kwargs.get("scrape_stats", True)
        self.scrape_odds = kwargs.get("scrape_odds", True)

        if kwargs.get("fast_odds", False):
            if hasattr(self.odds_scraper, "set_fast_mode"):
                self.odds_scraper.set_fast_mode(True)
                self.logger.debug("Odds scraper set to fast mode")

    def scrape(self, match_url: str) -> Match:
        """
        Scrape complete match information including basic details, stats, and odds.

        Args:
            match_url: URL of the match to scrape

        Returns:
            Match object with all available data
        """
        self.logger.add_context(url=match_url)
        self.logger.info(f"Starting match scraping process")

        try:
            # Navigate to the match page
            self.logger.debug(f"Navigating to match URL")
            self.driver.get(match_url)
            time.sleep(self.config.REQUEST_DELAY)

            # Accept cookies if needed
            self._accept_cookies()

            # Extract match ID from URL
            match_id = self._extract_match_id(match_url)
            self.logger.debug(f"Extracted match ID", extra={"match_id": match_id})

            # Extract basic match information
            self.logger.debug(f"Extracting basic match information")
            date = self._extract_date()
            competition, season = self._extract_competition_and_season()
            home_team, away_team = self._extract_teams()
            home_score, away_score = self._extract_score()
            stage = self._extract_stage()
            status = self._extract_status()
            venue, referee, attendance, weather = self._extract_additional_metadata()

            self.logger.info(
                f"Extracted match details",
                extra={
                    "date": date.strftime("%Y-%m-%d %H:%M"),
                    "teams": f"{home_team} vs {away_team}",
                    "score": f"{home_score}-{away_score}",
                    "competition": competition,
                    "season": season,
                    "stage": stage,
                    "status": status,
                },
            )

            # Extract stats for different periods
            stats_dict = {}
            if self.scrape_stats:
                self.logger.debug(f"Beginning stats extraction")
                stats_dict = self._extract_all_stats()
            else:
                self.logger.debug("Stats extraction skipped")

            # Extract comprehensive odds data
            match_odds = MatchOdds(match_id=match_id)
            if self.scrape_odds:
                self.logger.debug(f"Beginning odds extraction")
                match_odds = self._extract_odds(match_url)
            else:
                self.logger.debug("Odds extraction skipped")

            # Create and return the Match object
            match = Match(
                match_id=match_id,
                competition=competition,
                season=season,
                date=date,
                home_team=home_team,
                away_team=away_team,
                home_score=home_score,
                away_score=away_score,
                stage=stage,
                status=status,
                venue=venue,
                referee=referee,
                attendance=attendance,
                weather=weather,
                first_half_stats=stats_dict.get("first_half"),
                second_half_stats=stats_dict.get("second_half"),
                full_time_stats=stats_dict.get("full_time"),
                odds=match_odds,
                # For backward compatibility, extract 1X2 odds if any
                odds_history=(
                    [o for o in match_odds.match_winner_odds]
                    if match_odds.match_winner_odds
                    else []
                ),
            )

            self.logger.info(f"Match scraping completed successfully")
            return match

        except Exception as e:
            self.logger.error(
                f"Failed to scrape match",
                extra={"error_type": type(e).__name__, "error_details": str(e)},
            )
            raise

    def _accept_cookies(self):
        """Handle cookie consent if present."""
        try:
            self.logger.debug("Looking for cookie consent dialog")
            cookie_selector = "button[aria-label='Accept cookies'], button.css-47sehv, button#onetrust-accept-btn-handler"
            cookie_button = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, cookie_selector))
            )
            cookie_button.click()
            self.logger.debug("Cookie consent accepted")
        except TimeoutException:
            self.logger.debug("No cookie consent popup found or already accepted")
            pass

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

    def _extract_date(self) -> datetime:
        """Extract and parse match date."""
        try:
            date_selector = (
                ".duelParticipant__startTime, div.duelParticipant__startTime"
            )
            date_element = self.wait_for_element(date_selector)
            date_str = date_element.text.strip()

            # Format is typically: DD.MM.YYYY HH:MM
            self.logger.debug(f"Extracted date string", extra={"date_str": date_str})

            try:
                return datetime.strptime(date_str, "%d.%m.%Y %H:%M")
            except ValueError:
                # Try alternative formats if standard format fails
                formats = ["%d.%m.%Y", "%Y-%m-%d %H:%M", "%Y-%m-%d"]
                for fmt in formats:
                    try:
                        return datetime.strptime(date_str, fmt)
                    except ValueError:
                        continue

                # If all formats fail, log and return current date
                self.logger.warning(
                    f"Failed to parse date string", extra={"date_str": date_str}
                )
                return datetime.now()

        except Exception as e:
            self.logger.error(f"Failed to extract match date", extra={"error": str(e)})
            return datetime.now()

    def _extract_competition_and_season(self) -> Tuple[str, str]:
        """Extract competition and season information."""
        try:
            # Look for the competition information which is often in a breadcrumb or header
            competition_selector = ".tournamentHeader__country, .event__title--type, div.tournamentHeaderDescription"
            comp_element = self.wait_for_element(competition_selector, timeout=5)
            comp_text = comp_element.text.strip()

            # Parse competition and season from the text
            # Example format: "COLOMBIA: LIGA AGUILA - CLAUSURA - PLAY OFFS - FINAL"
            parts = comp_text.split(" - ")
            competition = parts[0] if parts else "Unknown"

            # Try to extract season from URL or other elements
            # Fallback to current year if not found
            try:
                season_element = self.driver.find_element(
                    By.CSS_SELECTOR, ".tournamentHeader__season"
                )
                season = season_element.text.strip()
            except NoSuchElementException:
                # Extract from URL or fallback to match date year
                url = self.driver.current_url
                if "primera-a-" in url:
                    season = re.search(r"primera-a-(\d{4})", url).group(1)  # type: ignore
                else:
                    date_element = self.wait_for_element(
                        ".duelParticipant__startTime", timeout=5
                    )
                    date_str = date_element.text.strip()
                    match = re.search(r"\d{2}\.\d{2}\.(\d{4})", date_str)
                    season = match.group(1) if match else str(datetime.now().year)

            self.logger.debug(
                f"Extracted competition and season",
                extra={"competition": competition, "season": season},
            )
            return competition, season

        except Exception as e:
            self.logger.warning(
                f"Failed to extract competition and season", extra={"error": str(e)}
            )
            return "Unknown", str(datetime.now().year)

    def _extract_teams(self) -> Tuple[str, str]:
        """Extract home and away team names with enhanced fallback methods."""
        try:
            # Primary approach - standard participant selectors
            team_selector = ".participant__participantName, .duelParticipant__home .participant__participantName, .duelParticipant__away .participant__participantName"
            teams = self.driver.find_elements(By.CSS_SELECTOR, team_selector)

            if len(teams) >= 2:
                home_team = teams[0].text.strip()
                away_team = teams[1].text.strip()

                # Validate names aren't identical
                if home_team and away_team and home_team != away_team:
                    self.logger.debug(
                        f"Extracted team names",
                        extra={"home_team": home_team, "away_team": away_team},
                    )
                    return home_team, away_team

            # Fallback approach - team links in DOM
            home_links = self.driver.find_elements(
                By.XPATH,
                "//div[contains(@class, 'duelParticipant__home')]//a[contains(@href, '/team/')]",
            )
            away_links = self.driver.find_elements(
                By.XPATH,
                "//div[contains(@class, 'duelParticipant__away')]//a[contains(@href, '/team/')]",
            )

            if home_links and away_links:
                # Extract team names from link text or href
                home_team = home_links[0].text.strip()
                away_team = away_links[0].text.strip()

                # If text is empty, extract from href
                if not home_team or not away_team:
                    home_href = home_links[0].get_attribute("href")
                    away_href = away_links[0].get_attribute("href")

                    if home_href and away_href:
                        home_team = (
                            home_href.split("/team/")[1]
                            .split("/")[0]
                            .replace("-", " ")
                            .title()
                        )
                        away_team = (
                            away_href.split("/team/")[1]
                            .split("/")[0]
                            .replace("-", " ")
                            .title()
                        )

                if home_team and away_team and home_team != away_team:
                    self.logger.debug(
                        f"Extracted team names from links",
                        extra={"home_team": home_team, "away_team": away_team},
                    )
                    return home_team, away_team

            # If we reach here, extraction failed - raise exception
            raise ValueError("Could not extract distinct team names")

        except Exception as e:
            self.logger.error(f"Failed to extract team names", extra={"error": str(e)})
            return "Unknown Home", "Unknown Away"

    def _extract_score(self) -> Tuple[int, int]:
        """Extract match score."""
        try:
            # Look for the score element
            score_selector = (
                ".detailScore__wrapper, .event__score--home, .event__score--away"
            )
            score_element = self.wait_for_element(score_selector)
            score_text = score_element.text.strip()

            # Parse the score text (format is typically "X-Y")
            self.logger.debug(f"Extracted score text", extra={"score_text": score_text})

            # Handle different possible formats
            if "-" in score_text:
                parts = score_text.split("-")
                home_score = int(parts[0].strip())
                away_score = int(parts[1].strip())
            else:
                # Try to find individual score elements
                home_score_elem = self.driver.find_element(
                    By.CSS_SELECTOR, ".event__score--home"
                )
                away_score_elem = self.driver.find_element(
                    By.CSS_SELECTOR, ".event__score--away"
                )
                home_score = int(home_score_elem.text.strip())
                away_score = int(away_score_elem.text.strip())

            self.logger.debug(
                f"Parsed score values",
                extra={"home_score": home_score, "away_score": away_score},
            )
            return home_score, away_score

        except Exception as e:
            self.logger.error(f"Failed to extract match score", extra={"error": str(e)})
            return 0, 0

    def _extract_status(self) -> str:
        """Extract match status (e.g., Finished, Postponed, etc.)."""
        try:
            # Look for status indicator
            status_selectors = [
                ".detailScore__status",
                ".event__stage--block",
                ".eventStatus",
            ]

            for selector in status_selectors:
                try:
                    status_element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    status = status_element.text.strip()
                    if status:
                        self.logger.debug(
                            f"Extracted match status", extra={"status": status}
                        )
                        return status
                except NoSuchElementException:
                    continue

            # If no explicit status found, check for keywords in the page
            page_text = self.driver.find_element(By.TAG_NAME, "body").text
            if "FINISHED" in page_text or "FT" in page_text:
                return "Finished"
            elif "POSTPONED" in page_text:
                return "Postponed"
            elif "CANCELLED" in page_text:
                return "Cancelled"

            # Default status
            return "Finished"

        except Exception as e:
            self.logger.warning(
                f"Failed to extract match status", extra={"error": str(e)}
            )
            return "Unknown"

    def _extract_additional_metadata(
        self,
    ) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[str]]:
        """Extract additional match metadata like venue, referee, attendance, and weather."""
        venue = None
        referee = None
        attendance = None
        weather = None

        try:
            # Look for metadata section that often contains this information
            metadata_selectors = [".matchInfoData", ".matchInfo__item", ".match-info"]

            for selector in metadata_selectors:
                metadata_elements = self.driver.find_elements(By.CSS_SELECTOR, selector)

                for element in metadata_elements:
                    text = element.text.strip().lower()

                    # Extract venue
                    if "venue" in text or "stadium" in text:
                        venue_match = re.search(
                            r"venue:?\s*(.+)|stadium:?\s*(.+)", text, re.IGNORECASE
                        )
                        if venue_match:
                            venue = venue_match.group(1) or venue_match.group(2)

                    # Extract referee
                    if "referee" in text:
                        ref_match = re.search(r"referee:?\s*(.+)", text, re.IGNORECASE)
                        if ref_match:
                            referee = ref_match.group(1)

                    # Extract attendance
                    if "attendance" in text:
                        att_match = re.search(
                            r"attendance:?\s*([\d,\.]+)", text, re.IGNORECASE
                        )
                        if att_match:
                            # Convert to integer, removing commas and periods
                            attendance_str = (
                                att_match.group(1).replace(",", "").replace(".", "")
                            )
                            try:
                                attendance = int(attendance_str)
                            except ValueError:
                                pass

                    # Extract weather
                    if "weather" in text:
                        weather_match = re.search(
                            r"weather:?\s*(.+)", text, re.IGNORECASE
                        )
                        if weather_match:
                            weather = weather_match.group(1)

            # Log what we found
            if any([venue, referee, attendance, weather]):
                self.logger.debug(
                    f"Extracted additional metadata",
                    extra={
                        "venue": venue,
                        "referee": referee,
                        "attendance": attendance,
                        "weather": weather,
                    },
                )

            return venue, referee, attendance, weather

        except Exception as e:
            self.logger.debug(
                f"Failed to extract additional metadata", extra={"error": str(e)}
            )
            return None, None, None, None

    def _extract_stage(self) -> str:
        """Extract match stage (e.g., Group Stage, Final, etc.)."""
        try:
            # Look for stage information which might be in the competition or other elements
            stage_selectors = [
                ".tournamentHeader__country",
                ".event__title--type",
                ".tournament__info",
                ".event__title--name",
            ]

            for selector in stage_selectors:
                try:
                    element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    text = element.text.strip()

                    # Common stage keywords to look for
                    stage_keywords = [
                        "final",
                        "semi-final",
                        "quarter-final",
                        "round of 16",
                        "group",
                        "qualification",
                        "play-offs",
                        "play offs",
                        "regular season",
                    ]

                    for keyword in stage_keywords:
                        if keyword.lower() in text.lower():
                            self.logger.debug(
                                f"Extracted stage", extra={"stage": keyword.title()}
                            )
                            return keyword.title()

                except NoSuchElementException:
                    continue

            # If no specific stage found, try to extract from breadcrumbs or title
            try:
                breadcrumb = self.driver.find_element(
                    By.CSS_SELECTOR, ".breadcrumb__link, .event__title--name"
                )
                text = breadcrumb.text.strip()

                if " - " in text:
                    parts = text.split(" - ")
                    if len(parts) > 1:
                        stage = parts[-1].strip()
                        self.logger.debug(
                            f"Extracted stage from breadcrumb", extra={"stage": stage}
                        )
                        return stage
            except NoSuchElementException:
                pass

            # Default fallback
            self.logger.debug(
                f"No specific stage found, defaulting to 'Regular Season'"
            )
            return "Regular Season"

        except Exception as e:
            self.logger.warning(
                f"Failed to extract match stage", extra={"error": str(e)}
            )
            return "Unknown"

    def _extract_all_stats(self) -> Dict[str, Any]:
        """Enhanced method to extract statistics for all periods."""
        stats_dict = {}

        try:
            # Extract full time stats first (default view)
            self.logger.debug("Extracting full time stats")
            current_url = self.driver.current_url

            # Ensure we're on the statistics page
            if "/match-statistics" not in current_url:
                base_url = (
                    current_url.split("#")[0] if "#" in current_url else current_url
                )
                stats_url = f"{base_url}#/match-summary/match-statistics"
                self.logger.debug(f"Navigating to stats URL: {stats_url}")
                self.driver.get(stats_url)
                time.sleep(3)  # Allow page to load

            # Extract full time stats
            full_time_stats = self.stats_scraper.scrape(
                self.driver.current_url, extract_only=True
            )
            stats_dict["full_time"] = full_time_stats

            # Log what stats were found
            stats_attrs = [
                attr
                for attr in dir(full_time_stats)
                if not attr.startswith("_")
                and attr != "to_dict"
                and getattr(full_time_stats, attr) is not None
            ]
            self.logger.info(f"Extracted {len(stats_attrs)} full time statistics")

            # Try to extract first half and second half stats
            try:
                # Based on the UI, specifically target the half-time period tabs
                first_half_selector = (
                    "a[href*='/match-statistics/1'], a[title='1st Half']"
                )
                second_half_selector = (
                    "a[href*='/match-statistics/2'], a[title='2nd Half']"
                )

                # Extract first half stats
                first_half_elements = self.driver.find_elements(
                    By.CSS_SELECTOR, first_half_selector
                )
                if first_half_elements:
                    self.logger.debug("Navigating to first half statistics")
                    first_half_url = (
                        self.driver.current_url.split("#")[0]
                        + "#/match-summary/match-statistics/1"
                    )
                    self.driver.get(first_half_url)
                    time.sleep(2)  # Allow page to load

                    first_half_stats = self.stats_scraper.scrape(
                        self.driver.current_url, extract_only=True
                    )
                    stats_dict["first_half"] = first_half_stats
                    self.logger.debug("Extracted first half statistics")
                else:
                    self.logger.warning("First half tab not found")

                # Extract second half stats
                second_half_elements = self.driver.find_elements(
                    By.CSS_SELECTOR, second_half_selector
                )
                if second_half_elements:
                    self.logger.debug("Navigating to second half statistics")
                    second_half_url = (
                        self.driver.current_url.split("#")[0]
                        + "#/match-summary/match-statistics/2"
                    )
                    self.driver.get(second_half_url)
                    time.sleep(2)  # Allow page to load

                    second_half_stats = self.stats_scraper.scrape(
                        self.driver.current_url, extract_only=True
                    )
                    stats_dict["second_half"] = second_half_stats
                    self.logger.debug("Extracted second half statistics")
                else:
                    self.logger.warning("Second half tab not found")

            except Exception as e:
                self.logger.warning(f"Failed to extract half-time stats: {str(e)}")
                # Continue with whatever stats we have

            self.logger.info(
                "Stats extraction completed", extra={"periods_found": len(stats_dict)}
            )
            return stats_dict

        except Exception as e:
            self.logger.error(
                f"Failed to extract stats",
                extra={"error_type": type(e).__name__, "error_details": str(e)},
            )
            # Return whatever stats we have so far
            return stats_dict

    def _extract_odds(self, match_url: str):
        """
        Enhanced method to extract odds with better error handling.

        Args:
            match_url: URL of the match

        Returns:
            Match odds object
        """
        try:
            self.logger.debug("Beginning odds extraction")

            # Try direct navigation to odds page first
            odds_url = match_url.replace(
                "#/match-summary", "#/odds-comparison/1x2-odds/full-time"
            )
            odds_url = odds_url.split("#")[0] + "#/odds-comparison/1x2-odds/full-time"

            # Save current URL to restore later if needed
            current_url = self.driver.current_url

            try:
                self.logger.debug(f"Navigating directly to odds URL: {odds_url}")
                self.driver.get(odds_url)
                time.sleep(2)  # Allow page to load
            except Exception as e:
                self.logger.warning(f"Failed to navigate to odds URL: {str(e)}")
                # Restore original URL
                self.driver.get(current_url)
                time.sleep(1)

            # Extract odds using the enhanced odds scraper
            match_odds = self.odds_scraper.scrape(match_url)

            # Log comprehensive summary
            odds_summary = {
                "match_winner_odds": len(match_odds.match_winner_odds),
                "over_under_odds": len(match_odds.over_under_odds),
                "btts_odds": len(match_odds.btts_odds),
                "correct_score_odds": len(match_odds.correct_score_odds),
                "odd_even_odds": len(match_odds.odd_even_odds),
            }

            total_odds = sum(odds_summary.values())
            self.logger.info(
                f"Extracted odds data: {total_odds} total entries", extra=odds_summary
            )

            return match_odds

        except Exception as e:
            self.logger.error(
                f"Failed to extract odds",
                extra={"error_type": type(e).__name__, "error_details": str(e)},
            )
            # Return empty odds object rather than None
            match_id = self._extract_match_id(match_url)
            return MatchOdds(match_id=match_id)
