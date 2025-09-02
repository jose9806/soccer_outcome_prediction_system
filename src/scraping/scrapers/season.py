from typing import List, Dict
import time
import json
from pathlib import Path
from datetime import datetime

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    StaleElementReferenceException,
)

from src.config.logging_config import get_logger
from src.scraping.scrapers.base import BaseScraper
from src.scraping.exceptions import ScrapingError
from src.scraping.models.soccer_extraction import Season
from src.scraping.utils.validation import create_directory_if_not_exists


class SeasonScraper(BaseScraper):
    """Scraper for collecting season-level data including all match URLs."""

    def __init__(self, driver, config, **kwargs):
        super().__init__(driver, config)
        # Initialize the custom logger with appropriate configuration
        self.logger = get_logger(
            name="SeasonScraper",
            color="cyan",
            level=(
                self.config.LOG_LEVEL if hasattr(self.config, "LOG_LEVEL") else 20
            ),  # INFO=20
            enable_file=True,
            file_path="src/logs/season_scraper.log",
        )
        self.logger.info("SeasonScraper initialized")

    def scrape(self, year: int) -> Season:
        """
        Scrape all matches for a given season year.

        Args:
            year: The year to scrape

        Returns:
            Season object containing all match URLs
        """
        # Add context to all subsequent log messages
        self.logger.add_context(year=year)

        # Construct the correct season URL
        # Fixed URL structure: https://www.soccer24.com/colombia/primera-a-{year}/results/
        base_url = (
            self.config.BASE_URL
            if hasattr(self.config, "BASE_URL")
            else "https://www.soccer24.com"
        )
        season_url = f"{base_url}/colombia/primera-a-{year}/results/"

        self.logger.info(f"Starting season scraping process", extra={"url": season_url})
        self.logger.debug(
            f"Using updated URL format for season data",
            extra={"url_format": "primera-a-{year}/results/"},
        )

        try:
            # Navigate to the season page
            self.logger.debug(f"Navigating to season URL", extra={"url": season_url})
            self.driver.get(season_url)
            time.sleep(self.config.REQUEST_DELAY)
            self.logger.debug(f"Page loaded, waiting {self.config.REQUEST_DELAY}s")

            # Handle cookie consent if present
            self._accept_cookies()

            # Get all tournament seasons available
            self.logger.debug("Collecting league tournaments for season")
            season_leagues = self._get_season_leagues()
            self.logger.info(
                f"Retrieved league tournaments", extra={"count": len(season_leagues)}
            )

            # Get all match URLs from the page
            self.logger.debug("Beginning collection of match URLs")
            start_time = time.time()
            match_urls = self._get_all_match_urls()
            elapsed_time = time.time() - start_time
            self.logger.info(
                f"Completed match URL collection",
                extra={
                    "matches_found": len(match_urls),
                    "elapsed_seconds": round(elapsed_time, 2),
                },
            )

            # Create Season object
            season = Season(
                year=year,
                match_urls=match_urls,
                total_matches=len(match_urls),
                tournaments=season_leagues,
            )

            # Save to JSON for caching/reference
            self._save_season_data(season)

            self.logger.info(
                f"Season scraping completed successfully",
                extra={
                    "total_matches": season.total_matches,
                    "total_tournaments": len(season.tournaments),
                },
            )
            return season

        except Exception as e:
            self.logger.error(
                f"Failed to scrape season",
                extra={"error_type": type(e).__name__, "error_details": str(e)},
            )
            raise ScrapingError(f"Failed to scrape season {year}: {str(e)}")

    def _accept_cookies(self):
        """Handle cookie consent if present."""
        try:
            self.logger.debug("Looking for cookie consent dialog")
            cookie_selector = "button[aria-label='Accept cookies'], button.css-47sehv"
            cookie_button = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, cookie_selector))
            )
            cookie_button.click()
            self.logger.debug("Cookie consent accepted")
        except TimeoutException:
            self.logger.debug("No cookie consent popup found or already accepted")
            pass

    def _get_season_leagues(self) -> List[Dict[str, str]]:
        """
        Extract all league tournaments available for the current season.

        Returns:
            List of dictionaries containing league information
        """
        leagues = []
        try:
            # Wait for archive rows to be present
            self.logger.debug(
                f"Waiting for league elements to load (timeout: {self.config.ELEMENT_WAIT_TIMEOUT}s)"
            )
            WebDriverWait(self.driver, self.config.ELEMENT_WAIT_TIMEOUT).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, self.config.SELECTORS["archive_row"])
                )
            )

            # Get all tournament rows
            league_rows = self.driver.find_elements(
                By.CSS_SELECTOR, self.config.SELECTORS["archive_row"]
            )
            self.logger.debug(f"Found {len(league_rows)} league row elements")

            for idx, row in enumerate(league_rows):
                try:
                    # Get league name
                    league_name_element = row.find_element(
                        By.CSS_SELECTOR, self.config.SELECTORS["archive_text"]
                    )
                    league_name = league_name_element.text.strip()

                    # Get league URL
                    league_url = league_name_element.get_attribute("href")

                    # Get winner if available
                    winner = ""
                    try:
                        winner_element = row.find_element(
                            By.CSS_SELECTOR, self.config.SELECTORS["archive_winner"]
                        )
                        winner = winner_element.text.strip()
                    except NoSuchElementException:
                        self.logger.debug(
                            f"No winner found for league", extra={"league": league_name}
                        )
                        pass

                    leagues.append(
                        {"name": league_name, "url": league_url, "winner": winner}
                    )
                    self.logger.debug(
                        f"Extracted league information",
                        extra={
                            "index": idx + 1,
                            "league": league_name,
                            "has_winner": bool(winner),
                        },
                    )

                except (NoSuchElementException, StaleElementReferenceException) as e:
                    self.logger.warning(
                        f"Failed to extract league info for row {idx + 1}",
                        extra={"error_type": type(e).__name__, "error_details": str(e)},
                    )
                    continue

            return leagues

        except TimeoutException:
            self.logger.warning(
                "No league tournaments found - page may have changed structure or failed to load"
            )
            return []

    def _get_all_match_urls(self) -> List[str]:
        """
        Extract all match URLs from the season page, handling pagination by clicking 'Show more matches'.

        Returns:
            List of match URLs
        """
        match_urls = []
        more_results_available = True
        attempt = 0
        max_attempts = 30  # Limit to prevent infinite loops

        self.logger.debug(
            f"Starting match URL collection (max pagination attempts: {max_attempts})"
        )

        while more_results_available and attempt < max_attempts:
            # Allow time for page to load
            time.sleep(1)

            # Get current match links
            match_elements = self.driver.find_elements(
                By.CSS_SELECTOR, self.config.SELECTORS["match_link"]
            )
            self.logger.debug(
                f"Found {len(match_elements)} match elements on current page"
            )

            # Extract URLs
            current_urls = []
            for element in match_elements:
                href = element.get_attribute("href")
                if href is not None and "/match/" in href:
                    current_urls.append(href)

            # Add new URLs to our list (avoid duplicates)
            new_urls_count = 0
            for url in current_urls:
                if url not in match_urls:
                    match_urls.append(url)
                    new_urls_count += 1

            self.logger.debug(
                f"Extracted match URLs",
                extra={
                    "new_urls": new_urls_count,
                    "total_urls": len(match_urls),
                    "pagination_attempt": attempt + 1,
                },
            )

            # Try to find and click "Show more matches" button
            try:
                self.logger.debug("Looking for 'Show more matches' button")
                show_more_button = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable(
                        (By.CSS_SELECTOR, self.config.SELECTORS["show_more_button"])
                    )
                )
                self.driver.execute_script(
                    "arguments[0].scrollIntoView();", show_more_button
                )
                time.sleep(0.5)  # Short delay for stability
                show_more_button.click()
                self.logger.debug(
                    "Clicked 'Show more matches' button", extra={"attempt": attempt + 1}
                )
                attempt += 1

            except (
                TimeoutException,
                NoSuchElementException,
                StaleElementReferenceException,
            ) as e:
                self.logger.debug(
                    "No more 'Show more matches' button found",
                    extra={"error_type": type(e).__name__, "pagination_complete": True},
                )
                more_results_available = False

        # Remove duplicates and clean up URLs
        original_count = len(match_urls)
        match_urls = list(set(match_urls))

        if original_count != len(match_urls):
            self.logger.debug(
                f"Removed duplicate URLs",
                extra={
                    "before": original_count,
                    "after": len(match_urls),
                    "duplicates_removed": original_count - len(match_urls),
                },
            )

        self.logger.info(
            f"Match URL collection completed",
            extra={"total_urls": len(match_urls), "pagination_attempts": attempt},
        )
        return match_urls

    def _save_season_data(self, season: Season) -> None:
        """
        Save season data to a JSON file for caching and reference.

        Args:
            season: Season object to save
        """
        try:
            # Create directory if it doesn't exist
            output_dir = Path(self.config.DATA_DIR) / str(season.year)
            create_directory_if_not_exists(output_dir)
            self.logger.debug(
                f"Ensuring output directory exists", extra={"path": str(output_dir)}
            )

            # Convert Season object to dict
            season_data = {
                "year": season.year,
                "total_matches": season.total_matches,
                "tournaments": season.tournaments,
                "match_urls": season.match_urls,
                "scraped_at": str(datetime.now()),
            }

            # Save to JSON file
            output_file = output_dir / "season_data.json"
            self.logger.debug(
                f"Saving season data to file", extra={"file": str(output_file)}
            )

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(season_data, f, indent=4, ensure_ascii=False)

            self.logger.info(
                f"Season data saved successfully",
                extra={
                    "file": str(output_file),
                    "file_size_bytes": output_file.stat().st_size,
                },
            )

        except Exception as e:
            self.logger.error(
                f"Failed to save season data",
                extra={
                    "error_type": type(e).__name__,
                    "error_details": str(e),
                    "output_dir": (
                        str(output_dir) if "output_dir" in locals() else "unknown"
                    ),
                },
            )
