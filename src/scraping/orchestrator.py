from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from src.config.logging_config import get_logger
from src.scraping.scrapers.season import SeasonScraper
from src.scraping.scrapers.match import MatchScraper
from src.scraping.exceptions import ScrapingError, RateLimitError, StorageError
from src.scraping.driver import WebDriverFactory
from src.config.scraping_config import ScrapingConfig
from src.scraping.utils.validation import validate_match_data
from src.scraping.storage.json_storage import JsonStorage


class ScrapingOrchestrator:
    """Coordinates the scraping process across multiple components."""

    def __init__(self, config: ScrapingConfig):
        self.config = config
        self.logger = get_logger(
            name="ScrapingOrchestrator",
            color="blue",
            level=self.config.LOG_LEVEL,
            enable_file=True,
            file_path="src/logs/scraping_orchestrator.log",
        )
        self.storage = JsonStorage(config.DATA_DIR)
        self.logger.info("ScrapingOrchestrator initialized")

    def scrape_season(
        self,
        year: int,
        max_workers: int = 4,
        max_matches: Optional[int] = None,
        scrape_stats: bool = True,
        scrape_odds: bool = True,
        fast_odds: bool = False,
    ) -> None:
        """
        Enhanced method to scrape an entire season with configurable options.

        Args:
            year: The year to scrape
            max_workers: Maximum number of parallel workers
            max_matches: Maximum number of matches to scrape (for testing/debugging)
            scrape_stats: Whether to scrape match statistics
            scrape_odds: Whether to scrape odds data
            fast_odds: Whether to use fast mode for odds scraping (1X2 only)
        """
        self.logger.info(
            f"Starting season scrape for year {year}",
            extra={
                "workers": max_workers,
                "scrape_stats": scrape_stats,
                "scrape_odds": scrape_odds,
                "fast_odds": fast_odds,
            },
        )

        driver = WebDriverFactory.create_driver()
        try:
            # Initialize season scraper
            season_scraper = SeasonScraper(driver, self.config)

            try:
                season = season_scraper.scrape(year)

                # Get match URLs
                match_urls = season.match_urls
                if max_matches:
                    match_urls = match_urls[:max_matches]
                    self.logger.info(f"Limited to {max_matches} matches for testing")

                total_matches = len(match_urls)
                self.logger.info(f"Found {total_matches} matches for season {year}")

                if total_matches == 0:
                    self.logger.warning(f"No matches found for season {year}")
                    return

                # Keep track of progress
                successful = 0
                failed = 0
                skipped = 0

                # Scrape matches in parallel
                self.logger.info(
                    f"Starting parallel scraping with {max_workers} workers"
                )
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_url = {
                        executor.submit(
                            self._scrape_with_retry,
                            url,
                            2,  # max_retries
                            scrape_stats,
                            scrape_odds,
                            fast_odds,
                        ): url
                        for url in match_urls
                    }

                    for i, future in enumerate(as_completed(future_to_url)):
                        url = future_to_url[future]
                        try:
                            match_data = future.result()
                            if match_data:
                                file_path = self.storage.save_match(match_data, year)
                                match_id = match_data.get("match_id", "unknown")
                                self.logger.info(
                                    f"Successfully scraped and saved match {i+1}/{total_matches}",
                                    extra={
                                        "match_id": match_id,
                                        "file": str(file_path),
                                        "progress": f"{(i+1)/total_matches:.1%}",
                                    },
                                )
                                successful += 1
                            else:
                                self.logger.warning(
                                    f"Skipped match {i+1}/{total_matches}",
                                    extra={
                                        "url": url,
                                        "progress": f"{(i+1)/total_matches:.1%}",
                                    },
                                )
                                skipped += 1
                        except StorageError as e:
                            self.logger.error(
                                f"Storage error for match {i+1}/{total_matches}",
                                extra={"url": url, "error": str(e)},
                            )
                            failed += 1
                        except Exception as e:
                            self.logger.error(
                                f"Error processing match {i+1}/{total_matches}",
                                extra={
                                    "url": url,
                                    "error_type": type(e).__name__,
                                    "error": str(e),
                                },
                            )
                            failed += 1

                # Log summary
                self.logger.info(
                    f"Season {year} scraping completed",
                    extra={
                        "total": total_matches,
                        "successful": successful,
                        "failed": failed,
                        "skipped": skipped,
                        "success_rate": (
                            f"{successful/total_matches:.1%}"
                            if total_matches > 0
                            else "0%"
                        ),
                    },
                )

            except ScrapingError as e:
                self.logger.error(
                    f"Scraping error for season {year}", extra={"error": str(e)}
                )
            except Exception as e:
                self.logger.error(
                    f"Unexpected error scraping season {year}",
                    extra={"error_type": type(e).__name__, "error": str(e)},
                )

        finally:
            driver.quit()
            self.logger.debug("Driver closed")

    def _scrape_with_retry(
        self,
        match_url: str,
        max_retries: int = 2,
        scrape_stats: bool = True,
        scrape_odds: bool = True,
        fast_odds: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Scrape a match with retry mechanism.

        Args:
            match_url: URL of the match to scrape
            max_retries: Maximum number of retry attempts
            scrape_stats: Whether to scrape match statistics
            scrape_odds: Whether to scrape odds data
            fast_odds: Whether to use fast mode for odds scraping

        Returns:
            Dictionary containing match data if successful, None otherwise
        """
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    self.logger.info(f"Retry attempt {attempt} for {match_url}")
                    # Progressive backoff
                    time.sleep(self.config.RETRY_DELAY * attempt)

                result = self._scrape_match(
                    match_url, scrape_stats, scrape_odds, fast_odds
                )

                if result:
                    if attempt > 0:
                        self.logger.info(
                            f"Successfully scraped on retry attempt {attempt}"
                        )
                    return result

            except RateLimitError as e:
                # Always respect rate limits
                wait_time = e.retry_after or self.config.RETRY_DELAY * (attempt + 1)
                self.logger.warning(
                    f"Rate limit hit on attempt {attempt+1}",
                    extra={"url": match_url, "wait_time": wait_time},
                )
                time.sleep(wait_time)

            except Exception as e:
                self.logger.error(
                    f"Error on attempt {attempt+1}",
                    extra={
                        "url": match_url,
                        "error_type": type(e).__name__,
                        "error": str(e),
                    },
                )

                # If it's the last attempt, give up
                if attempt == max_retries:
                    self.logger.error(f"All retry attempts failed for {match_url}")
                    return None

        return None

    def _scrape_match(
        self,
        match_url: str,
        scrape_stats: bool = True,
        scrape_odds: bool = True,
        fast_odds: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Enhanced method to scrape a single match with configurable options.

        Args:
            match_url: URL of the match to scrape
            scrape_stats: Whether to scrape match statistics
            scrape_odds: Whether to scrape odds data
            fast_odds: Whether to use fast mode for odds scraping

        Returns:
            Dictionary containing match data if successful, None otherwise
        """
        self.logger.debug(
            f"Starting to scrape match",
            extra={
                "url": match_url,
                "scrape_stats": scrape_stats,
                "scrape_odds": scrape_odds,
                "fast_odds": fast_odds,
            },
        )

        driver = None
        try:
            # Create a new driver for this match
            driver = WebDriverFactory.create_driver()

            # Initialize match scraper with options
            match_scraper = MatchScraper(
                driver,
                self.config,
                scrape_stats=scrape_stats,
                scrape_odds=scrape_odds,
                fast_odds=fast_odds,
            )

            # Collect match data (includes stats and odds)
            self.logger.debug(f"Scraping match data", extra={"url": match_url})
            match_data = match_scraper.scrape(match_url)

            if not match_data:
                raise ScrapingError(
                    f"Failed to scrape basic match data for {match_url}"
                )

            # Convert to dictionary for storage
            match_dict = match_data.to_dict()

            # Add source URL
            match_dict["url"] = match_url

            # Validate data
            if validate_match_data(match_dict):
                self.logger.debug(
                    f"Match data validated successfully",
                    extra={"match_id": match_data.match_id},
                )
                return match_dict

            raise ScrapingError("Match data validation failed")

        except RateLimitError as e:
            self.logger.warning(
                f"Rate limit hit", extra={"url": match_url, "error": str(e)}
            )
            raise  # Let retry mechanism handle it

        except ScrapingError as e:
            self.logger.error(
                f"Scraping error for match", extra={"url": match_url, "error": str(e)}
            )
            return None

        except Exception as e:
            self.logger.error(
                f"Unexpected error scraping match",
                extra={
                    "url": match_url,
                    "error_type": type(e).__name__,
                    "error": str(e),
                },
            )
            return None

        finally:
            if driver:
                driver.quit()
                self.logger.debug(f"Driver closed for match", extra={"url": match_url})

    def resume_season(
        self,
        year: int,
        max_workers: int = 4,
        max_matches: Optional[int] = None,
        scrape_stats: bool = True,
        scrape_odds: bool = True,
        fast_odds: bool = False,
    ) -> None:
        """
        Enhanced method to resume scraping a season with configurable options.

        Args:
            year: The year to resume
            max_workers: Maximum number of parallel workers
            max_matches: Maximum number of matches to scrape (for testing)
            scrape_stats: Whether to scrape match statistics
            scrape_odds: Whether to scrape odds data
            fast_odds: Whether to use fast mode for odds scraping
        """
        self.logger.info(
            f"Resuming season scrape for year {year}",
            extra={
                "workers": max_workers,
                "scrape_stats": scrape_stats,
                "scrape_odds": scrape_odds,
                "fast_odds": fast_odds,
            },
        )

        try:
            # Get list of already scraped match IDs
            existing_matches = set(self.storage.list_matches(year))
            self.logger.info(
                f"Found {len(existing_matches)} existing matches for year {year}"
            )

            # Create driver and get season data
            driver = WebDriverFactory.create_driver()
            try:
                season_scraper = SeasonScraper(driver, self.config)
                season = season_scraper.scrape(year)

                # Extract match IDs from URLs
                all_match_ids = [
                    url.split("/")[-1].split("#")[0] for url in season.match_urls
                ]

                # Filter out already scraped matches
                new_match_ids = [
                    mid for mid in all_match_ids if mid not in existing_matches
                ]
                new_urls = [
                    url
                    for url, mid in zip(season.match_urls, all_match_ids)
                    if mid in new_match_ids
                ]

                # Apply max_matches limit if specified
                if max_matches and len(new_urls) > max_matches:
                    new_urls = new_urls[:max_matches]
                    self.logger.info(
                        f"Limited to {max_matches} new matches for testing"
                    )

                self.logger.info(
                    f"Found {len(new_urls)} new matches to scrape for season {year}",
                    extra={
                        "total_matches": len(season.match_urls),
                        "existing_matches": len(existing_matches),
                        "new_matches": len(new_urls),
                    },
                )

                if not new_urls:
                    self.logger.info(f"No new matches to scrape for season {year}")
                    return

                # Scrape the new matches
                # Reuse the logic from scrape_season but with filtered URLs
                successful = 0
                failed = 0
                skipped = 0
                total_matches = len(new_urls)

                self.logger.info(
                    f"Starting parallel scraping of new matches with {max_workers} workers"
                )
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_url = {
                        executor.submit(
                            self._scrape_with_retry,
                            url,
                            2,  # max_retries
                            scrape_stats,
                            scrape_odds,
                            fast_odds,
                        ): url
                        for url in new_urls
                    }

                    for i, future in enumerate(as_completed(future_to_url)):
                        url = future_to_url[future]
                        try:
                            match_data = future.result()
                            if match_data:
                                file_path = self.storage.save_match(match_data, year)
                                match_id = match_data.get("match_id", "unknown")
                                self.logger.info(
                                    f"Successfully scraped and saved new match {i+1}/{total_matches}",
                                    extra={
                                        "match_id": match_id,
                                        "file": str(file_path),
                                        "progress": f"{(i+1)/total_matches:.1%}",
                                    },
                                )
                                successful += 1
                            else:
                                self.logger.warning(
                                    f"Skipped new match {i+1}/{total_matches}",
                                    extra={
                                        "url": url,
                                        "progress": f"{(i+1)/total_matches:.1%}",
                                    },
                                )
                                skipped += 1
                        except StorageError as e:
                            self.logger.error(
                                f"Storage error for new match {i+1}/{total_matches}",
                                extra={"url": url, "error": str(e)},
                            )
                            failed += 1
                        except Exception as e:
                            self.logger.error(
                                f"Error processing new match {i+1}/{total_matches}",
                                extra={
                                    "url": url,
                                    "error_type": type(e).__name__,
                                    "error": str(e),
                                },
                            )
                            failed += 1

                # Log summary
                self.logger.info(
                    f"Season {year} resume scraping completed",
                    extra={
                        "total_new": total_matches,
                        "successful": successful,
                        "failed": failed,
                        "skipped": skipped,
                        "success_rate": (
                            f"{successful/total_matches:.1%}"
                            if total_matches > 0
                            else "0%"
                        ),
                    },
                )

            finally:
                if driver:
                    driver.quit()
                    self.logger.debug("Driver closed")

        except Exception as e:
            self.logger.error(
                f"Error resuming season {year}",
                extra={"error_type": type(e).__name__, "error": str(e)},
            )


if __name__ == "__main__":
    config = ScrapingConfig()
    orchestrator = ScrapingOrchestrator(config)

    # Example usage
    # Scrape a specific season with full options
    # orchestrator.scrape_season(
    #     2017,
    #     max_workers=4,
    #     max_matches=10,  # Limit for testing
    #     scrape_stats=True,
    #     scrape_odds=True,
    #     fast_odds=True   # Only get 1X2 odds for better performance
    # )

    # Or resume an existing season with options

    orchestrator.resume_season(
        2025,
        max_workers=4,
        scrape_stats=True,
        scrape_odds=True,
        fast_odds=True,
    )
