#!/usr/bin/env python3
"""
Test script for MatchScraper component.
This script tests the MatchScraper in isolation to identify any issues.
"""

import sys
import traceback
from pathlib import Path
import json

# Adjust this path to match your project structure
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.config.scraping_config import ScrapingConfig
from src.scraping.scrapers.match import MatchScraper
from src.scraping.driver import WebDriverFactory
from src.config.logger import Logger


def test_match_scraper(match_url):
    """Test the MatchScraper with a specific URL."""
    config = ScrapingConfig()

    # Initialize logger
    logger = Logger(
        name="TestMatchScraper",
        color="green",
        level=config.LOG_LEVEL,
        file_output="src/logs/test_match_scraper.log",
    )

    logger.info(f"Starting MatchScraper test with URL: {match_url}")

    driver = None
    try:
        # Initialize the driver
        driver = WebDriverFactory.create_driver()

        # Create the match scraper instance
        match_scraper = MatchScraper(driver, config)

        # Attempt to scrape the match
        logger.info(f"Scraping match data...")
        match_data = match_scraper.scrape(match_url)

        if match_data:
            # Convert to dictionary for display
            match_dict = match_data.to_dict()

            # Print basic match information
            logger.info(
                f"Match scraped successfully: {match_data.home_team} vs {match_data.away_team}"
            )
            logger.info(f"Score: {match_data.home_score}-{match_data.away_score}")
            logger.info(f"Date: {match_data.date}")

            # Check if stats were extracted
            if match_data.full_time_stats:
                logger.info("Full time stats were extracted successfully")
                # Print some sample stats
                if match_data.full_time_stats.possession:
                    logger.info(f"Possession: {match_data.full_time_stats.possession}")
            else:
                logger.warning("No full time stats were extracted")

            # Check if odds were extracted
            if match_data.odds and hasattr(match_data.odds, "match_winner_odds"):
                logger.info(
                    f"Odds were extracted successfully: {len(match_data.odds.match_winner_odds)} bookmakers found"
                )
            else:
                logger.warning("No odds were extracted")

            # Save the result to a test file
            output_file = Path("test_match_result.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(match_dict, f, ensure_ascii=False, indent=2, default=str)

            logger.info(f"Match data saved to {output_file}")

            return match_data
        else:
            logger.error("Failed to scrape match: No data returned")
            return None

    except Exception as e:
        logger.error(f"Error during match scraping: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

    finally:
        if driver:
            driver.quit()
            logger.debug("Driver closed")


if __name__ == "__main__":
    # Test with a specific match URL - replace with a valid URL from your target site
    match_url = "https://www.soccer24.com/match/KAWSNeji/#/match-summary"

    # Allow command-line URL override
    if len(sys.argv) > 1:
        match_url = sys.argv[1]

    print(f"Testing MatchScraper with URL: {match_url}")
    result = test_match_scraper(match_url)

    if result:
        print("Test completed successfully")
    else:
        print("Test failed")
