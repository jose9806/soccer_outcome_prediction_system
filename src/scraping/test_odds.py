#!/usr/bin/env python3
"""
Test script for OddsScraper component.
This script tests the streamlined OddsScraper that extracts specific betting markets:
1X2, Over/Under, Both Teams to Score, Correct Score, and Odd/Even.
"""

import sys
import traceback
from pathlib import Path
import json
import time
from selenium.webdriver.common.by import By

# Adjust this path to match your project structure
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.config.scraping_config import ScrapingConfig
from src.scraping.scrapers.odds import OddsScraper
from src.scraping.driver import WebDriverFactory
from src.config.logger import Logger
from src.scraping.models.soccer_extraction import OddsType


def test_odds_scraper(match_url, fast_mode=False, market_limit=None):
    """
    Test the OddsScraper with a specific URL.

    Args:
        match_url: URL of the match to scrape
        fast_mode: If True, only scrape 1X2 markets for faster testing
        market_limit: Maximum number of betting markets to extract
    """
    config = ScrapingConfig()

    # Initialize logger
    logger = Logger(
        name="TestOddsScraper",
        color="magenta",
        level=config.LOG_LEVEL,
        file_output="src/logs/test_odds_scraper.log",
    )

    logger.info(f"Starting OddsScraper test with URL: {match_url}")
    if fast_mode:
        logger.info("Running in fast mode (1X2 odds only)")
    if market_limit:
        logger.info(f"Market limit set to {market_limit}")

    driver = None
    try:
        # Initialize the driver
        driver = WebDriverFactory.create_driver()
        driver.set_page_load_timeout(60)  # Higher timeout for better reliability

        # Create the odds scraper instance
        odds_scraper = OddsScraper(driver, config)

        # Set fast mode if requested
        if fast_mode:
            odds_scraper.set_fast_mode(True)

        # Set market limit if requested
        if market_limit:
            odds_scraper.set_market_limit(market_limit)

        # Navigate to the match page
        logger.info(f"Navigating to match URL: {match_url}")
        driver.get(match_url)
        time.sleep(config.REQUEST_DELAY * 2)  # Double delay for better loading

        # Handle cookie consent if it appears
        try:
            cookie_selectors = [
                "button[id*='cookie'], button[class*='cookie'], button[id*='consent'], button[class*='consent']",
                "//button[contains(text(), 'Accept')]",
                "//button[contains(text(), 'agree')]",
            ]

            for selector in cookie_selectors:
                try:
                    if selector.startswith("//"):
                        elements = driver.find_elements(By.XPATH, selector)
                    else:
                        elements = driver.find_elements(By.CSS_SELECTOR, selector)

                    if elements:
                        elements[0].click()
                        logger.info(f"Clicked cookie consent button")
                        time.sleep(1)
                        break
                except Exception:
                    continue
        except Exception:
            pass  # Continue if cookie handling fails

        # Find and click on the odds tab
        try:
            logger.info("Attempting to locate and click on odds tab")
            odds_tab_selectors = [
                config.SELECTORS["odds_tab"],
                "a[href*='odds-comparison']",
                "a[href*='1x2-odds']",
                "a.filterOverTab",
                "//a[contains(text(), 'ODDS')]",
            ]

            clicked = False
            for selector in odds_tab_selectors:
                try:
                    if isinstance(selector, str) and selector.startswith("//"):
                        elements = driver.find_elements(By.XPATH, selector)
                        if elements:
                            elements[0].click()
                            clicked = True
                            break
                    else:
                        elements = driver.find_elements(By.CSS_SELECTOR, selector)
                        if elements:
                            elements[0].click()
                            clicked = True
                            break
                except Exception:
                    continue

            if clicked:
                logger.info("Successfully clicked on odds tab")
                time.sleep(2)
            else:
                # If direct click failed, try JavaScript click
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, "a[href*='odds']")
                    if elements:
                        driver.execute_script("arguments[0].click();", elements[0])
                        logger.info("Clicked on odds tab using JavaScript")
                        time.sleep(2)
                    else:
                        logger.warning("Could not find odds tab to click")
                except Exception as e:
                    logger.error(
                        f"Failed to click on odds tab using JavaScript: {str(e)}"
                    )
        except Exception as e:
            logger.error(f"Failed to click on odds tab: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.info("Will attempt to scrape odds directly")

        # Take a screenshot for debugging
        screenshot_path = Path("odds_page_screenshot.png")
        driver.save_screenshot(str(screenshot_path))
        logger.info(f"Saved screenshot to {screenshot_path}")

        # Get page source for debugging
        page_source_path = Path("odds_page_source.html")
        with open(page_source_path, "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        logger.info(f"Saved page source to {page_source_path}")

        # Attempt to scrape the odds
        logger.info(f"Scraping match odds...")
        start_time = time.time()
        odds_data = odds_scraper.scrape(match_url)
        elapsed_time = time.time() - start_time
        logger.info(f"Odds scraping completed in {elapsed_time:.2f} seconds")

        if odds_data:
            # Count odds data in each market
            odds_summary = {
                "Match Winner (1X2)": len(getattr(odds_data, "match_winner_odds", [])),
                "Over/Under": len(getattr(odds_data, "over_under_odds", [])),
                "Both Teams to Score": len(getattr(odds_data, "btts_odds", [])),
                "Correct Score": len(getattr(odds_data, "correct_score_odds", [])),
                "Odd/Even": len(getattr(odds_data, "odd_even_odds", [])),
            }

            # Calculate total odds entries
            total_odds = sum(odds_summary.values())

            # Print odds summary
            logger.info(f"Extracted a total of {total_odds} odds entries:")
            for market, count in odds_summary.items():
                if count > 0:
                    logger.info(f"  - {market}: {count} entries")

            # Create a dictionary representation for saving
            odds_dict = {"match_id": getattr(odds_data, "match_id", "unknown")}

            # Save sample odds for each market
            markets_to_save = {
                "match_winner_odds": "Match Winner (1X2)",
                "over_under_odds": "Over/Under",
                "btts_odds": "Both Teams to Score",
                "correct_score_odds": "Correct Score",
                "odd_even_odds": "Odd/Even",
            }

            for attr_name, display_name in markets_to_save.items():
                odds_list = getattr(odds_data, attr_name, [])
                if odds_list:
                    sample_size = min(
                        3, len(odds_list)
                    )  # Save up to 3 samples per market
                    odds_dict[attr_name] = []

                    for i in range(sample_size):
                        odds_obj = odds_list[i]
                        # Extract basic attributes common to all odds types
                        odds_entry = {
                            "bookmaker": odds_obj.bookmaker,
                            "period": odds_obj.period,
                            "market_type": odds_obj.market_type.name,
                        }

                        # Add market-specific attributes
                        if attr_name == "match_winner_odds":
                            odds_entry.update(
                                {
                                    "home_win": odds_obj.home_win,
                                    "draw": odds_obj.draw,
                                    "away_win": odds_obj.away_win,
                                }
                            )
                        elif attr_name == "over_under_odds":
                            odds_entry.update(
                                {
                                    "total": odds_obj.total,
                                    "over": odds_obj.over,
                                    "under": odds_obj.under,
                                }
                            )
                        elif attr_name == "btts_odds":
                            odds_entry.update({"yes": odds_obj.yes, "no": odds_obj.no})
                        elif attr_name == "correct_score_odds":
                            odds_entry.update(
                                {"score": odds_obj.score, "odds": odds_obj.odds}
                            )
                        elif attr_name == "odd_even_odds":
                            odds_entry.update(
                                {"odd": odds_obj.odd, "even": odds_obj.even}
                            )

                        odds_dict[attr_name].append(odds_entry)

            if odds_data and hasattr(odds_data, "match_id"):
                output_file = Path(f"test_odds_result_{odds_data.match_id}.json")
            else:
                timestamp = int(time.time())
                output_file = Path(f"test_odds_result_{timestamp}.json")

            # Save the result to the file
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(odds_dict, f, ensure_ascii=False, indent=2)

            logger.info(f"Odds data saved to {output_file}")

            logger.info(f"Odds data saved to {output_file}")

            return odds_data
        else:
            logger.error("Failed to scrape odds: No data returned")
            return None

    except Exception as e:
        logger.error(f"Error during odds scraping: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

    finally:
        if driver:
            driver.quit()
            logger.debug("Driver closed")


if __name__ == "__main__":
    # Test with a specific match URL - replace with a valid URL from your target site
    match_url = "https://www.soccer24.com/match/KAWSNeji/#/match-summary"

    # Default test settings
    fast_mode = False
    market_limit = None

    # Parse command-line arguments
    if len(sys.argv) > 1:
        match_url = sys.argv[1]

    if len(sys.argv) > 2 and sys.argv[2].lower() in ("fast", "true", "1"):
        fast_mode = True

    if len(sys.argv) > 3:
        try:
            market_limit = int(sys.argv[3])
        except ValueError:
            print(f"Invalid market limit: {sys.argv[3]}. Using default.")

    print(f"Testing OddsScraper with URL: {match_url}")
    print(f"Fast mode: {fast_mode}")
    print(f"Market limit: {market_limit}")

    result = test_odds_scraper(match_url, fast_mode, market_limit)

    if result:
        print("Test completed successfully")

        # Print a simple summary of what was extracted
        print("\nExtracted odds summary:")
        print(f"- Match Winner (1X2): {len(result.match_winner_odds)} entries")
        print(f"- Over/Under: {len(result.over_under_odds)} entries")
        print(f"- Both Teams to Score: {len(result.btts_odds)} entries")
        print(f"- Correct Score: {len(result.correct_score_odds)} entries")
        print(f"- Odd/Even: {len(result.odd_even_odds)} entries")

        # Total count
        total = (
            len(result.match_winner_odds)
            + len(result.over_under_odds)
            + len(result.btts_odds)
            + len(result.correct_score_odds)
            + len(result.odd_even_odds)
        )
        print(f"\nTotal odds entries: {total}")
    else:
        print("Test failed")
