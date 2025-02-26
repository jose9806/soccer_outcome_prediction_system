#!/usr/bin/env python3
"""
Test script for StatsScraper component.
This script tests the StatsScraper in isolation to identify any issues.
"""

import sys
import traceback
from pathlib import Path
import json
import time

# Adjust this path to match your project structure
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from selenium.webdriver.common.by import By

from src.config.scraping_config import ScrapingConfig
from src.scraping.scrapers.stats import StatsScraper
from src.scraping.driver import WebDriverFactory
from src.config.logger import Logger


# Add these imports to the test script
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def test_stats_scraper(match_url):
    """Enhanced test for the StatsScraper with better navigation and error handling."""
    config = ScrapingConfig()

    # Initialize logger
    logger = Logger(
        name="TestStatsScraper",
        color="blue",
        level=config.LOG_LEVEL,
        file_output="src/logs/test_stats_scraper.log",
    )

    logger.info(f"Starting StatsScraper test with URL: {match_url}")

    driver = None
    try:
        # Initialize the driver with increased timeouts
        driver = WebDriverFactory.create_driver()
        driver.set_page_load_timeout(60)  # Longer timeout for page load

        # Create the stats scraper instance
        stats_scraper = StatsScraper(driver, config)

        # Navigate to the match page
        logger.info(f"Navigating to match URL: {match_url}")
        driver.get(match_url)
        time.sleep(config.REQUEST_DELAY * 2)  # Wait longer for initial page load

        # Handle cookie consent if it appears
        try:
            cookie_selectors = [
                config.SELECTORS["cookie_consent"],
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
                        logger.info(
                            f"Clicked cookie consent button using selector: {selector}"
                        )
                        time.sleep(1)
                        break
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Failed to handle cookie consent: {str(e)}")

        # FIXED: Try to detect if we're already on stats page
        current_url = driver.current_url
        if "statistics" in current_url or "/match-statistics" in current_url:
            logger.info("Already on statistics page")
        else:
            # Try multiple approaches to get to the stats tab
            stats_tab_clicked = False

            try:
                # First, try to find the tab using wait
                wait = WebDriverWait(driver, 10)
                for selector in [
                    "a[href*='match-statistics']",
                    "button.wcl-tab",
                    "a.filterOverTab",
                    "a[title='Statistics']",
                ]:
                    try:
                        element = wait.until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                        )
                        element.click()
                        stats_tab_clicked = True
                        logger.info(f"Clicked stats tab using {selector}")
                        time.sleep(2)
                        break
                    except Exception:
                        continue

                # If we couldn't click, try direct XPath approaches
                if not stats_tab_clicked:
                    for xpath in [
                        "//a[contains(@href, 'match-statistics')]",
                        "//a[contains(text(), 'Statistics')]",
                        "//button[contains(@class, 'wcl-tab')]",
                        "//a[contains(@title, 'Statistics')]",
                    ]:
                        try:
                            elements = driver.find_elements(By.XPATH, xpath)
                            if elements:
                                elements[0].click()
                                stats_tab_clicked = True
                                logger.info(f"Clicked stats tab using XPath: {xpath}")
                                time.sleep(2)
                                break
                        except Exception:
                            continue
            except Exception as e:
                logger.warning(f"Error trying to click stats tab: {e}")

            if not stats_tab_clicked:
                # Try append to URL
                try:
                    base_url = driver.current_url.split("#")[0]
                    if not base_url.endswith("/"):
                        base_url += "/"
                    stats_url = f"{base_url}#/match-summary/match-statistics"
                    logger.info(f"Navigating directly to stats URL: {stats_url}")
                    driver.get(stats_url)
                    time.sleep(3)
                except Exception as e:
                    logger.warning(f"Failed to navigate directly to stats URL: {e}")

        # Take a screenshot for debugging
        screenshot_path = Path("stats_page_screenshot.png")
        driver.save_screenshot(str(screenshot_path))
        logger.info(f"Saved screenshot to {screenshot_path}")

        # Get page source for debugging
        page_source_path = Path("stats_page_source.html")
        with open(page_source_path, "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        logger.info(f"Saved page source to {page_source_path}")

        # Attempt to scrape the stats
        logger.info(f"Scraping match statistics...")
        stats_data = stats_scraper.scrape(match_url, extract_only=True)

        if stats_data:
            # Convert stats to a dictionary for display
            stats_dict = {}

            # Add each non-None attribute to the dictionary
            for attr_name in dir(stats_data):
                if not attr_name.startswith("_") and attr_name != "to_dict":
                    value = getattr(stats_data, attr_name)
                    if value is not None:
                        if isinstance(value, tuple):
                            stats_dict[attr_name] = {"home": value[0], "away": value[1]}
                        else:
                            stats_dict[attr_name] = value

            # Print basic stats information
            for stat_name, values in stats_dict.items():
                if isinstance(values, dict) and "home" in values and "away" in values:
                    logger.info(f"{stat_name}: {values['home']} - {values['away']}")

            # Count how many stats were successfully extracted
            stats_count = sum(
                1
                for attr in dir(stats_data)
                if not attr.startswith("_")
                and getattr(stats_data, attr) is not None
                and attr != "to_dict"
            )

            logger.info(f"Successfully extracted {stats_count} statistics")

            # Save the result to a test file
            output_file = Path("test_stats_result.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(stats_dict, f, ensure_ascii=False, indent=2)

            logger.info(f"Stats data saved to {output_file}")

            return stats_data
        else:
            logger.error("Failed to scrape statistics: No data returned")
            return None

    except Exception as e:
        logger.error(f"Error during stats scraping: {str(e)}")
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

    print(f"Testing StatsScraper with URL: {match_url}")
    result = test_stats_scraper(match_url)

    if result:
        print("Test completed successfully")
    else:
        print("Test failed")
