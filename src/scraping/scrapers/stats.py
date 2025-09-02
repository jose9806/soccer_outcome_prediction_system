import time
import re
from typing import Tuple, Optional, Dict

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from selenium.common.exceptions import (
    NoSuchElementException,
    StaleElementReferenceException,
)

from src.config.logging_config import get_logger
from src.scraping.models.soccer_extraction import MatchStats
from src.scraping.scrapers.base import BaseScraper


class StatsScraper(BaseScraper):
    """Scraper for collecting match statistics."""

    # Define class-level stat mappings for use across all methods
    STAT_MAPPINGS = {
        "Ball Possession": "possession",
        "Goal Attempts": "goal_attempts",
        "Shots on Goal": "shots_on_goal",
        "Shots off Goal": "shots_off_goal",
        "Blocked Shots": "blocked_shots",
        "Corner Kicks": "corner_kicks",
        "Goalkeeper Saves": "goalkeeper_saves",
        "Offsides": "offsides",
        "Fouls": "fouls",
        "Yellow Cards": "yellow_cards",
        "Red Cards": "red_cards",
        "Big Chances": "big_chances",
        "Shots inside box": "shots_inside_box",
        "Shots outside box": "shots_outside_box",
        "Expected goals (xG)": "expected_goals",
    }

    def __init__(self, driver, config, **kwargs):
        # Only pass driver and config to BaseScraper
        super().__init__(driver, config)
        # Initialize the custom logger with appropriate configuration
        self.logger = get_logger(
            name="StatsScraper",
            color="blue",  # Use blue for this scraper to distinguish logs
            level=(
                self.config.LOG_LEVEL if hasattr(self.config, "LOG_LEVEL") else 20
            ),  # INFO=20
            enable_file=True,
            file_path="src/logs/stats_scraper.log",
        )
        self.logger.info("StatsScraper initialized")

    def wait_for_clickable_element(self, selector, timeout=None):
        """
        Wait for an element to be clickable and return it.

        Args:
            selector: Either a CSS selector string or a tuple of (By.X, "selector")
            timeout: Custom timeout in seconds, defaults to config value if None

        Returns:
            The WebElement once it's clickable
        """
        if timeout is None:
            timeout = self.config.ELEMENT_WAIT_TIMEOUT

        wait = WebDriverWait(self.driver, timeout)

        try:
            # Handle both string CSS selectors and (By.X, "selector") tuples
            if isinstance(selector, str):
                element = wait.until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
            else:
                # Assume it's a tuple of (By.X, "selector")
                element = wait.until(EC.element_to_be_clickable(selector))

            return element
        except Exception as e:
            self.logger.warning(f"Timeout waiting for element: {selector}")
            raise e

    def wait_for_stats_to_load(self, timeout=10):
        """Wait for statistics to be fully loaded on the page."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Check for key stats indicators
                stats_indicators = [
                    "div[data-testid='wcl-statistics']",
                    ".stat__row",
                    "div[class*='wcl-row_']",
                    "div[title='Ball Possession']",
                ]

                for indicator in stats_indicators:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, indicator)
                    if elements:
                        # If elements are found, wait a bit more for all stats to load
                        time.sleep(1)
                        return True

                # If no elements found yet, wait briefly and try again
                time.sleep(0.5)
            except Exception:
                # On any error, continue waiting
                time.sleep(0.5)

        # If we've waited the full timeout and still no stats, return False
        self.logger.warning(f"Waited {timeout}s but stats haven't loaded")
        return False

    def log_page_structure(self):
        """Log the overall structure of the page to help debug selector issues."""
        try:
            # Get just a high-level outline of the page structure
            body = self.driver.find_element(By.TAG_NAME, "body")

            # Log direct children of body
            children = body.find_elements(By.XPATH, "./*")
            self.logger.debug(f"Page has {len(children)} top-level elements")

            # Log major containers and their classes
            containers = body.find_elements(
                By.XPATH,
                ".//div[contains(@class, 'container') or contains(@class, 'wrapper') or contains(@class, 'content')]",
            )

            for container in containers[:5]:  # Limit to first 5
                try:
                    class_attr = container.get_attribute("class")
                    id_attr = container.get_attribute("id")
                    self.logger.debug(
                        f"Found container: class='{class_attr}', id='{id_attr}'"
                    )
                except:
                    pass

            # Check specifically for stats-related elements
            stats_elements = body.find_elements(
                By.XPATH,
                ".//*[contains(@class, 'stat') or contains(@class, 'wcl-') or contains(@data-testid, 'wcl-')]",
            )

            self.logger.debug(f"Found {len(stats_elements)} stats-related elements")

            # Log a few examples of stats elements
            for element in stats_elements[:3]:  # Limit to first 3
                try:
                    class_attr = element.get_attribute("class")
                    tag_name = element.tag_name
                    text = (
                        element.text[:50] + "..."
                        if len(element.text) > 50
                        else element.text
                    )
                    self.logger.debug(
                        f"Stats element: {tag_name}, class='{class_attr}', text='{text}'"
                    )
                except:
                    pass

        except Exception as e:
            self.logger.debug(f"Error logging page structure: {str(e)}")

    def scrape(self, match_url: str, extract_only: bool = False) -> MatchStats:
        """
        Scrape match statistics.

        Args:
            match_url: URL of the match to scrape
            extract_only: If True, assumes we're already on the stats tab and only extracts data
                         without navigating to the page or clicking tabs

        Returns:
            MatchStats object with all available statistics
        """
        try:
            if not extract_only:
                self.logger.add_context(url=match_url)
                self.logger.info("Starting stats scraping process")

                # Navigate to the match page if needed
                if not self.driver.current_url == match_url:
                    self.logger.debug(f"Navigating to match URL")
                    self.driver.get(match_url)
                    time.sleep(self.config.REQUEST_DELAY)

                    # Click on stats tab
                    self.logger.debug("Clicking on stats tab")
                    stats_tab = self.wait_for_clickable_element(
                        self.config.SELECTORS["stats_tab"]
                    )
                    stats_tab.click()
                    time.sleep(1)

                    # Wait for stats to load
                    self.logger.debug("Waiting for stats content to load")
                    self.wait_for_stats_to_load(timeout=10)

                # Log page structure for debugging
                self.log_page_structure()
            else:
                self.logger.debug(
                    "Extract only mode: assuming we're already on stats tab"
                )

            # First, determine if we have the new WCL structure or the traditional structure
            has_wcl_structure = False
            try:
                # Try multiple selectors for WCL detection
                wcl_selectors = [
                    self.config.SELECTORS["wcl_stats_container"],
                    "div[data-testid='wcl-statistics']",
                    "div[class*='wcl-row_']",
                    "strong[data-testid^='wcl-scores-simpleText']",
                ]

                for selector in wcl_selectors:
                    wcl_elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if wcl_elements:
                        has_wcl_structure = True
                        self.logger.debug(
                            f"Detected WCL stats structure using selector: {selector}"
                        )
                        break

                if not has_wcl_structure:
                    # Try one more approach using XPath to find elements with wcl in class names
                    wcl_xpath = "//*[contains(@class, 'wcl-')]"
                    wcl_elements = self.driver.find_elements(By.XPATH, wcl_xpath)
                    if wcl_elements:
                        has_wcl_structure = True
                        self.logger.debug("Detected WCL stats structure using XPath")
            except Exception as e:
                self.logger.debug(f"Failed to check for WCL structure: {str(e)}")

            # Extract all available statistics
            stats_dict = {}

            if has_wcl_structure:
                # Use the WCL extraction approach
                self.logger.debug("Using WCL extraction method")

                # Get all stat labels from the page
                wcl_labels = self.driver.find_elements(
                    By.CSS_SELECTOR, self.config.SELECTORS["wcl_stats_label"]
                )

                # For each found label, extract the corresponding values
                for label_element in wcl_labels:
                    try:
                        label_text = label_element.text.strip()
                        if not label_text:  # Skip empty labels
                            continue

                        # Find matching attribute in our model
                        attr_name = None
                        for stat_label, attr in self.STAT_MAPPINGS.items():
                            if stat_label.lower() in label_text.lower():
                                attr_name = attr
                                break

                        if (
                            attr_name and attr_name not in stats_dict
                        ):  # Only extract if not already found
                            # Try multiple approaches to find the parent row
                            try:
                                # Try first with the standard ancestor approach
                                parent_row = label_element.find_element(
                                    By.XPATH,
                                    "./ancestor::div[contains(@class, 'wcl-row_') or contains(@class, 'wcl-category_') or contains(@class, '-row_')]",
                                )

                                # Try to find home and away values with multiple selectors
                                home_value = None
                                away_value = None

                                # Try CSS selector first
                                try:
                                    home_element = parent_row.find_element(
                                        By.CSS_SELECTOR,
                                        "div[class*='wcl-homeValue_'], div[class*='-homeValue_']",
                                    )
                                    away_element = parent_row.find_element(
                                        By.CSS_SELECTOR,
                                        "div[class*='wcl-awayValue_'], div[class*='-awayValue_']",
                                    )

                                    # Extract numeric values
                                    home_text = home_element.text.strip()
                                    away_text = away_element.text.strip()

                                    # Extract numbers using regex
                                    home_match = re.search(r"([\d.]+)%?", home_text)
                                    away_match = re.search(r"([\d.]+)%?", away_text)

                                    if home_match and away_match:
                                        # Handle both integer and float values
                                        if "." in home_match.group(
                                            1
                                        ) or "." in away_match.group(1):
                                            home_value = float(home_match.group(1))
                                            away_value = float(away_match.group(1))
                                        else:
                                            home_value = int(home_match.group(1))
                                            away_value = int(away_match.group(1))
                                except Exception:
                                    pass

                                # If previous approach failed, try finding values based on their relative position
                                if home_value is None or away_value is None:
                                    try:
                                        # Find all value elements in the row
                                        value_elements = parent_row.find_elements(
                                            By.XPATH,
                                            ".//*[contains(@class, 'wcl-value_') or contains(@class, '-value_')]",
                                        )

                                        if len(value_elements) >= 2:
                                            # Assume first is home, second is away
                                            home_text = value_elements[0].text.strip()
                                            away_text = value_elements[1].text.strip()

                                            # Extract numbers using regex
                                            home_match = re.search(
                                                r"([\d.]+)%?", home_text
                                            )
                                            away_match = re.search(
                                                r"([\d.]+)%?", away_text
                                            )

                                            if home_match and away_match:
                                                # Handle both integer and float values
                                                if "." in home_match.group(
                                                    1
                                                ) or "." in away_match.group(1):
                                                    home_value = float(
                                                        home_match.group(1)
                                                    )
                                                    away_value = float(
                                                        away_match.group(1)
                                                    )
                                                else:
                                                    home_value = int(
                                                        home_match.group(1)
                                                    )
                                                    away_value = int(
                                                        away_match.group(1)
                                                    )
                                    except Exception:
                                        pass

                                # If we found valid values, add to our stats dictionary
                                if home_value is not None and away_value is not None:
                                    stats_dict[attr_name] = (home_value, away_value)
                                    self.logger.debug(
                                        f"Extracted {label_text} from WCL structure",
                                        extra={"home": home_value, "away": away_value},
                                    )

                            except Exception as e:
                                self.logger.debug(
                                    f"Failed to extract values for {label_text}: {str(e)}"
                                )
                    except Exception as e:
                        self.logger.debug(f"Error processing label element: {str(e)}")

            # If we didn't find all stats with the WCL approach, try all the other methods
            missing_stats = len(self.STAT_MAPPINGS) - len(stats_dict)
            if missing_stats > 0:
                self.logger.debug(
                    f"Missing {missing_stats} stats after WCL extraction, trying traditional methods"
                )

                # Try traditional approaches for remaining stats
                for stat_label, attr_name in self.STAT_MAPPINGS.items():
                    if (
                        attr_name not in stats_dict
                    ):  # Only extract if we don't already have it
                        try:
                            stat_value = self._extract_stat_by_label(stat_label)
                            if stat_value:
                                stats_dict[attr_name] = stat_value
                                self.logger.debug(
                                    f"Extracted {stat_label} using traditional method",
                                    extra={
                                        "home": stat_value[0],
                                        "away": stat_value[1],
                                    },
                                )
                        except Exception as e:
                            self.logger.debug(
                                f"Failed to extract {stat_label} using traditional method",
                                extra={"error": str(e)},
                            )

            # If we still don't have all stats, try direct extraction methods
            missing_stats = len(self.STAT_MAPPINGS) - len(stats_dict)
            if missing_stats > 0:
                self.logger.debug(
                    f"Still missing {missing_stats} stats, trying direct extraction methods"
                )
                stats_dict = self._extract_all_stats_directly(stats_dict)

            # Finally, try scanning all rows for any remaining stats
            missing_stats = len(self.STAT_MAPPINGS) - len(stats_dict)
            if missing_stats > 0:
                self.logger.debug(
                    f"Still missing {missing_stats} stats, scanning all rows"
                )
                stats_dict = self._extract_all_stats_from_rows(stats_dict)

            self.logger.info(
                "Stats extraction complete", extra={"stats_collected": len(stats_dict)}
            )

            # Create and return the MatchStats object with all extracted stats
            match_stats = MatchStats(**stats_dict)

            return match_stats

        except Exception as e:
            self.logger.error(
                f"Failed to scrape match statistics",
                extra={"error_type": type(e).__name__, "error_details": str(e)},
            )
            # Return empty stats object rather than raising
            return MatchStats()

    def _extract_stat_by_label(self, stat_label: str) -> Optional[Tuple[int, int]]:
        """
        Enhanced generic method to extract any statistic by its label.

        Args:
            stat_label: The text label of the statistic to extract

        Returns:
            Tuple of (home_value, away_value) or None if not found
        """
        try:
            # WCL structure approach
            try:
                # Try using XPath for more flexibility in matching text content
                wcl_xpath = f"//strong[contains(@data-testid, 'wcl-scores-simpleText') and contains(text(), '{stat_label}')]"
                label_elements = self.driver.find_elements(By.XPATH, wcl_xpath)

                if not label_elements:
                    # Try a more generic approach to find elements containing the text
                    wcl_xpath = f"//*[contains(text(), '{stat_label}') and (contains(@class, 'wcl-') or contains(@data-testid, 'wcl-'))]"
                    label_elements = self.driver.find_elements(By.XPATH, wcl_xpath)

                if label_elements:
                    # Find the row containing this label
                    parent_xpath = "./ancestor::div[contains(@class, 'wcl-row_') or contains(@class, '-row_')]"
                    parent_row = label_elements[0].find_element(By.XPATH, parent_xpath)

                    # Look for home and away values with flexible selectors
                    home_elements = parent_row.find_elements(
                        By.XPATH,
                        ".//*[contains(@class, 'wcl-homeValue_') or contains(@class, '-homeValue_')]",
                    )
                    away_elements = parent_row.find_elements(
                        By.XPATH,
                        ".//*[contains(@class, 'wcl-awayValue_') or contains(@class, '-awayValue_')]",
                    )

                    # If direct class match doesn't work, try positional approach
                    if not home_elements or not away_elements:
                        # Assume first value is home, second is away
                        value_elements = parent_row.find_elements(
                            By.XPATH,
                            ".//*[contains(@class, 'value_') or contains(@class, 'Value_')]",
                        )
                        if len(value_elements) >= 2:
                            home_elements = [value_elements[0]]
                            away_elements = [value_elements[1]]

                    if home_elements and away_elements:
                        home_text = home_elements[0].text.strip()
                        away_text = away_elements[0].text.strip()

                        # Extract numbers with regex, handling decimals for xG
                        home_match = re.search(r"([\d.]+)", home_text)
                        away_match = re.search(r"([\d.]+)", away_text)

                        if home_match and away_match:
                            # Handle potential float values (like xG)
                            if "." in home_match.group(1) or "." in away_match.group(1):
                                home_value = float(home_match.group(1))
                                away_value = float(away_match.group(1))
                                # Convert to int if needed for most stats
                                if "expected_goals" not in self.STAT_MAPPINGS.values():
                                    home_value = int(home_value)
                                    away_value = int(away_value)
                            else:
                                home_value = int(home_match.group(1))
                                away_value = int(away_match.group(1))

                            self.logger.debug(
                                f"Extracted {stat_label} from WCL structure",
                                extra={"home": home_value, "away": away_value},
                            )
                            return home_value, away_value
            except (
                NoSuchElementException,
                StaleElementReferenceException,
                ValueError,
                IndexError,
            ) as e:
                self.logger.debug(
                    f"WCL extraction approach failed for {stat_label}: {str(e)}"
                )

            # Traditional approaches - keep these as fallbacks

            # Approach 1: Find by title attribute
            try:
                stat_element = self.driver.find_element(
                    By.CSS_SELECTOR, f"div[title='{stat_label}']"
                )
                parent = stat_element.find_element(By.XPATH, "./..")

                # Extract the values
                text = parent.text
                values = re.findall(r"[\d.]+", text)

                if len(values) >= 2:
                    # Check if these are likely float values
                    if "." in values[0] or "." in values[1]:
                        home_value = float(values[0])
                        away_value = float(values[1])
                        # Convert to int if needed for most stats
                        if stat_label != "Expected goals (xG)":
                            home_value = int(home_value)
                            away_value = int(away_value)
                    else:
                        home_value = int(values[0])
                        away_value = int(values[1])

                    self.logger.debug(
                        f"Extracted {stat_label} by title",
                        extra={"home": home_value, "away": away_value},
                    )
                    return home_value, away_value
            except NoSuchElementException:
                pass

            # Approach 2: Find by text content in any element
            try:
                # Use XPath to find elements containing the stat label text
                xpath = f"//*[contains(text(), '{stat_label}')]"
                elements = self.driver.find_elements(By.XPATH, xpath)

                for element in elements:
                    try:
                        # Look up to find parent container and then extract values
                        parent = element.find_element(By.XPATH, "./ancestor::div[3]")
                        text = parent.text
                        values = re.findall(r"[\d.]+", text)

                        if len(values) >= 2:
                            # Check if these are likely float values
                            if "." in values[0] or "." in values[1]:
                                home_value = float(values[0])
                                away_value = float(values[1])
                                # Convert to int if needed for most stats
                                if stat_label != "Expected goals (xG)":
                                    home_value = int(home_value)
                                    away_value = int(away_value)
                            else:
                                home_value = int(values[0])
                                away_value = int(values[1])

                            self.logger.debug(
                                f"Extracted {stat_label} by text content",
                                extra={"home": home_value, "away": away_value},
                            )
                            return home_value, away_value
                    except (NoSuchElementException, IndexError):
                        continue
            except Exception as e:
                self.logger.debug(f"Text content approach failed: {str(e)}")

            # Approach 3: Parse all stat rows and look for matching text
            try:
                # Get all potential stat rows
                row_selectors = [
                    ".stat__row",
                    ".statRow",
                    "div[class*='wcl-row_']",
                    "div[class*='-row_']",
                ]

                for selector in row_selectors:
                    try:
                        rows = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        for row in rows:
                            row_text = row.text.strip()
                            if stat_label.lower() in row_text.lower():
                                values = re.findall(r"[\d.]+", row_text)
                                if len(values) >= 2:
                                    # Check if these are likely float values
                                    if "." in values[0] or "." in values[1]:
                                        home_value = float(values[0])
                                        away_value = float(values[1])
                                        # Convert to int if needed for most stats
                                        if stat_label != "Expected goals (xG)":
                                            home_value = int(home_value)
                                            away_value = int(away_value)
                                    else:
                                        home_value = int(values[0])
                                        away_value = int(values[1])

                                    self.logger.debug(
                                        f"Extracted {stat_label} from row text",
                                        extra={"home": home_value, "away": away_value},
                                    )
                                    return home_value, away_value
                    except Exception:
                        continue
            except Exception as e:
                self.logger.debug(f"Row parsing approach failed: {str(e)}")

            # Approach 4: Try direct parent-child relationships for WCL structure
            try:
                # Try to find the stat label again with a more direct approach
                wcl_xpath = f"//strong[contains(@class, 'wcl-simpleText') and contains(text(), '{stat_label}')]"
                elements = self.driver.find_elements(By.XPATH, wcl_xpath)

                if elements:
                    # For WCL structure, sometimes we need to go up multiple levels
                    for level in range(1, 5):  # Try different parent levels
                        try:
                            # Build parent XPath
                            parent_xpath = "./." + "/..".join(
                                ["" for _ in range(level)]
                            )
                            parent = elements[0].find_element(By.XPATH, parent_xpath)

                            # Now try to find home and away values directly in this parent
                            home_values = parent.find_elements(
                                By.XPATH, ".//*[contains(@class, 'homeValue')]"
                            )
                            away_values = parent.find_elements(
                                By.XPATH, ".//*[contains(@class, 'awayValue')]"
                            )

                            if home_values and away_values:
                                home_text = home_values[0].text.strip()
                                away_text = away_values[0].text.strip()

                                home_match = re.search(r"([\d.]+)", home_text)
                                away_match = re.search(r"([\d.]+)", away_text)

                                if home_match and away_match:
                                    # Handle both integer and float values
                                    if "." in home_match.group(
                                        1
                                    ) or "." in away_match.group(1):
                                        home_value = float(home_match.group(1))
                                        away_value = float(away_match.group(1))
                                        # Convert to int if needed for most stats
                                        if stat_label != "Expected goals (xG)":
                                            home_value = int(home_value)
                                            away_value = int(away_value)
                                    else:
                                        home_value = int(home_match.group(1))
                                        away_value = int(away_match.group(1))

                                    self.logger.debug(
                                        f"Extracted {stat_label} using direct parent-child",
                                        extra={"home": home_value, "away": away_value},
                                    )
                                    return home_value, away_value
                        except Exception:
                            continue  # Try next parent level
            except Exception as e:
                self.logger.debug(f"Direct parent-child approach failed: {str(e)}")

            # Approach 5: As a last resort, search the entire page for this stat
            try:
                # Get the entire page text
                page_text = self.driver.find_element(By.TAG_NAME, "body").text

                # Try to find a pattern like "Ball Possession 45% 55%" or "Shots on Goal 5 2"
                pattern = rf"{re.escape(stat_label)}.*?([\d.]+)[%]?.*?([\d.]+)[%]?"
                match = re.search(pattern, page_text, re.IGNORECASE | re.DOTALL)

                if match and len(match.groups()) >= 2:
                    # Handle both integer and float values
                    if "." in match.group(1) or "." in match.group(2):
                        home_value = float(match.group(1))
                        away_value = float(match.group(2))
                        # Convert to int if needed for most stats
                        if stat_label != "Expected goals (xG)":
                            home_value = int(home_value)
                            away_value = int(away_value)
                    else:
                        home_value = int(match.group(1))
                        away_value = int(match.group(2))

                    self.logger.debug(
                        f"Extracted {stat_label} from page text",
                        extra={"home": home_value, "away": away_value},
                    )
                    return home_value, away_value
            except Exception as e:
                self.logger.debug(f"Page text search failed: {str(e)}")

            # Return None if stat not found
            self.logger.debug(f"{stat_label} not found on page")
            return None

        except Exception as e:
            self.logger.warning(
                f"Failed to extract {stat_label}", extra={"error": str(e)}
            )
            return None

    def _extract_all_stats_directly(self, stats_dict):
        """
        Directly extract all known stats types using targeted selectors.
        This is a fallback approach when the main extraction methods miss some stats.

        Args:
            stats_dict: Dictionary to update with extracted stats

        Returns:
            Updated stats_dict
        """
        for stat_label, attr_name in self.STAT_MAPPINGS.items():
            if attr_name in stats_dict:  # Skip if already extracted
                continue

            try:
                # Try to find this specific stat by direct text search
                xpath = f"//*[contains(text(), '{stat_label}')]"
                elements = self.driver.find_elements(By.XPATH, xpath)

                if elements:
                    # For each element that might be our stat label
                    for element in elements:
                        try:
                            # Check various parent levels to find the stat values
                            for level in range(1, 5):  # Check up to 4 levels up
                                try:
                                    # Build XPath to go up specified number of levels
                                    ancestor_xpath = "./." + "/..".join(
                                        ["" for _ in range(level)]
                                    )
                                    parent = element.find_element(
                                        By.XPATH, ancestor_xpath
                                    )
                                    parent_text = parent.text.strip()

                                    # If parent contains the label text and at least two numbers
                                    if stat_label in parent_text:
                                        values = re.findall(r"([\d.]+)%?", parent_text)

                                        if len(values) >= 2:
                                            # Determine if we're dealing with integers or floats
                                            if (
                                                attr_name == "expected_goals"
                                                or "." in values[0]
                                                or "." in values[-1]
                                            ):
                                                home_value = float(values[0])
                                                away_value = float(values[-1])
                                                # Convert to int if needed for most stats
                                                if attr_name != "expected_goals":
                                                    home_value = int(home_value)
                                                    away_value = int(away_value)
                                            else:
                                                home_value = int(values[0])
                                                away_value = int(values[-1])

                                            stats_dict[attr_name] = (
                                                home_value,
                                                away_value,
                                            )
                                            self.logger.debug(
                                                f"Extracted {stat_label} directly",
                                                extra={
                                                    "home": home_value,
                                                    "away": away_value,
                                                },
                                            )
                                            break  # Found what we need
                                except Exception:
                                    continue  # Try next level

                            if attr_name in stats_dict:
                                break  # Successfully extracted this stat
                        except Exception:
                            continue  # Try next element
            except Exception as e:
                self.logger.debug(f"Failed to extract {stat_label} directly: {str(e)}")

        return stats_dict

    def _extract_all_stats_from_rows(self, stats_dict):
        """
        Scan all potential stat rows in the page to extract missing stats.

        Args:
            stats_dict: Dictionary to update with extracted stats

        Returns:
            Updated stats_dict
        """
        # Get all potential rows that might contain stats
        row_selectors = [
            "div[class*='wcl-row_']",
            "div[class*='-row_']",
            ".stat__row",
            ".statRow",
            "div[data-testid='wcl-statistics'] > div",
            "div.section > div",
        ]

        all_rows = []
        for selector in row_selectors:
            try:
                rows = self.driver.find_elements(By.CSS_SELECTOR, selector)
                all_rows.extend(rows)
            except Exception:
                pass

        # Process each row to extract stats
        for row in all_rows:
            try:
                row_text = row.text.strip()
                if not row_text:
                    continue

                # Check if this row contains a known stat
                for stat_label, attr_name in self.STAT_MAPPINGS.items():
                    if attr_name in stats_dict:  # Skip if already extracted
                        continue

                    if stat_label.lower() in row_text.lower():
                        values = re.findall(r"([\d.]+)%?", row_text)

                        if len(values) >= 2:
                            # Determine if we're dealing with integers or floats
                            if (
                                attr_name == "expected_goals"
                                or "." in values[0]
                                or "." in values[-1]
                            ):
                                home_value = float(values[0])
                                away_value = float(values[-1])
                                # Convert to int if needed for most stats
                                if attr_name != "expected_goals":
                                    home_value = int(home_value)
                                    away_value = int(away_value)
                            else:
                                home_value = int(values[0])
                                away_value = int(values[-1])

                            stats_dict[attr_name] = (home_value, away_value)
                            self.logger.debug(
                                f"Extracted {stat_label} from row",
                                extra={"home": home_value, "away": away_value},
                            )
                            break  # Found this stat, move to next row
            except Exception:
                pass  # Move to next row

        return stats_dict
