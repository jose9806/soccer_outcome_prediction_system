from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from src.config.logging_config import get_logger


class BaseScraper:
    """Base class for all scrapers with common functionality."""

    def __init__(self, driver, config):
        self.driver = driver
        self.config = config
        # Default logger will be overridden by child classes
        self.logger = get_logger(name="BaseScraper", color="white")

    def wait_for_element(self, selector, timeout=None):
        """
        Wait for an element to be present in the DOM and return it.

        Args:
            selector: CSS selector of the element to wait for
            timeout: Maximum time to wait in seconds

        Returns:
            WebElement when found

        Raises:
            TimeoutException if element is not found within timeout
        """
        if timeout is None:
            timeout = self.config.ELEMENT_WAIT_TIMEOUT

        return WebDriverWait(self.driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
        )

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
