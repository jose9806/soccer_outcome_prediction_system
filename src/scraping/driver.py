from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from typing import Optional
import logging


class WebDriverFactory:
    """Factory class for creating and configuring WebDriver instances."""

    @staticmethod
    def create_driver(
        headless: bool = True,
        user_agent: Optional[str] = None,
        proxy: Optional[str] = None,
    ) -> webdriver.Chrome:
        """
        Create and configure a Chrome WebDriver instance.

        Args:
            headless: Whether to run in headless mode
            user_agent: Custom user agent string
            proxy: Proxy server address

        Returns:
            Configured Chrome WebDriver instance
        """
        options = Options()

        if headless:
            options.add_argument("--headless")

        # Basic options for stability
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")

        # Set custom user agent if provided
        if user_agent:
            options.add_argument(f"user-agent={user_agent}")

        # Set proxy if provided
        if proxy:
            options.add_argument(f"--proxy-server={proxy}")

        # Additional options for better performance
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-notifications")
        options.add_argument("--disable-infobars")

        try:
            driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()), options=options
            )

            # Set page load timeout
            driver.set_page_load_timeout(30)

            return driver

        except Exception as e:
            logging.error(f"Failed to create WebDriver: {str(e)}")
            raise

    @staticmethod
    def quit_driver(driver: webdriver.Chrome) -> None:
        """Safely quit a WebDriver instance."""
        try:
            if driver:
                driver.quit()
        except Exception as e:
            logging.error(f"Error while quitting WebDriver: {str(e)}")
