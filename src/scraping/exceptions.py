from typing import Optional


class ScrapingError(Exception):
    """Base exception for all scraping-related errors."""

    pass


class ElementNotFoundError(ScrapingError):
    """Raised when an expected element is not found on the page."""

    def __init__(self, selector: str, message: Optional[str] = None):
        self.selector = selector
        self.message = message or f"Element not found with selector: {selector}"
        super().__init__(self.message)


class ParsingError(ScrapingError):
    """Raised when there's an error parsing scraped data."""

    def __init__(self, data_type: str, raw_data: str, message: Optional[str] = None):
        self.data_type = data_type
        self.raw_data = raw_data
        self.message = (
            message or f"Error parsing {data_type} from data: {raw_data[:100]}..."
        )
        super().__init__(self.message)


class RateLimitError(ScrapingError):
    """Raised when rate limiting is detected."""

    def __init__(self, retry_after: Optional[int] = None):
        self.retry_after = retry_after
        message = (
            f"Rate limit exceeded. Retry after {retry_after} seconds."
            if retry_after
            else "Rate limit exceeded."
        )
        super().__init__(message)


class StorageError(ScrapingError):
    """Raised when there's an error storing scraped data."""

    def __init__(self, file_path: str, operation: str, message: Optional[str] = None):
        self.file_path = file_path
        self.operation = operation
        self.message = (
            message or f"Error during {operation} operation on file: {file_path}"
        )
        super().__init__(self.message)


class NetworkError(ScrapingError):
    """Raised when there's a network-related error."""

    def __init__(self, url: str, status_code: Optional[int] = None):
        self.url = url
        self.status_code = status_code
        message = f"Network error accessing {url}"
        if status_code:
            message += f" (Status code: {status_code})"
        super().__init__(message)
