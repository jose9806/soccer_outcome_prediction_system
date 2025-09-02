"""
Legacy logger module - DEPRECATED
Use logging_config.py for new implementations.

This module provides backward compatibility for existing code.
"""
import warnings
from typing import Optional, Dict, Any, Union
from pathlib import Path

from .logging_config import (
    AdvancedLogger, 
    LoggerConfig, 
    get_logger as _get_logger,
    LogLevel,
    LogColors
)

# Issue deprecation warning
warnings.warn(
    "logger.py is deprecated. Use logging_config.py instead.", 
    DeprecationWarning, 
    stacklevel=2
)


class Logger:
    """
    Backward compatibility wrapper for the old Logger class.
    Delegates to the new AdvancedLogger implementation.
    """
    
    def __init__(
        self,
        name: str,
        color: str = "white",
        level: Union[str, int] = "INFO",
        file_output: Optional[Union[str, Path]] = None,
        enable_console: bool = True,
        colored: bool = True,
        json_format: bool = False,
        extra_context: Optional[Dict[str, Any]] = None,
    ):
        # Map old parameters to new config
        config = LoggerConfig(
            name=name,
            level=level,
            color=color,
            enable_console=enable_console,
            enable_file=bool(file_output),
            file_path=file_output,
            colored_output=colored,
            json_format=json_format,
            context=extra_context or {}
        )
        
        # Create the advanced logger instance
        self._advanced_logger = AdvancedLogger.get_logger(config)
    
    def debug(self, message: str, **kwargs) -> None:
        """Debug logging with extra context support."""
        extra = kwargs.pop('extra', {})
        self._advanced_logger.debug(message, **extra, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Info logging with extra context support.""" 
        extra = kwargs.pop('extra', {})
        self._advanced_logger.info(message, **extra, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Warning logging with extra context support."""
        extra = kwargs.pop('extra', {})
        self._advanced_logger.warning(message, **extra, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Error logging with extra context support."""
        extra = kwargs.pop('extra', {})
        self._advanced_logger.error(message, exc_info=False, **extra, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Critical logging with extra context support."""
        extra = kwargs.pop('extra', {})
        self._advanced_logger.critical(message, exc_info=False, **extra, **kwargs)
    
    def add_context(self, **context: Any) -> None:
        """Add extra context that will be attached to every log message."""
        self._advanced_logger.add_context(**context)
    
    def update_config(
        self, 
        level: Optional[Union[str, int]] = None, 
        colored: Optional[bool] = None
    ) -> None:
        """Dynamically update logger configuration at runtime."""
        if level is not None:
            self._advanced_logger.update_level(level)
        
        # Note: colored output update is handled automatically in new system
        if colored is not None:
            self._advanced_logger.config.colored_output = colored


# Export legacy components for backward compatibility
__all__ = [
    'Logger',
    'LogLevel', 
    'LogColors',
    'AdvancedLogger',
    'LoggerConfig',
    'get_logger'
]

# Convenience function that matches old interface
def get_logger(name: str, **kwargs) -> AdvancedLogger:
    """Get logger using new advanced system."""
    return _get_logger(name, **kwargs)