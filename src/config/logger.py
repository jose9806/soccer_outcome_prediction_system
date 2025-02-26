import logging
import sys
import json
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
from pathlib import Path


class LogColors(str, Enum):
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    RESET = "\033[0m"

    @classmethod
    def from_name(cls, name: str) -> "LogColors":
        color_map = {
            "black": cls.BLACK,
            "red": cls.RED,
            "green": cls.GREEN,
            "yellow": cls.YELLOW,
            "blue": cls.BLUE,
            "magenta": cls.MAGENTA,
            "cyan": cls.CYAN,
            "white": cls.WHITE,
        }
        return color_map.get(name.lower(), cls.WHITE)


class LogLevel(Enum):
    DEBUG = (logging.DEBUG, LogColors.CYAN)
    INFO = (logging.INFO, LogColors.GREEN)
    WARNING = (logging.WARNING, LogColors.YELLOW)
    ERROR = (logging.ERROR, LogColors.RED)
    CRITICAL = (logging.CRITICAL, LogColors.MAGENTA)


class ColoredFormatter(logging.Formatter):
    """
    Formatter that adds colored output for console logging.
    Supports optional JSON formatting.
    """

    def __init__(
        self,
        project_name: str,
        project_color: LogColors,
        colored: bool = True,
        json_format: bool = False,
    ):
        super().__init__()
        self.project_name = project_name
        self.project_color = project_color
        self.colored = colored
        self.json_format = json_format
        self.level_colors = {level.value[0]: level.value[1] for level in LogLevel}

    def format(self, record: logging.LogRecord) -> str:
        # This calls the parent class to generate the base message (time, level, etc.)
        original_message = super().format(record)

        # Extract the extra_context dict that your LoggerAdapter injected
        extra_context = getattr(record, "extra_context", {})

        # Build your final string
        timestamp = datetime.fromtimestamp(record.created).strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )[:-3]

        if self.colored:
            project_prefix = f"{self.project_color.value}({timestamp} {self.project_name}){LogColors.RESET.value}"
            level_color = self.level_colors.get(record.levelno, LogColors.WHITE).value
            level_str = f"{level_color}[{record.levelname}]{LogColors.RESET.value}"
        else:
            project_prefix = f"({timestamp} {self.project_name})"
            level_str = f"[{record.levelname}]"

        # If you want to display all extra fields in JSON form:
        if extra_context:
            # e.g., append them as JSON after the message
            extra_json = json.dumps(extra_context, ensure_ascii=False)
            original_message = f"{original_message} ||| {extra_json}"

        return f"{project_prefix} {level_str} {original_message}"


class ContextLoggerAdapter(logging.LoggerAdapter):
    """
    LoggerAdapter to automatically inject extra context.
    """

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        extra = kwargs.setdefault("extra", {})
        extra.update(self.extra)
        # Allow extra context to be merged into the record.
        kwargs["extra"] = {"extra_context": extra}
        return msg, kwargs


class Logger:
    """
    A scalable custom logger supporting:
      - Colored console output with an enhanced formatter.
      - Rotating file logging.
      - Structured logging with extra context.
      - Dynamic runtime configuration.
    """

    def __init__(
        self,
        name: str,
        color: str = "white",
        level: int = logging.INFO,
        file_output: Optional[str] | Path = None,
        enable_console: bool = True,
        colored: bool = True,
        json_format: bool = False,
        extra_context: Optional[Dict[str, Any]] = None,
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.extra_context = extra_context or {}

        if not self.logger.handlers:
            if enable_console:
                console_handler = logging.StreamHandler(sys.stdout)
                console_formatter = ColoredFormatter(
                    project_name=name,
                    project_color=LogColors.from_name(color),
                    colored=colored,
                    json_format=json_format,
                )
                console_handler.setFormatter(console_formatter)
                self.logger.addHandler(console_handler)
            if file_output:
                from logging.handlers import RotatingFileHandler

                file_handler = RotatingFileHandler(
                    file_output, maxBytes=10 * 1024 * 1024, backupCount=5
                )
                file_formatter = logging.Formatter(
                    "%(asctime)s.%(msecs)03d %(name)s [%(levelname)s]: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)
        # Wrap with an adapter to inject extra context automatically.
        self.adapter = ContextLoggerAdapter(self.logger, self.extra_context)

    def debug(self, message: str, **kwargs) -> None:
        self.adapter.debug(message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        self.adapter.info(message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        self.adapter.warning(message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        self.adapter.error(message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        self.adapter.critical(message, **kwargs)

    def add_context(self, **context: Any) -> None:
        """
        Add extra context that will be attached to every log message.
        """
        temp = dict(self.adapter.extra) if self.adapter.extra is not None else {}
        temp.update(context)
        self.adapter.extra = temp

    def update_config(
        self, level: Optional[int] = None, colored: Optional[bool] = None
    ) -> None:
        """
        Dynamically update logger configuration at runtime.
        """
        if level is not None:
            self.logger.setLevel(level)
        if colored is not None:
            for handler in self.logger.handlers:
                if isinstance(handler.formatter, ColoredFormatter):
                    handler.formatter.colored = colored
