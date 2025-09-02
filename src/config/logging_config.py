"""
Advanced logging configuration with enterprise-grade features and best practices.
"""
import logging
import logging.handlers
import sys
import json
import traceback
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass, field
from threading import Lock
import os
import gzip
import shutil
from contextlib import contextmanager


class LogLevel(str, Enum):
    """Log levels with descriptive values."""
    TRACE = "TRACE"  # Custom ultra-verbose level
    DEBUG = "DEBUG"
    INFO = "INFO" 
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogColors(str, Enum):
    """ANSI color codes for terminal output."""
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    @classmethod
    def from_name(cls, name: str) -> "LogColors":
        """Get color from name with fallback."""
        color_map = {
            "black": cls.BLACK, "red": cls.RED, "green": cls.GREEN,
            "yellow": cls.YELLOW, "blue": cls.BLUE, "magenta": cls.MAGENTA,
            "cyan": cls.CYAN, "white": cls.WHITE,
            "bright_black": cls.BRIGHT_BLACK, "bright_red": cls.BRIGHT_RED,
            "bright_green": cls.BRIGHT_GREEN, "bright_yellow": cls.BRIGHT_YELLOW,
            "bright_blue": cls.BRIGHT_BLUE, "bright_magenta": cls.BRIGHT_MAGENTA,
            "bright_cyan": cls.BRIGHT_CYAN, "bright_white": cls.BRIGHT_WHITE
        }
        return color_map.get(name.lower(), cls.WHITE)


@dataclass
class LoggerConfig:
    """Configuration for logger instances."""
    name: str
    level: Union[str, int] = LogLevel.INFO
    color: str = "white"
    enable_console: bool = True
    enable_file: bool = False
    file_path: Optional[Union[str, Path]] = None
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    backup_count: int = 10
    colored_output: bool = True
    json_format: bool = False
    include_caller_info: bool = True
    include_process_info: bool = False
    include_thread_info: bool = False
    enable_compression: bool = True
    log_directory: Optional[Path] = None
    context: Dict[str, Any] = field(default_factory=dict)
    sanitize_sensitive_data: bool = True
    performance_tracking: bool = False


class SensitiveDataFilter(logging.Filter):
    """Filter to sanitize sensitive information from logs."""
    
    SENSITIVE_PATTERNS = [
        "password", "passwd", "secret", "token", "key", "auth",
        "credential", "private", "confidential", "ssn", "social",
        "credit_card", "cc_number", "api_key", "bearer"
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter and sanitize log record."""
        if hasattr(record, 'msg') and record.msg:
            msg = str(record.msg).lower()
            for pattern in self.SENSITIVE_PATTERNS:
                if pattern in msg:
                    record.msg = record.msg.replace(
                        record.msg[msg.find(pattern):msg.find(pattern)+20], 
                        "[REDACTED]"
                    )
        return True


class PerformanceFilter(logging.Filter):
    """Filter to add performance metrics to log records."""
    
    def __init__(self):
        super().__init__()
        self._start_time = datetime.now()
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add performance metrics."""
        current_time = datetime.now()
        record.elapsed = (current_time - self._start_time).total_seconds()
        record.timestamp_iso = current_time.isoformat()
        return True


class JSONFormatter(logging.Formatter):
    """Enhanced JSON formatter for structured logging."""
    
    def __init__(self, include_caller_info: bool = True, 
                 include_process_info: bool = False,
                 include_thread_info: bool = False):
        super().__init__()
        self.include_caller_info = include_caller_info
        self.include_process_info = include_process_info  
        self.include_thread_info = include_thread_info
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add caller information
        if self.include_caller_info and hasattr(record, 'pathname'):
            log_entry["file"] = {
                "path": record.pathname,
                "name": record.filename,
                "line": record.lineno
            }
        
        # Add process information
        if self.include_process_info:
            log_entry["process"] = {
                "id": record.process,
                "name": record.processName
            }
        
        # Add thread information
        if self.include_thread_info:
            log_entry["thread"] = {
                "id": record.thread,
                "name": record.threadName
            }
        
        # Add exception information
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra context
        if hasattr(record, 'extra_context'):
            log_entry.update(record.extra_context)
        
        # Add performance metrics if available
        if hasattr(record, 'elapsed'):
            log_entry["performance"] = {
                "elapsed_seconds": record.elapsed,
                "timestamp_iso": record.timestamp_iso
            }
        
        return json.dumps(log_entry, ensure_ascii=False, default=str)


class ColoredFormatter(logging.Formatter):
    """Enhanced colored formatter with emoji and better formatting."""
    
    LEVEL_COLORS = {
        logging.DEBUG: LogColors.BRIGHT_BLACK,
        logging.INFO: LogColors.BRIGHT_GREEN,
        logging.WARNING: LogColors.BRIGHT_YELLOW,
        logging.ERROR: LogColors.BRIGHT_RED,
        logging.CRITICAL: LogColors.BRIGHT_MAGENTA,
    }
    
    LEVEL_EMOJIS = {
        logging.DEBUG: "🔍",
        logging.INFO: "ℹ️",
        logging.WARNING: "⚠️",
        logging.ERROR: "❌",
        logging.CRITICAL: "🚨",
    }
    
    def __init__(self, logger_name: str, logger_color: LogColors,
                 colored: bool = True, include_emojis: bool = True,
                 include_caller_info: bool = True):
        super().__init__()
        self.logger_name = logger_name
        self.logger_color = logger_color
        self.colored = colored
        self.include_emojis = include_emojis
        self.include_caller_info = include_caller_info
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors and structure."""
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Build components
        if self.colored:
            timestamp_colored = f"{LogColors.DIM}{timestamp}{LogColors.RESET}"
            logger_colored = f"{self.logger_color}{LogColors.BOLD}{self.logger_name}{LogColors.RESET}"
            level_color = self.LEVEL_COLORS.get(record.levelno, LogColors.WHITE)
            level_colored = f"{level_color}{LogColors.BOLD}[{record.levelname}]{LogColors.RESET}"
        else:
            timestamp_colored = timestamp
            logger_colored = self.logger_name
            level_colored = f"[{record.levelname}]"
        
        # Add emoji if enabled
        emoji = self.LEVEL_EMOJIS.get(record.levelno, "") if self.include_emojis else ""
        emoji_part = f"{emoji} " if emoji else ""
        
        # Build main message
        message = record.getMessage()
        
        # Add caller info if enabled
        caller_info = ""
        if self.include_caller_info and self.colored:
            caller_info = f" {LogColors.DIM}({record.filename}:{record.lineno}){LogColors.RESET}"
        elif self.include_caller_info:
            caller_info = f" ({record.filename}:{record.lineno})"
        
        # Add extra context if present
        extra_info = ""
        if hasattr(record, 'extra_context') and record.extra_context:
            extra_json = json.dumps(record.extra_context, ensure_ascii=False, default=str)
            if self.colored:
                extra_info = f" {LogColors.CYAN}||| {extra_json}{LogColors.RESET}"
            else:
                extra_info = f" ||| {extra_json}"
        
        # Combine all parts
        log_line = f"{timestamp_colored} {logger_colored} {level_colored} {emoji_part}{message}{caller_info}{extra_info}"
        
        # Add exception traceback if present
        if record.exc_info:
            log_line += "\n" + self.formatException(record.exc_info)
        
        return log_line


class CompressedRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """File handler with compression support."""
    
    def __init__(self, *args, **kwargs):
        self.compress_logs = kwargs.pop('compress', True)
        super().__init__(*args, **kwargs)
    
    def doRollover(self):
        """Enhanced rollover with compression."""
        super().doRollover()
        
        if self.compress_logs and self.backupCount > 0:
            # Compress the most recent backup
            backup_name = f"{self.baseFilename}.1"
            if os.path.exists(backup_name):
                compressed_name = f"{backup_name}.gz"
                with open(backup_name, 'rb') as f_in:
                    with gzip.open(compressed_name, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(backup_name)


class AdvancedLogger:
    """Enterprise-grade logger with comprehensive features."""
    
    _instances: Dict[str, 'AdvancedLogger'] = {}
    _lock = Lock()
    
    def __init__(self, config: LoggerConfig):
        self.config = config
        self.logger = logging.getLogger(config.name)
        self.logger.setLevel(getattr(logging, config.level) if isinstance(config.level, str) else config.level)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
            self._setup_filters()
        
        # Create context manager for structured logging
        self._context_stack: List[Dict[str, Any]] = []
    
    @classmethod  
    def get_logger(cls, config: LoggerConfig) -> 'AdvancedLogger':
        """Get singleton logger instance."""
        with cls._lock:
            if config.name not in cls._instances:
                cls._instances[config.name] = cls(config)
            return cls._instances[config.name]
    
    def _setup_handlers(self):
        """Setup console and file handlers."""
        # Console handler
        if self.config.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            
            if self.config.json_format:
                console_formatter = JSONFormatter(
                    include_caller_info=self.config.include_caller_info,
                    include_process_info=self.config.include_process_info,
                    include_thread_info=self.config.include_thread_info
                )
            else:
                console_formatter = ColoredFormatter(
                    logger_name=self.config.name,
                    logger_color=LogColors.from_name(self.config.color),
                    colored=self.config.colored_output and sys.stdout.isatty(),
                    include_caller_info=self.config.include_caller_info
                )
            
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.config.enable_file and self.config.file_path:
            file_path = Path(self.config.file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if self.config.enable_compression:
                file_handler = CompressedRotatingFileHandler(
                    filename=file_path,
                    maxBytes=self.config.max_file_size,
                    backupCount=self.config.backup_count,
                    compress=True
                )
            else:
                file_handler = logging.handlers.RotatingFileHandler(
                    filename=file_path,
                    maxBytes=self.config.max_file_size,
                    backupCount=self.config.backup_count
                )
            
            # Always use JSON format for files
            file_formatter = JSONFormatter(
                include_caller_info=self.config.include_caller_info,
                include_process_info=self.config.include_process_info,
                include_thread_info=self.config.include_thread_info
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def _setup_filters(self):
        """Setup logging filters."""
        if self.config.sanitize_sensitive_data:
            sensitive_filter = SensitiveDataFilter()
            for handler in self.logger.handlers:
                handler.addFilter(sensitive_filter)
        
        if self.config.performance_tracking:
            perf_filter = PerformanceFilter()
            for handler in self.logger.handlers:
                handler.addFilter(perf_filter)
    
    @contextmanager
    def context(self, **context_data):
        """Context manager for structured logging."""
        self._context_stack.append(context_data)
        try:
            yield self
        finally:
            self._context_stack.pop()
    
    def _get_current_context(self) -> Dict[str, Any]:
        """Get merged context from stack and config."""
        context = dict(self.config.context)
        for ctx in self._context_stack:
            context.update(ctx)
        return context
    
    def _log(self, level: int, message: str, extra: Optional[Dict[str, Any]] = None, 
             exc_info: bool = False, **kwargs):
        """Internal logging method with context support."""
        current_context = self._get_current_context()
        if extra:
            current_context.update(extra)
        if kwargs:
            current_context.update(kwargs)
        
        # Create enhanced record
        self.logger.log(
            level, 
            message, 
            extra={"extra_context": current_context},
            exc_info=exc_info
        )
    
    def trace(self, message: str, **kwargs):
        """Ultra-verbose logging."""
        self._log(5, message, **kwargs)  # Below DEBUG
    
    def debug(self, message: str, **kwargs):
        """Debug logging."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Info logging."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Warning logging."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, exc_info: bool = True, **kwargs):
        """Error logging with optional exception info."""
        self._log(logging.ERROR, message, exc_info=exc_info, **kwargs)
    
    def critical(self, message: str, exc_info: bool = True, **kwargs):
        """Critical logging with optional exception info."""
        self._log(logging.CRITICAL, message, exc_info=exc_info, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        self._log(logging.ERROR, message, exc_info=True, **kwargs)
    
    def add_context(self, **context):
        """Add persistent context to logger."""
        self.config.context.update(context)
    
    def remove_context(self, *keys):
        """Remove context keys."""
        for key in keys:
            self.config.context.pop(key, None)
    
    def update_level(self, level: Union[str, int]):
        """Update logging level dynamically."""
        new_level = getattr(logging, level) if isinstance(level, str) else level
        self.logger.setLevel(new_level)
        self.config.level = level
    
    def get_child(self, suffix: str) -> 'AdvancedLogger':
        """Create child logger."""
        child_config = LoggerConfig(
            name=f"{self.config.name}.{suffix}",
            level=self.config.level,
            color=self.config.color,
            enable_console=self.config.enable_console,
            enable_file=self.config.enable_file,
            file_path=self.config.file_path,
            colored_output=self.config.colored_output,
            json_format=self.config.json_format,
            context=dict(self.config.context)
        )
        return AdvancedLogger.get_logger(child_config)


# Convenience functions for quick logger creation
def get_logger(name: str, **kwargs) -> AdvancedLogger:
    """Quick logger factory."""
    config = LoggerConfig(name=name, **kwargs)
    return AdvancedLogger.get_logger(config)


def get_file_logger(name: str, file_path: Union[str, Path], **kwargs) -> AdvancedLogger:
    """Create logger with file output."""
    config = LoggerConfig(
        name=name, 
        enable_file=True, 
        file_path=file_path,
        **kwargs
    )
    return AdvancedLogger.get_logger(config)


def get_json_logger(name: str, **kwargs) -> AdvancedLogger:
    """Create JSON logger."""
    config = LoggerConfig(name=name, json_format=True, **kwargs)
    return AdvancedLogger.get_logger(config)