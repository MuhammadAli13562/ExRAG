"""
Centralized logging configuration for the ExRAG pipeline.
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from functools import lru_cache


def setup_logging(
    name: str = "exrag",
    level: str = "INFO",
    log_file: Optional[Path] = None,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """
    Setup structured logging for the pipeline.
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logging
        log_format: Optional custom format string
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Set level
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Default format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


@lru_cache
def get_logger(name: str = "exrag") -> logging.Logger:
    """
    Get or create a cached logger.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    # Try to get settings for configuration
    try:
        from .settings import get_settings
        settings = get_settings()
        return setup_logging(
            name=name,
            level=settings.log_level,
            log_file=settings.log_file if settings.log_to_file else None,
            log_format=settings.log_format,
        )
    except Exception:
        # Fallback to simple logging if settings not available
        return setup_logging(name=name)


def get_child_logger(parent_name: str, child_name: str) -> logging.Logger:
    """
    Get a child logger for a specific module.
    
    Args:
        parent_name: Parent logger name
        child_name: Child logger name
    
    Returns:
        Child logger instance
    """
    return logging.getLogger(f"{parent_name}.{child_name}")

