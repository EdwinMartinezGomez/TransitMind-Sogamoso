"""
TransitMind Sogamoso — Centralized Logger
==========================================
Structured logging with structlog for consistent log output across all layers.
Supports both JSON and console-friendly formats.
"""

import logging
import sys
from typing import Optional

import structlog


def setup_logger(
    name: str = "transitmind",
    level: str = "INFO",
    json_format: bool = False,
) -> structlog.stdlib.BoundLogger:
    """
    Configure and return a structured logger instance.

    Args:
        name: Logger name (typically module or layer name).
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_format: If True, output logs as JSON. Otherwise, use console format.

    Returns:
        A configured structlog BoundLogger instance.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Choose processors based on format
    if json_format:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(
            colors=sys.stdout.isatty(),
        )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            renderer,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    logger = structlog.get_logger(name)
    return logger


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """
    Get a logger instance. Uses module name if not specified.

    Args:
        name: Optional logger name. Defaults to 'transitmind'.

    Returns:
        A structlog BoundLogger instance.
    """
    return structlog.get_logger(name or "transitmind")
