"""Structured JSONL logging via structlog (spec §7.5)."""
from __future__ import annotations

import sys

import structlog

_configured = False


def configure_logging(stream=None, *, force: bool = False) -> None:
    """Configure structlog to emit one JSON object per line.

    Idempotent by default: safe to call multiple times in normal use. Tests
    may pass ``force=True`` to reset structlog state and reconfigure with a
    custom ``stream`` (e.g. an io.StringIO) regardless of prior calls.
    """
    global _configured
    if _configured and not force:
        return
    structlog.reset_defaults()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO+
        logger_factory=structlog.PrintLoggerFactory(file=stream or sys.stderr),
        cache_logger_on_first_use=False,
    )
    _configured = True


def get_logger(name: str | None = None):
    configure_logging()
    return structlog.get_logger(name)
