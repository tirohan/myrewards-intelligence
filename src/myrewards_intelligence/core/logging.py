"""Logging configuration."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from .config import resolve_path


def setup_logging(
    log_level: str = "INFO",
    log_dir: str | Path = "logs",
    log_name: str = "milestone5",
) -> logging.Logger:
    """Configure logging with file and console handlers."""
    log_dir_path = resolve_path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_file = log_dir_path / f"{log_name}_{ts}.log"

    logger = logging.getLogger("myrewards_intelligence")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info("Logging initialized. Log file: %s", log_file)
    return logger


def utc_now_iso() -> str:
    """Return current UTC time in ISO format."""
    return datetime.now(UTC).isoformat()
