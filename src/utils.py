"""Shared path and logging utilities for TRACE."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional


def project_root() -> Path:
    """Return the TRACE repository root."""
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str, *, base_dir: Optional[str] = None) -> str:
    """Expand a path and resolve relative paths against ``base_dir``."""
    expanded = os.path.expandvars(os.path.expanduser(str(path)))
    resolved = Path(expanded)
    if not resolved.is_absolute() and base_dir:
        resolved = Path(base_dir) / resolved
    return str(resolved)


def resolve_data_dir(data_dir: str, *, env_var: str = "GENAR_DATA_DIR") -> str:
    """Resolve a data directory, optionally overridden by an environment variable."""
    override = os.environ.get(env_var)
    selected = override if override and override.strip() else data_dir
    if not selected or not str(selected).strip():
        raise ValueError("A data directory must be provided")
    return resolve_path(str(selected), base_dir=str(project_root()))


def setup_logging(config: Optional[dict] = None) -> logging.Logger:
    """Configure the shared TRACE logger for console and file output."""
    config = config or {}
    logger = logging.getLogger("trace")
    if logger.handlers:
        return logger

    level_name = str(config.get("log_level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    log_dir = resolve_path(
        str(config.get("log_dir", "logs")), base_dir=str(project_root())
    )
    os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(
        os.path.join(log_dir, "trace.log"), mode="a", encoding="utf-8"
    )
    console_handler = logging.StreamHandler(sys.stdout)
    for handler in (file_handler, console_handler):
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


__all__ = [
    "project_root",
    "resolve_path",
    "resolve_data_dir",
    "setup_logging",
]
