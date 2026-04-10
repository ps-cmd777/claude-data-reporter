"""Helper utilities used across the claude-data-reporter pipeline."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import click


def validate_csv_path(path: str) -> Path:
    """Validate that path exists, is a file, and has a .csv extension.

    Raises click.BadParameter so the CLI surfaces a clean, formatted error.
    """
    p = Path(path)
    if not p.exists():
        raise click.BadParameter(f"File not found: {path}")
    if not p.is_file():
        raise click.BadParameter(f"Path is not a file: {path}")
    if p.suffix.lower() != ".csv":
        raise click.BadParameter(
            f"Expected a .csv file, got '{p.suffix}': {path}"
        )
    return p


def ensure_output_dir(output_dir: str | Path) -> Path:
    """Create the output directory (and parents) if it does not exist.

    Returns the resolved Path object.
    """
    p = Path(output_dir).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def format_timestamp() -> str:
    """Return a UTC timestamp string in YYYYMMDD_HHMMSS format."""
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")


def sanitize_filename(name: str) -> str:
    """Strip characters that are unsafe in filenames, replacing with underscores.

    Keeps alphanumerics, hyphens, and underscores.
    """
    return re.sub(r"[^\w\-]", "_", name)
