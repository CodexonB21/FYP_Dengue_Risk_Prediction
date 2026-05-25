"""Shared utility functions for the FYP project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT


def ensure_dir(path: Path) -> Path:
    """Create directory if missing and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Load a CSV with a clear error if the file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found: {path}")
    return pd.read_csv(path, **kwargs)


def save_csv(df: pd.DataFrame, path: Path, **kwargs) -> None:
    """Save a DataFrame to CSV, creating parent directories as needed."""
    ensure_dir(path.parent)
    df.to_csv(path, index=False, **kwargs)


def project_path(*parts: str) -> Path:
    """Build a path relative to the project root."""
    return PROJECT_ROOT.joinpath(*parts)
