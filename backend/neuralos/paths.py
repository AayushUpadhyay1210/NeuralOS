"""Filesystem locations. Override with NEURALOS_MODELS_DIR / NEURALOS_EXPERIMENTS_DIR."""

from __future__ import annotations

import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def models_dir() -> Path:
    p = Path(os.environ.get("NEURALOS_MODELS_DIR", REPO_ROOT / "models"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def experiments_dir() -> Path:
    p = Path(os.environ.get("NEURALOS_EXPERIMENTS_DIR", REPO_ROOT / "experiments" / "results"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_id(value: str) -> str:
    """Validate an id used as a directory name (blocks path traversal)."""
    if not _ID_RE.match(value) or ".." in value:
        raise ValueError(
            "ids may only contain letters, digits, '_', '-', '.', and must be 1-64 chars"
        )
    return value
