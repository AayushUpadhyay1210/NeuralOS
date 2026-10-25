"""Checkpoint registry: training runs stored under ``models/<run_id>/``.

Layout of a run directory::

    metadata.json   configuration, status, versions, best validation score
    progress.json   live training progress (reward curve, validation history)
    train.log       human-readable log
    best.zip        checkpoint with the best validation objective
    final.zip       checkpoint at the end of training
    CANCEL          present if cancellation was requested

This module deliberately imports no ML libraries.
"""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ..paths import models_dir, safe_id

RL_PACKAGES = ("torch", "gymnasium", "stable_baselines3", "sb3_contrib")


def missing_rl_dependencies() -> list[str]:
    return [p for p in RL_PACKAGES if importlib.util.find_spec(p) is None]


def run_dir(run_id: str) -> Path:
    return models_dir() / safe_id(run_id)


def write_json(path: Path, data: Any) -> None:
    """Atomic write so readers never see half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def metadata(run_id: str) -> dict[str, Any] | None:
    data = read_json(run_dir(run_id) / "metadata.json")
    return data if isinstance(data, dict) else None


def progress(run_id: str) -> dict[str, Any] | None:
    data = read_json(run_dir(run_id) / "progress.json")
    return data if isinstance(data, dict) else None


def checkpoint_path(run_id: str, which: str = "best") -> Path | None:
    if which not in ("best", "final"):
        raise ValueError("checkpoint must be 'best' or 'final'")
    p = run_dir(run_id) / f"{which}.zip"
    return p if p.exists() else None


def list_runs() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for d in sorted(models_dir().iterdir()):
        if not d.is_dir():
            continue
        meta = read_json(d / "metadata.json")
        if not isinstance(meta, dict):
            continue
        meta["has_best"] = (d / "best.zip").exists()
        meta["has_final"] = (d / "final.zip").exists()
        out.append(meta)
    out.sort(key=lambda m: str(m.get("created_at", "")), reverse=True)
    return out


def request_cancel(run_id: str) -> None:
    (run_dir(run_id) / "CANCEL").touch()


def cancel_requested(run_id: str) -> bool:
    return (run_dir(run_id) / "CANCEL").exists()
