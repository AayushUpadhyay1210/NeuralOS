from __future__ import annotations

import pytest


@pytest.fixture()
def isolated_dirs(tmp_path, monkeypatch):
    """Point models/ and experiments/ at a temp dir so tests never touch real artifacts."""
    models = tmp_path / "models"
    exps = tmp_path / "experiments"
    monkeypatch.setenv("NEURALOS_MODELS_DIR", str(models))
    monkeypatch.setenv("NEURALOS_EXPERIMENTS_DIR", str(exps))
    return models, exps


@pytest.fixture()
def client(isolated_dirs):
    from fastapi.testclient import TestClient

    from neuralos.api import state
    from neuralos.api.app import create_app

    state.models.unload()
    with TestClient(create_app()) as c:
        yield c
    state.models.unload()
