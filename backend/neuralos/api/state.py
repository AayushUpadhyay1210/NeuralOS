"""In-process state: simulation cache, loaded models, training jobs."""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..rl import registry
from ..rl.loader import ModelUnavailable
from ..sim.schedulers import Scheduler

log = logging.getLogger("neuralos.api")
STALE_AFTER_S = 120.0


class SimulationStore:
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self._data: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, payload: dict[str, Any]) -> str:
        sid = uuid.uuid4().hex[:12]
        with self._lock:
            self._data[sid] = payload
            while len(self._data) > self.capacity:
                self._data.popitem(last=False)
        return sid

    def get(self, sid: str) -> dict[str, Any] | None:
        with self._lock:
            return self._data.get(sid)


@dataclass
class LoadedModel:
    run_id: str
    checkpoint: str
    model: Any
    space: Any
    loaded_at: str


class ModelManager:
    """Caches loaded checkpoints; tracks which one is 'active' for the UI."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], LoadedModel] = {}
        self.active: tuple[str, str] | None = None
        self._lock = threading.Lock()

    def load(self, run_id: str, checkpoint: str = "best") -> LoadedModel:
        key = (run_id, checkpoint)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        from ..rl.loader import load_scheduler

        sched = load_scheduler(run_id, checkpoint)  # raises ModelUnavailable
        lm = LoadedModel(
            run_id=run_id,
            checkpoint=checkpoint,
            model=sched.model,  # type: ignore[attr-defined]
            space=sched.builder.space,  # type: ignore[attr-defined]
            loaded_at=datetime.now().isoformat(timespec="seconds"),
        )
        with self._lock:
            self._cache[key] = lm
        return lm

    def activate(self, run_id: str, checkpoint: str) -> LoadedModel:
        lm = self.load(run_id, checkpoint)
        self.active = (run_id, checkpoint)
        return lm

    def unload(self) -> None:
        with self._lock:
            self._cache.clear()
            self.active = None

    def scheduler(self, run_id: str | None, checkpoint: str = "best") -> Scheduler:
        from ..rl.policy import RLScheduler

        if run_id is None:
            if self.active is None:
                raise ModelUnavailable(
                    "No trained model is loaded. Train one or load a checkpoint on the "
                    "AI Training page first."
                )
            run_id, checkpoint = self.active
        lm = self.load(run_id, checkpoint)
        return RLScheduler(lm.model, lm.space, model_id=f"{run_id}:{checkpoint}")

    def active_info(self) -> dict[str, Any] | None:
        if self.active is None:
            return None
        lm = self._cache.get(self.active)
        if lm is None:
            return None
        meta = registry.metadata(lm.run_id) or {}
        return {
            "run_id": lm.run_id,
            "checkpoint": lm.checkpoint,
            "loaded_at": lm.loaded_at,
            "algorithm": meta.get("algorithm"),
            "seed": meta.get("seed"),
            "timesteps": meta.get("timesteps_done"),
            "status": meta.get("status"),
            "best": meta.get("best"),
        }


def _train_entry(cfg: dict[str, Any]) -> None:  # runs in a child process
    import logging as _logging

    from ..rl.spaces import RewardConfig
    from ..rl.train import TrainConfig, train

    _logging.basicConfig(level=_logging.INFO)
    reward = RewardConfig(**cfg.pop("reward", {}))
    train(TrainConfig(reward=reward, **cfg))


class TrainingManager:
    """At most one training process at a time, launched with the 'spawn' context."""

    def __init__(self) -> None:
        self._proc: mp.process.BaseProcess | None = None
        self._run_id: str | None = None
        self._lock = threading.Lock()

    def current(self) -> str | None:
        with self._lock:
            if self._proc is not None and self._proc.is_alive():
                return self._run_id
            return None

    def start(self, cfg: dict[str, Any]) -> str:
        missing = registry.missing_rl_dependencies()
        if missing:
            raise ModelUnavailable(
                "Cannot train: missing RL dependencies " + ", ".join(missing)
                + ". Install with: pip install -r backend/requirements-rl.txt"
            )
        run_id = str(cfg["run_id"])
        with self._lock:
            if self._proc is not None and self._proc.is_alive():
                raise RuntimeError(f"training run '{self._run_id}' is already in progress")
            if registry.metadata(run_id) is not None:
                raise FileExistsError(f"a model named '{run_id}' already exists")
            ctx = mp.get_context("spawn")
            proc = ctx.Process(
                target=_train_entry, args=(cfg,), daemon=True, name=f"train-{run_id}"
            )
            proc.start()
            self._proc, self._run_id = proc, run_id
        log.info("started training process pid=%s for %s", proc.pid, run_id)
        return run_id

    def status(self, run_id: str) -> dict[str, Any]:
        meta = registry.metadata(run_id)
        prog = registry.progress(run_id)
        with self._lock:
            tracked = self._run_id == run_id and self._proc is not None
            alive = tracked and self._proc is not None and self._proc.is_alive()
            exitcode = self._proc.exitcode if tracked and self._proc is not None else None
        if meta is None and not alive:
            if tracked:
                return {"run_id": run_id, "state": "failed", "detail":
                        f"training process exited with code {exitcode} before writing metadata"}
            return {"run_id": run_id, "state": "unknown"}
        state = (meta or {}).get("status", "starting")
        if state == "running" and not alive:
            age = time.time() - (registry.run_dir(run_id) / "progress.json").stat().st_mtime \
                if (registry.run_dir(run_id) / "progress.json").exists() else 1e9
            if tracked:
                state = "failed"
            elif age > STALE_AFTER_S:
                state = "interrupted"  # e.g. API restarted mid-run
        return {
            "run_id": run_id,
            "state": state,
            "managed_by_api": tracked,
            "exitcode": exitcode,
            "error": (meta or {}).get("error"),
            "progress": prog,
        }

    def cancel(self, run_id: str) -> None:
        registry.request_cancel(run_id)

    def any_running(self) -> list[str]:
        out = []
        for meta in registry.list_runs():
            rid = str(meta["run_id"])
            if self.status(rid)["state"] == "running":
                out.append(rid)
        return out


simulations = SimulationStore()
models = ModelManager()
training = TrainingManager()
