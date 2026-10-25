"""Use a trained checkpoint as a :class:`~neuralos.sim.schedulers.Scheduler`."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..sim.engine import Simulator
from ..sim.process import Process
from ..sim.schedulers import Decision, Scheduler
from .spaces import ObservationBuilder, SpaceConfig


class RLScheduler(Scheduler):
    """Deterministic (argmax) inference with the same masks used in training."""

    name = "rl"

    def __init__(self, model: Any, space: SpaceConfig, model_id: str = "unknown"):
        self.model = model
        self.builder = ObservationBuilder(space)
        self.model_id = model_id
        self._sim: Simulator | None = None
        expected = space.obs_dim
        got = model.observation_space.shape[0]
        if got != expected:
            raise ValueError(
                f"checkpoint expects observations of size {got}, env config gives {expected}"
            )

    def attach(self, sim: Simulator) -> None:
        self._sim = sim

    def select(self, ready: Sequence[Process], now: int) -> Decision:
        if self._sim is None:
            raise RuntimeError("RLScheduler must be attached to a simulator (use sim.runner.run)")
        obs = self.builder.build(self._sim)
        mask = self.builder.mask(self._sim)
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        pid, quantum = self.builder.decode(self._sim, int(np.asarray(action).item()))
        return Decision(pid, quantum)

    def describe(self) -> dict[str, object]:
        return {"name": self.name, "preemptive": False, "model_id": self.model_id}


def load_model(path: str | Path) -> Any:
    from sb3_contrib import MaskablePPO

    return MaskablePPO.load(str(path), device="cpu")
