"""Gymnasium environment wrapping the shared simulation engine."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..sim.engine import SimConfig, Simulator
from ..sim.metrics import compute_metrics
from ..sim.process import ProcessSpec
from ..sim.workload import WorkloadConfig, WorkloadKind, generate
from .spaces import (
    OBS_HIGH,
    ObservationBuilder,
    RewardConfig,
    SpaceConfig,
    cumulative_cost,
    reward_normaliser,
)

# Disjoint seed ranges keep training, validation and test workloads apart.
TRAIN_SEED_RANGE = (0, 4_999_999)
VALIDATION_SEED_BASE = 5_000_000
TEST_SEED_BASE = 10_000_000

TRAIN_KINDS: tuple[WorkloadKind, ...] = (
    "cpu_bound",
    "io_bound",
    "short_jobs",
    "mixed_priority",
    "bursty",
)
# Never seen during training; used for out-of-distribution evaluation.
HELD_OUT_KINDS: tuple[WorkloadKind, ...] = ("long_jobs", "mixed")


@dataclass(frozen=True)
class TrainDistribution:
    kinds: tuple[WorkloadKind, ...] = TRAIN_KINDS
    n_min: int = 10
    n_max: int = 40
    context_switch: tuple[int, ...] = (0, 1)

    def __post_init__(self) -> None:
        if not self.kinds:
            raise ValueError("need at least one workload kind")
        if not 1 <= self.n_min <= self.n_max <= 200:
            raise ValueError("need 1 <= n_min <= n_max <= 200")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kinds": list(self.kinds),
            "n_min": self.n_min,
            "n_max": self.n_max,
            "context_switch": list(self.context_switch),
        }


@dataclass
class EnvSettings:
    space: SpaceConfig = field(default_factory=SpaceConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    train: TrainDistribution = field(default_factory=TrainDistribution)


class SchedulingEnv(gym.Env[np.ndarray, int]):
    """One episode = one complete workload, from first arrival to last completion.

    * ``reset`` samples a training workload (or takes the next one from a fixed
      list) and advances the engine to the first decision point.
    * ``step(action)`` dispatches the chosen ready process for the chosen
      quantum and runs the engine to the next decision point (idle gaps are
      skipped automatically, so there is always >= 1 ready process when an
      action is requested).
    * ``terminated`` is True once every process has finished.
    * ``truncated`` is a safety net that should never fire: each step runs a
      process for >= 1 tick, so the episode needs at most total-CPU steps.
    * ``action_masks()`` is used by sb3-contrib's MaskablePPO. If an invalid
      action is still passed in, the env raises instead of silently remapping.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        settings: EnvSettings | None = None,
        workloads: Sequence[tuple[Sequence[ProcessSpec], SimConfig]] | None = None,
    ):
        super().__init__()
        self.settings = settings or EnvSettings()
        self.builder = ObservationBuilder(self.settings.space)
        sp = self.settings.space
        self.observation_space = spaces.Box(0.0, OBS_HIGH, shape=(sp.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(sp.n_actions)  # type: ignore[assignment]
        self._fixed = list(workloads) if workloads is not None else None
        self._fixed_idx = 0
        self.sim: Simulator | None = None
        self._cost = 0.0
        self._norm = 1.0
        self._steps = 0
        self._max_steps = 0
        self.last_workload: dict[str, Any] = {}

    # ----------------------------------------------------------------- helpers
    def _sample(self) -> tuple[list[ProcessSpec], SimConfig, dict[str, Any]]:
        if self._fixed is not None:
            specs, cfg = self._fixed[self._fixed_idx % len(self._fixed)]
            self._fixed_idx += 1
            return list(specs), cfg, {"fixed_index": self._fixed_idx - 1}
        td = self.settings.train
        rng = self.np_random
        kind = td.kinds[int(rng.integers(len(td.kinds)))]
        n = int(rng.integers(td.n_min, td.n_max + 1))
        seed = int(rng.integers(TRAIN_SEED_RANGE[0], TRAIN_SEED_RANGE[1] + 1))
        cs = td.context_switch[int(rng.integers(len(td.context_switch)))]
        specs = generate(WorkloadConfig(kind=kind, n_processes=n, seed=seed))
        return specs, SimConfig(context_switch=cs), {"kind": kind, "n": n, "seed": seed, "cs": cs}

    def _obs(self) -> np.ndarray:
        assert self.sim is not None
        if self.sim.done:
            return np.zeros(self.settings.space.obs_dim, dtype=np.float32)
        return self.builder.build(self.sim)

    # --------------------------------------------------------------------- api
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        specs, cfg, info = self._sample()
        self.sim = Simulator(specs, cfg)
        self.last_workload = info
        self._cost = cumulative_cost(self.sim, self.settings.reward)
        self._norm = reward_normaliser(self.sim)
        self._steps = 0
        self._max_steps = sum(s.total_cpu for s in specs) + 10
        return self._obs(), {"workload": info}

    def action_masks(self) -> np.ndarray:
        assert self.sim is not None
        if self.sim.done:
            m = np.zeros(self.settings.space.n_actions, dtype=bool)
            m[0] = True
            return m
        return self.builder.mask(self.sim)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.sim is None:
            raise RuntimeError("call reset() before step()")
        if self.sim.done:
            raise RuntimeError("episode finished; call reset()")
        pid, quantum = self.builder.decode(self.sim, int(action))
        self.sim.dispatch(pid, quantum)
        self._steps += 1
        cost = cumulative_cost(self.sim, self.settings.reward)
        reward = -(cost - self._cost) / self._norm
        self._cost = cost
        terminated = self.sim.done
        truncated = (not terminated) and self._steps >= self._max_steps
        info: dict[str, Any] = {}
        if terminated:
            m = compute_metrics(self.sim)
            info["metrics"] = m.to_dict()
            info["episode_cost"] = cost / self._norm
        return self._obs(), float(reward), terminated, truncated, info
