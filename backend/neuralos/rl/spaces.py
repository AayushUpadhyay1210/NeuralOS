"""Observation, action and reward definitions shared by the env and the policy scheduler.

This module only needs numpy, so it can be imported (and tested) without
gymnasium or stable-baselines3 installed.

Observation (float32 vector, every entry clipped to [0, OBS_HIGH])
-------------------------------------------------------------------
``K`` slots describe the oldest ``K`` ready processes in ready-queue order
(FIFO by ``enqueue_seq``). Showing the *oldest* entries guarantees that a
long-waiting process is always visible to the agent, so starvation is never
caused by the process being hidden. Times are divided by the workload's mean
CPU burst ``S`` and squashed with ``log1p`` so heavy-tailed workloads stay in
range.

Per slot (``SLOT_FEATURES`` = 10):
    0 present flag
    1 log1p(remaining time in current CPU burst / S)
    2 log1p(total remaining CPU work / S)
    3 log1p(current uninterrupted wait / S)
    4 log1p(total waiting so far / S)
    5 log1p(time since arrival / S)
    6 priority / max priority in workload (0 = highest priority)
    7 never-run flag (1 until first dispatch, i.e. response still pending)
    8 CPU bursts left / 5 (clipped to 1) - signals I/O-bound processes
    9 is-last-run flag (dispatching it costs no context switch)

Global (``GLOBAL_FEATURES`` = 9):
    0 ready count / K
    1 processes beyond the K visible slots / K
    2 blocked fraction, 3 not-yet-arrived fraction, 4 terminated fraction
    5 log1p(time to next arrival or I/O completion / S)
    6 context-switch cost / S
    7 log1p(mean remaining burst of hidden ready processes / S)
    8 log1p(max current wait among hidden ready processes / S)

Action
------
``Discrete(K * len(quanta))``: ``action = slot * len(quanta) + q`` dispatches
the process in ``slot`` for ``quanta[q]`` ticks (``None`` = until its CPU
burst ends). Actions pointing at empty slots are masked out; the env only
asks for an action when at least one process is ready.

Reward
------
See :class:`RewardConfig`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from ..sim.engine import Simulator
from ..sim.process import Process

SLOT_FEATURES = 10
GLOBAL_FEATURES = 9
OBS_HIGH = 10.0


@dataclass(frozen=True)
class SpaceConfig:
    slots: int = 10
    quanta: tuple[int | None, ...] = (2, 6, None)

    def __post_init__(self) -> None:
        if not 1 <= self.slots <= 64:
            raise ValueError("slots must be in [1, 64]")
        if not self.quanta:
            raise ValueError("at least one quantum option is required")
        if any(q is not None and q < 1 for q in self.quanta):
            raise ValueError("quanta must be >= 1 or None")

    @property
    def obs_dim(self) -> int:
        return self.slots * SLOT_FEATURES + GLOBAL_FEATURES

    @property
    def n_actions(self) -> int:
        return self.slots * len(self.quanta)

    def to_dict(self) -> dict[str, object]:
        return {"slots": self.slots, "quanta": list(self.quanta)}

    @staticmethod
    def from_dict(d: dict[str, object]) -> SpaceConfig:
        quanta = d.get("quanta", [2, 6, None])
        assert isinstance(quanta, list)
        slots = d.get("slots", 10)
        assert isinstance(slots, int)
        return SpaceConfig(slots=slots, quanta=tuple(quanta))


@dataclass(frozen=True)
class RewardConfig:
    """Weights of the scheduling cost the agent minimises.

    The env tracks a cumulative cost ``C(t)`` and pays ``-(C(t1) - C(t0)) /
    (N * S)`` for each decision, so the undiscounted episode return equals
    ``-C(end) / (N * S)``: it telescopes, and cannot be gamed by shifting cost
    between steps. With ``N`` processes and mean CPU burst ``S``::

        C = wait   * sum_i W_i                 (total waiting time)
          + fair   * sum_i sum_stretches L^2 / (2 S)
          + resp   * sum_i R_i                 (response time)
          + switch * n_context_switches * S

    * The ``fair`` term is quadratic in the length ``L`` of each uninterrupted
      wait, so leaving one process waiting for 100 ticks costs far more than
      ten processes waiting 10 ticks each. This is what stops the agent from
      lowering average waiting time by starving long jobs (the failure mode of
      a pure SRTF-style objective).
    * Turnaround = waiting + CPU + I/O, and CPU/I-O demand is fixed by the
      workload, so the waiting term also optimises turnaround.
    * Throughput only changes through idle time and switch overhead; idle time
      is mostly fixed by arrivals (the agent must always dispatch something),
      so the switch term carries it.
    """

    # Default calibration: with fair=0.1 the classic baselines (FCFS, SJF, RR)
    # score within ~15% of each other on the validation set, so no single
    # textbook policy is already optimal and the agent must trade efficiency
    # against fairness. fair=0 reproduces a pure average-wait objective.
    wait: float = 1.0
    fair: float = 0.1
    resp: float = 0.25
    switch: float = 0.1

    def __post_init__(self) -> None:
        for k, v in asdict(self).items():
            if not (v >= 0 and math.isfinite(v)):
                raise ValueError(f"reward weight '{k}' must be finite and >= 0")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def workload_scale(sim: Simulator) -> float:
    bursts = [b for s in sim.specs for b in s.cpu_bursts]
    return max(1.0, sum(bursts) / len(bursts))


def _l(x: float, s: float) -> float:
    return math.log1p(max(0.0, x) / s)


@dataclass
class ObservationBuilder:
    space: SpaceConfig = field(default_factory=SpaceConfig)

    def visible(self, sim: Simulator) -> list[Process]:
        return sorted(sim.ready, key=lambda p: p.enqueue_seq)[: self.space.slots]

    def build(self, sim: Simulator) -> np.ndarray:
        k = self.space.slots
        s = workload_scale(sim)
        n = len(sim.procs)
        now = sim.now
        max_prio = max(1, max(p.priority for p in sim.procs.values()))
        ordered = sorted(sim.ready, key=lambda p: p.enqueue_seq)
        vis, hidden = ordered[:k], ordered[k:]
        obs = np.zeros(self.space.obs_dim, dtype=np.float32)
        for i, p in enumerate(vis):
            o = i * SLOT_FEATURES
            obs[o + 0] = 1.0
            obs[o + 1] = _l(p.remaining_in_burst, s)
            obs[o + 2] = _l(p.remaining_cpu, s)
            obs[o + 3] = _l(p.current_wait(now), s)
            obs[o + 4] = _l(p.wait_so_far(now), s)
            obs[o + 5] = _l(now - p.arrival, s)
            obs[o + 6] = p.priority / max_prio
            obs[o + 7] = 1.0 if p.start_time is None else 0.0
            obs[o + 8] = min(1.0, p.cpu_bursts_left / 5.0)
            obs[o + 9] = 1.0 if sim.last_pid == p.pid else 0.0
        g = k * SLOT_FEATURES
        ne = sim.next_event_time()
        obs[g + 0] = len(sim.ready) / k
        obs[g + 1] = len(hidden) / k
        obs[g + 2] = len(sim.blocked) / n
        obs[g + 3] = sim.not_arrived / n
        obs[g + 4] = sim.n_terminated / n
        obs[g + 5] = _l(ne - now, s) if ne is not None else 0.0
        obs[g + 6] = sim.config.context_switch / s
        if hidden:
            obs[g + 7] = _l(sum(p.remaining_in_burst for p in hidden) / len(hidden), s)
            obs[g + 8] = _l(max(p.current_wait(now) for p in hidden), s)
        np.clip(obs, 0.0, OBS_HIGH, out=obs)
        return obs

    def mask(self, sim: Simulator) -> np.ndarray:
        nq = len(self.space.quanta)
        m = np.zeros(self.space.n_actions, dtype=bool)
        m[: min(len(sim.ready), self.space.slots) * nq] = True
        return m

    def decode(self, sim: Simulator, action: int) -> tuple[int, int | None]:
        """Map an action to ``(pid, quantum)``. Raises on an invalid action."""
        nq = len(self.space.quanta)
        if not 0 <= action < self.space.n_actions:
            raise ValueError(f"action {action} out of range")
        slot, qi = divmod(int(action), nq)
        vis = self.visible(sim)
        if slot >= len(vis):
            raise ValueError(f"action {action} selects empty slot {slot}")
        return vis[slot].pid, self.space.quanta[qi]


def cumulative_cost(sim: Simulator, rc: RewardConfig) -> float:
    """``C(now)`` as defined in :class:`RewardConfig`, in ticks."""
    s = workload_scale(sim)
    now = sim.now
    wait = 0.0
    sq = 0.0
    resp = 0.0
    for p in sim.procs.values():
        if p.arrival > now:
            continue
        wait += p.wait_so_far(now)
        sq += p.wait_sq_so_far(now)
        resp += (p.start_time if p.start_time is not None else now) - p.arrival
    return (
        rc.wait * wait
        + rc.fair * sq / s
        + rc.resp * resp
        + rc.switch * sim.context_switches * s
    )


def reward_normaliser(sim: Simulator) -> float:
    return len(sim.procs) * workload_scale(sim)
