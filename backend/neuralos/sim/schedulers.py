"""Baseline CPU schedulers.

Every scheduler implements the same small interface used by :func:`run`:

* ``select(ready, now)`` returns the process to dispatch and a quantum
  (``None`` = run until the current CPU burst ends).
* ``preemptive`` tells the engine whether to call ``should_preempt`` when a
  process arrives or returns from I/O while another one is running.

Burst lengths are assumed to be known in advance (the classic textbook
"oracle" assumption for SJF/SRTF). The RL agent receives exactly the same
information, so comparisons are like for like.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .process import Process

if TYPE_CHECKING:
    from .engine import Simulator


@dataclass(frozen=True)
class Decision:
    pid: int
    quantum: int | None = None


class Scheduler:
    name: str = "base"
    preemptive: bool = False

    def reset(self) -> None:
        """Called once before each simulation run."""

    def attach(self, sim: Simulator) -> None:
        """Gives schedulers that need global state (the RL policy) the simulator."""

    def select(self, ready: Sequence[Process], now: int) -> Decision:
        raise NotImplementedError

    def should_preempt(self, running: Process, ready: Sequence[Process], now: int) -> bool:
        return False

    def describe(self) -> dict[str, object]:
        return {"name": self.name, "preemptive": self.preemptive}


def _fifo_key(p: Process) -> int:
    return p.enqueue_seq


class FCFS(Scheduler):
    """First Come First Served: head of the ready queue, run the whole burst."""

    name = "fcfs"

    def select(self, ready: Sequence[Process], now: int) -> Decision:
        p = min(ready, key=_fifo_key)
        return Decision(p.pid)


class SJF(Scheduler):
    """Non-preemptive Shortest Job First on the *current* CPU burst length."""

    name = "sjf"

    def select(self, ready: Sequence[Process], now: int) -> Decision:
        p = min(ready, key=lambda q: (q.remaining_in_burst, q.enqueue_seq))
        return Decision(p.pid)


class SRTF(Scheduler):
    """Preemptive Shortest Remaining Time First.

    A running process is preempted only when a ready process has *strictly*
    less remaining time in its current burst, so equal times never thrash.
    """

    name = "srtf"
    preemptive = True

    def select(self, ready: Sequence[Process], now: int) -> Decision:
        p = min(ready, key=lambda q: (q.remaining_in_burst, q.enqueue_seq))
        return Decision(p.pid)

    def should_preempt(self, running: Process, ready: Sequence[Process], now: int) -> bool:
        return any(q.remaining_in_burst < running.remaining_in_burst for q in ready)


class RoundRobin(Scheduler):
    name = "rr"

    def __init__(self, quantum: int = 4):
        if quantum < 1:
            raise ValueError("Round Robin quantum must be >= 1")
        self.quantum = quantum

    def select(self, ready: Sequence[Process], now: int) -> Decision:
        p = min(ready, key=_fifo_key)
        return Decision(p.pid, self.quantum)

    def describe(self) -> dict[str, object]:
        return {**super().describe(), "quantum": self.quantum}


class PriorityScheduler(Scheduler):
    """Priority scheduling; lower number = higher priority.

    * ``preemptive=False`` (default): the chosen process runs its full burst.
    * ``preemptive=True``: a newly ready process with strictly higher
      (numerically lower) effective priority preempts the running one.
    * ``aging``: if set, a waiting process's effective priority improves by 1
      for every ``aging`` ticks of its current wait (floor at 0). This is the
      standard fix for starvation of low-priority work.

    Ties are broken by ready-queue order (FCFS).
    """

    name = "priority"

    def __init__(self, preemptive: bool = False, aging: int | None = None):
        if aging is not None and aging < 1:
            raise ValueError("aging interval must be >= 1")
        self.preemptive = preemptive
        self.aging = aging
        self.name = "priority_p" if preemptive else "priority"

    def effective(self, p: Process, now: int) -> int:
        if self.aging is None:
            return p.priority
        return max(0, p.priority - p.current_wait(now) // self.aging)

    def select(self, ready: Sequence[Process], now: int) -> Decision:
        p = min(ready, key=lambda q: (self.effective(q, now), q.enqueue_seq))
        return Decision(p.pid)

    def should_preempt(self, running: Process, ready: Sequence[Process], now: int) -> bool:
        return any(self.effective(q, now) < running.priority for q in ready)

    def describe(self) -> dict[str, object]:
        return {**super().describe(), "aging": self.aging}


class RandomScheduler(Scheduler):
    """Seeded uniform-random choice. A sanity floor for the RL agent."""

    name = "random"

    def __init__(self, seed: int = 0, quantum: int | None = None):
        self.seed = seed
        self.quantum = quantum
        self._rng = random.Random(seed)

    def reset(self) -> None:
        self._rng = random.Random(self.seed)

    def select(self, ready: Sequence[Process], now: int) -> Decision:
        ordered = sorted(ready, key=_fifo_key)
        return Decision(self._rng.choice(ordered).pid, self.quantum)


BASELINES = ("fcfs", "sjf", "srtf", "rr", "priority", "priority_p", "random")


def make_baseline(
    name: str,
    quantum: int = 4,
    aging: int | None = None,
    seed: int = 0,
) -> Scheduler:
    if name == "fcfs":
        return FCFS()
    if name == "sjf":
        return SJF()
    if name == "srtf":
        return SRTF()
    if name == "rr":
        return RoundRobin(quantum)
    if name == "priority":
        return PriorityScheduler(preemptive=False, aging=aging)
    if name == "priority_p":
        return PriorityScheduler(preemptive=True, aging=aging)
    if name == "random":
        return RandomScheduler(seed=seed)
    raise ValueError(f"unknown scheduler '{name}'. Choose one of: {', '.join(BASELINES)}")
