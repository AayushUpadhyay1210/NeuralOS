"""Process model: immutable specification plus mutable runtime state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ProcessState(str, Enum):
    NEW = "NEW"
    READY = "READY"
    RUNNING = "RUNNING"
    BLOCKED = "BLOCKED"
    TERMINATED = "TERMINATED"


ALLOWED_TRANSITIONS: dict[ProcessState, frozenset[ProcessState]] = {
    ProcessState.NEW: frozenset({ProcessState.READY}),
    ProcessState.READY: frozenset({ProcessState.RUNNING}),
    ProcessState.RUNNING: frozenset(
        {ProcessState.READY, ProcessState.BLOCKED, ProcessState.TERMINATED}
    ),
    ProcessState.BLOCKED: frozenset({ProcessState.READY}),
    ProcessState.TERMINATED: frozenset(),
}


class InvalidTransition(RuntimeError):
    pass


@dataclass(frozen=True)
class ProcessSpec:
    """Static description of a process.

    ``bursts`` alternates CPU and I/O durations and always starts and ends with
    a CPU burst: ``(cpu, io, cpu, io, ..., cpu)``. A purely CPU-bound process
    has a single burst. All durations are positive integers (ticks).
    Lower ``priority`` numbers mean higher priority (Unix convention).
    """

    pid: int
    arrival: int
    bursts: tuple[int, ...]
    priority: int = 0

    def __post_init__(self) -> None:
        if self.pid < 0:
            raise ValueError("pid must be non-negative")
        if self.arrival < 0:
            raise ValueError(f"process {self.pid}: arrival must be >= 0")
        if len(self.bursts) == 0 or len(self.bursts) % 2 == 0:
            raise ValueError(
                f"process {self.pid}: bursts must have odd length (CPU, IO, ..., CPU)"
            )
        if any((not isinstance(b, int)) or b < 1 for b in self.bursts):
            raise ValueError(f"process {self.pid}: every burst must be an integer >= 1")
        if self.priority < 0:
            raise ValueError(f"process {self.pid}: priority must be >= 0")

    @property
    def cpu_bursts(self) -> tuple[int, ...]:
        return self.bursts[0::2]

    @property
    def io_bursts(self) -> tuple[int, ...]:
        return self.bursts[1::2]

    @property
    def total_cpu(self) -> int:
        return sum(self.cpu_bursts)

    @property
    def total_io(self) -> int:
        return sum(self.io_bursts)


@dataclass(frozen=True)
class Transition:
    time: int
    pid: int
    from_state: ProcessState
    to_state: ProcessState


@dataclass
class Process:
    """Runtime state of a process inside one simulation."""

    spec: ProcessSpec
    state: ProcessState = ProcessState.NEW
    burst_index: int = 0
    remaining_in_burst: int = 0
    remaining_cpu: int = 0
    start_time: int | None = None
    completion_time: int | None = None
    ready_since: int | None = None
    enqueue_seq: int = -1
    io_until: int | None = None
    total_wait: int = 0
    max_wait: int = 0
    wait_sq_half: float = 0.0  # sum over completed READY stretches of L^2 / 2
    cpu_time: int = 0
    io_time: int = 0
    dispatches: int = 0
    history: list[Transition] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.remaining_in_burst = self.spec.bursts[0]
        self.remaining_cpu = self.spec.total_cpu

    @property
    def pid(self) -> int:
        return self.spec.pid

    @property
    def arrival(self) -> int:
        return self.spec.arrival

    @property
    def priority(self) -> int:
        return self.spec.priority

    @property
    def cpu_bursts_left(self) -> int:
        """CPU bursts not yet finished, including the current one."""
        return len(self.spec.cpu_bursts) - self.burst_index // 2

    def current_wait(self, now: int) -> int:
        """Length of the current uninterrupted READY stretch (0 if not ready)."""
        if self.state is ProcessState.READY and self.ready_since is not None:
            return now - self.ready_since
        return 0

    def wait_so_far(self, now: int) -> int:
        return self.total_wait + self.current_wait(now)

    def wait_sq_so_far(self, now: int) -> float:
        w = self.current_wait(now)
        return self.wait_sq_half + 0.5 * w * w

    def transition(self, to: ProcessState, time: int) -> Transition:
        if to not in ALLOWED_TRANSITIONS[self.state]:
            raise InvalidTransition(
                f"process {self.pid}: illegal transition {self.state.value} -> {to.value}"
                f" at t={time}"
            )
        tr = Transition(time=time, pid=self.pid, from_state=self.state, to_state=to)
        self.state = to
        self.history.append(tr)
        return tr
