"""Deterministic discrete-event CPU scheduling engine.

Time model
----------
Time is measured in integer ticks. The engine jumps directly between
*events* (process arrival, I/O completion, end of a CPU slice) instead of
iterating tick by tick.

Ordering rules at a single instant ``t`` (these make every run reproducible):

1. The running slice that ends at ``t`` stops executing.
2. New arrivals at ``t`` enter the ready queue, in ascending pid order.
3. Processes whose I/O completes at ``t`` enter the ready queue, ascending pid.
4. A process whose quantum expired (or that was preempted) at ``t`` goes to
   the *tail* of the ready queue, i.e. after the processes from steps 2-3.

The ready queue keeps insertion order (``enqueue_seq``). Schedulers that rank
processes by a key (SJF, SRTF, priority) break ties by that order, which is
equivalent to FCFS then lower pid.

Context switches
----------------
A context switch happens when the dispatcher loads a process different from
the last process that ran. The very first dispatch of a run is not a switch;
neither is re-dispatching the same process (e.g. Round Robin with one ready
process) or resuming the same process after an idle gap. Every switch is
counted; if ``context_switch > 0`` it also costs that many ticks of CPU time
during which no process makes progress. The dispatched process stays READY
(and keeps accumulating waiting time) during the switch.

For preemptive schedulers, the preemption check runs at every arrival or I/O
completion that happens while a process executes, and once after a context
switch if events occurred during the switch.

I/O
---
Each blocked process performs its own I/O in parallel (unlimited I/O
devices); only the CPU is contended.
"""

from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .process import Process, ProcessSpec, ProcessState, Transition


class SimulationError(RuntimeError):
    pass


@dataclass(frozen=True)
class SimConfig:
    context_switch: int = 0
    max_time: int = 5_000_000
    # A process "starves" if one uninterrupted READY stretch exceeds this many
    # ticks. ``None`` means 10x the workload's mean CPU burst length.
    starvation_threshold: int | None = None

    def __post_init__(self) -> None:
        if self.context_switch < 0:
            raise ValueError("context_switch must be >= 0")
        if self.max_time < 1:
            raise ValueError("max_time must be >= 1")
        if self.starvation_threshold is not None and self.starvation_threshold < 1:
            raise ValueError("starvation_threshold must be >= 1")


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    kind: str  # "run" | "idle" | "switch"
    pid: int | None = None


class PreemptionPolicy(Protocol):
    def should_preempt(
        self, running: Process, ready: Sequence[Process], now: int
    ) -> bool: ...


class Simulator:
    """Steppable simulation. Call :meth:`dispatch` whenever ``ready`` is non-empty."""

    def __init__(self, specs: Sequence[ProcessSpec], config: SimConfig | None = None):
        if not specs:
            raise ValueError("workload must contain at least one process")
        pids = [s.pid for s in specs]
        if len(set(pids)) != len(pids):
            raise ValueError("process ids must be unique")
        self.config = config or SimConfig()
        self.specs: tuple[ProcessSpec, ...] = tuple(sorted(specs, key=lambda s: s.pid))
        self.procs: dict[int, Process] = {s.pid: Process(s) for s in self.specs}
        self._pending: deque[Process] = deque(
            sorted(self.procs.values(), key=lambda p: (p.arrival, p.pid))
        )
        self._blocked: list[tuple[int, int]] = []  # heap of (io_until, pid)
        self.ready: list[Process] = []
        self.now = 0
        self.timeline: list[Segment] = []
        self.transitions: list[Transition] = []
        self.last_pid: int | None = None
        self.context_switches = 0
        self.switch_time = 0
        self.busy_time = 0
        self.decisions = 0
        self._seq = 0
        self._terminated = 0
        self.first_arrival = min(s.arrival for s in self.specs)
        self._process_events_through(self.now)
        self._advance_to_decision()

    # ------------------------------------------------------------------ state
    @property
    def done(self) -> bool:
        return self._terminated == len(self.procs)

    @property
    def n_terminated(self) -> int:
        return self._terminated

    @property
    def blocked(self) -> list[Process]:
        return [self.procs[pid] for _, pid in sorted(self._blocked)]

    @property
    def not_arrived(self) -> int:
        return len(self._pending)

    def next_event_time(self) -> int | None:
        times: list[int] = []
        if self._pending:
            times.append(self._pending[0].arrival)
        if self._blocked:
            times.append(self._blocked[0][0])
        return min(times) if times else None

    # -------------------------------------------------------------- internals
    def _record(self, tr: Transition) -> None:
        self.transitions.append(tr)

    def _enqueue(self, p: Process, t: int) -> None:
        self._record(p.transition(ProcessState.READY, t))
        p.ready_since = t
        p.enqueue_seq = self._seq
        self._seq += 1
        self.ready.append(p)

    def _process_events_at(self, t: int) -> bool:
        """Move arrivals then I/O completions occurring exactly at ``t`` to READY."""
        changed = False
        arrivals: list[Process] = []
        while self._pending and self._pending[0].arrival == t:
            arrivals.append(self._pending.popleft())
        for p in sorted(arrivals, key=lambda q: q.pid):
            self._enqueue(p, t)
            changed = True
        returns: list[int] = []
        while self._blocked and self._blocked[0][0] == t:
            returns.append(heapq.heappop(self._blocked)[1])
        for pid in sorted(returns):
            p = self.procs[pid]
            p.io_until = None
            p.burst_index += 1
            p.remaining_in_burst = p.spec.bursts[p.burst_index]
            self._enqueue(p, t)
            changed = True
        return changed

    def _process_events_through(self, t: int) -> bool:
        changed = False
        while True:
            ne = self.next_event_time()
            if ne is None or ne > t:
                return changed
            changed = self._process_events_at(ne) or changed

    def _add_segment(self, start: int, end: int, kind: str, pid: int | None) -> None:
        if end <= start:
            return
        if self.timeline:
            last = self.timeline[-1]
            if last.kind == kind and last.pid == pid and last.end == start:
                self.timeline[-1] = Segment(last.start, end, kind, pid)
                return
        self.timeline.append(Segment(start, end, kind, pid))

    def _check_time(self) -> None:
        if self.now > self.config.max_time:
            raise SimulationError(
                f"simulation exceeded max_time={self.config.max_time} ticks"
            )

    def _advance_to_decision(self) -> None:
        """Idle the CPU until a process is ready or every process has finished."""
        while not self.ready and not self.done:
            ne = self.next_event_time()
            if ne is None:  # pragma: no cover - would indicate an engine bug
                raise SimulationError("no ready process and no future event")
            self._add_segment(self.now, ne, "idle", None)
            self.now = ne
            self._check_time()
            self._process_events_at(ne)

    def _leave_ready(self, p: Process, t: int) -> None:
        assert p.ready_since is not None
        stretch = t - p.ready_since
        p.total_wait += stretch
        p.wait_sq_half += 0.5 * stretch * stretch
        p.max_wait = max(p.max_wait, stretch)
        p.ready_since = None

    def _execute(self, p: Process, until: int) -> None:
        d = until - self.now
        if d < 0:  # pragma: no cover
            raise SimulationError("negative execution interval")
        p.remaining_in_burst -= d
        p.remaining_cpu -= d
        p.cpu_time += d
        self.busy_time += d
        self.now = until
        self._check_time()

    # ---------------------------------------------------------------- dispatch
    def dispatch(
        self,
        pid: int,
        quantum: int | None = None,
        preemption: PreemptionPolicy | None = None,
    ) -> None:
        """Run ready process ``pid`` for up to ``quantum`` ticks (``None`` = rest of burst).

        Returns once the simulation reaches the next scheduling decision (a
        non-empty ready queue) or finishes.
        """
        if self.done:
            raise SimulationError("simulation already finished")
        if quantum is not None and quantum < 1:
            raise ValueError("quantum must be >= 1")
        idx = next((i for i, q in enumerate(self.ready) if q.pid == pid), None)
        if idx is None:
            raise SimulationError(f"process {pid} is not in the ready queue")
        p = self.ready.pop(idx)
        self.decisions += 1

        if self.last_pid is not None and self.last_pid != pid:
            self.context_switches += 1
            cs = self.config.context_switch
            if cs > 0:
                self._add_segment(self.now, self.now + cs, "switch", pid)
                self.switch_time += cs
                self.now += cs
                self._check_time()
                events = self._process_events_through(self.now)
                self.last_pid = pid
                if (
                    events
                    and preemption is not None
                    and preemption.should_preempt(p, self.ready, self.now)
                ):
                    # Abort the dispatch: p never ran and keeps its queue position.
                    self.ready.append(p)
                    self.ready.sort(key=lambda q: q.enqueue_seq)
                    self._advance_to_decision()
                    return
        self.last_pid = pid

        self._leave_ready(p, self.now)
        self._record(p.transition(ProcessState.RUNNING, self.now))
        p.dispatches += 1
        if p.start_time is None:
            p.start_time = self.now

        run_start = self.now
        slice_len = p.remaining_in_burst if quantum is None else min(quantum, p.remaining_in_burst)
        run_end = self.now + slice_len

        while True:
            ne = self.next_event_time()
            if ne is not None and ne < run_end:
                self._execute(p, ne)
                self._process_events_at(ne)
                if preemption is not None and preemption.should_preempt(
                    p, self.ready, self.now
                ):
                    self._add_segment(run_start, self.now, "run", pid)
                    self._enqueue(p, self.now)
                    self._advance_to_decision()
                    return
                continue
            self._execute(p, run_end)
            break

        self._add_segment(run_start, self.now, "run", pid)
        # Rule 2-3: events at the slice end are enqueued before p is re-enqueued.
        self._process_events_through(self.now)
        if p.remaining_in_burst == 0:
            if p.burst_index + 1 < len(p.spec.bursts):
                p.burst_index += 1  # now pointing at the I/O burst
                io = p.spec.bursts[p.burst_index]
                p.io_time += io
                p.io_until = self.now + io
                self._record(p.transition(ProcessState.BLOCKED, self.now))
                heapq.heappush(self._blocked, (p.io_until, p.pid))
            else:
                self._record(p.transition(ProcessState.TERMINATED, self.now))
                p.completion_time = self.now
                self._terminated += 1
        else:
            self._enqueue(p, self.now)
        self._advance_to_decision()
