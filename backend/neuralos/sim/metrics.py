"""Metrics computed from a finished simulation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .engine import Simulator


@dataclass(frozen=True)
class ProcessStats:
    pid: int
    arrival: int
    priority: int
    total_cpu: int
    total_io: int
    cpu_bursts: int
    start: int
    completion: int
    waiting: int
    turnaround: int
    response: int
    max_wait: int
    slowdown: float
    dispatches: int


@dataclass(frozen=True)
class Metrics:
    n_processes: int
    makespan: int
    avg_waiting: float
    std_waiting: float
    avg_turnaround: float
    std_turnaround: float
    avg_response: float
    std_response: float
    p95_turnaround: float
    throughput: float  # processes completed per 100 ticks
    cpu_utilization: float  # fraction of [first arrival, last completion] spent executing
    context_switches: int
    switch_overhead: int
    idle_time: int
    starvation_threshold: int
    starvation_count: int
    max_wait: int
    avg_slowdown: float
    max_slowdown: float
    jain_fairness: float  # Jain index over slowdowns; 1.0 = perfectly even

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _std(xs: list[float]) -> float:
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _percentile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def default_starvation_threshold(sim: Simulator) -> int:
    bursts = [b for s in sim.specs for b in s.cpu_bursts]
    return max(1, round(10 * sum(bursts) / len(bursts)))


def process_stats(sim: Simulator) -> list[ProcessStats]:
    if not sim.done:
        raise ValueError("simulation has not finished")
    out: list[ProcessStats] = []
    for p in sim.procs.values():
        assert p.completion_time is not None and p.start_time is not None
        turnaround = p.completion_time - p.arrival
        waiting = turnaround - p.spec.total_cpu - p.spec.total_io
        if waiting != p.total_wait:  # pragma: no cover - engine invariant
            raise AssertionError(
                f"waiting-time invariant broken for pid {p.pid}: {waiting} != {p.total_wait}"
            )
        service = p.spec.total_cpu + p.spec.total_io
        out.append(
            ProcessStats(
                pid=p.pid,
                arrival=p.arrival,
                priority=p.priority,
                total_cpu=p.spec.total_cpu,
                total_io=p.spec.total_io,
                cpu_bursts=len(p.spec.cpu_bursts),
                start=p.start_time,
                completion=p.completion_time,
                waiting=waiting,
                turnaround=turnaround,
                response=p.start_time - p.arrival,
                max_wait=p.max_wait,
                slowdown=turnaround / service,
                dispatches=p.dispatches,
            )
        )
    return out


def compute_metrics(sim: Simulator) -> Metrics:
    stats = process_stats(sim)
    n = len(stats)
    end = max(s.completion for s in stats)
    makespan = max(1, end - sim.first_arrival)
    waits = [float(s.waiting) for s in stats]
    tats = [float(s.turnaround) for s in stats]
    resps = [float(s.response) for s in stats]
    slow = [s.slowdown for s in stats]
    threshold = sim.config.starvation_threshold or default_starvation_threshold(sim)
    idle = sum(seg.end - seg.start for seg in sim.timeline if seg.kind == "idle"
               and seg.start >= sim.first_arrival)
    jain = (sum(slow) ** 2) / (n * sum(x * x for x in slow))
    return Metrics(
        n_processes=n,
        makespan=makespan,
        avg_waiting=_mean(waits),
        std_waiting=_std(waits),
        avg_turnaround=_mean(tats),
        std_turnaround=_std(tats),
        avg_response=_mean(resps),
        std_response=_std(resps),
        p95_turnaround=_percentile(tats, 0.95),
        throughput=100.0 * n / makespan,
        cpu_utilization=sim.busy_time / makespan,
        context_switches=sim.context_switches,
        switch_overhead=sim.switch_time,
        idle_time=idle,
        starvation_threshold=threshold,
        starvation_count=sum(1 for s in stats if s.max_wait > threshold),
        max_wait=max(s.max_wait for s in stats),
        avg_slowdown=_mean(slow),
        max_slowdown=max(slow),
        jain_fairness=jain,
    )
