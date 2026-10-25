"""Run a scheduler over a workload and package the result."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .engine import Segment, SimConfig, Simulator
from .metrics import Metrics, ProcessStats, compute_metrics, process_stats
from .process import ProcessSpec, Transition
from .schedulers import Scheduler


@dataclass
class SimulationResult:
    scheduler: dict[str, object]
    config: SimConfig
    metrics: Metrics
    processes: list[ProcessStats]
    timeline: list[Segment]
    transitions: list[Transition]
    decisions: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheduler": self.scheduler,
            "config": asdict(self.config),
            "metrics": self.metrics.to_dict(),
            "processes": [asdict(p) for p in self.processes],
            "timeline": [asdict(s) for s in self.timeline],
            "transitions": [
                {
                    "time": t.time,
                    "pid": t.pid,
                    "from": t.from_state.value,
                    "to": t.to_state.value,
                }
                for t in self.transitions
            ],
            "decisions": self.decisions,
        }


def simulate(
    specs: Sequence[ProcessSpec],
    scheduler: Scheduler,
    config: SimConfig | None = None,
) -> Simulator:
    """Drive ``scheduler`` over ``specs`` and return the finished simulator."""
    sim = Simulator(specs, config)
    scheduler.reset()
    scheduler.attach(sim)
    preemption = scheduler if scheduler.preemptive else None
    while not sim.done:
        decision = scheduler.select(sim.ready, sim.now)
        sim.dispatch(decision.pid, decision.quantum, preemption)
    return sim


def result_from(sim: Simulator, scheduler: Scheduler) -> SimulationResult:
    return SimulationResult(
        scheduler=scheduler.describe(),
        config=sim.config,
        metrics=compute_metrics(sim),
        processes=process_stats(sim),
        timeline=list(sim.timeline),
        transitions=list(sim.transitions),
        decisions=sim.decisions,
    )


def run(
    specs: Sequence[ProcessSpec],
    scheduler: Scheduler,
    config: SimConfig | None = None,
) -> SimulationResult:
    return result_from(simulate(specs, scheduler, config), scheduler)
