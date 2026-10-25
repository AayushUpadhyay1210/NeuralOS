"""Property-style invariants checked on many generated workloads."""

from __future__ import annotations

import pytest

from neuralos.sim import (
    BASELINES,
    WORKLOAD_KINDS,
    ProcessState,
    SimConfig,
    Simulator,
    WorkloadConfig,
    generate,
    make_baseline,
    run,
)
from neuralos.sim.process import ALLOWED_TRANSITIONS, InvalidTransition, Process, ProcessSpec


@pytest.mark.parametrize("kind", WORKLOAD_KINDS)
@pytest.mark.parametrize("sched", BASELINES)
@pytest.mark.parametrize("cs", [0, 2])
def test_invariants(kind, sched, cs):
    specs = generate(WorkloadConfig(kind=kind, n_processes=25, seed=7))
    r = run(specs, make_baseline(sched, quantum=3, seed=1), SimConfig(context_switch=cs))

    # Timeline is contiguous from t=0, never overlaps.
    t = 0
    for seg in r.timeline:
        assert seg.start == t and seg.end > seg.start
        t = seg.end
    end = t
    assert end == max(p.completion for p in r.processes)

    # Each process receives exactly its CPU demand.
    got: dict[int, int] = {}
    for seg in r.timeline:
        if seg.kind == "run":
            got[seg.pid] = got.get(seg.pid, 0) + seg.end - seg.start
    by = {s.pid: s for s in specs}
    assert got == {pid: s.total_cpu for pid, s in by.items()}

    for p in r.processes:
        assert p.waiting >= 0 and p.response >= 0
        assert p.response <= p.waiting
        assert p.turnaround == p.waiting + p.total_cpu + p.total_io
        assert p.start >= p.arrival

    # Switch accounting.
    switch_ticks = sum(s.end - s.start for s in r.timeline if s.kind == "switch")
    assert switch_ticks == r.metrics.switch_overhead == cs * r.metrics.context_switches

    # Transitions are legal and every process ends TERMINATED.
    last: dict[int, ProcessState] = {}
    for tr in r.transitions:
        prev = last.get(tr.pid, ProcessState.NEW)
        assert tr.from_state == prev
        assert tr.to_state in ALLOWED_TRANSITIONS[prev]
        last[tr.pid] = tr.to_state
    assert all(s is ProcessState.TERMINATED for s in last.values())
    assert 0 < r.metrics.cpu_utilization <= 1.0
    assert 0 < r.metrics.jain_fairness <= 1.0 + 1e-9


def test_round_robin_slices_never_exceed_quantum():
    specs = generate(WorkloadConfig(kind="cpu_bound", n_processes=30, seed=3))
    sim = Simulator(specs)
    sched = make_baseline("rr", quantum=4)
    while not sim.done:
        d = sched.select(sim.ready, sim.now)
        t0 = sim.now
        before = sim.procs[d.pid].cpu_time
        sim.dispatch(d.pid, d.quantum)
        assert sim.procs[d.pid].cpu_time - before <= 4
        assert sim.now >= t0


def test_illegal_transition_rejected():
    p = Process(ProcessSpec(pid=0, arrival=0, bursts=(3,)))
    with pytest.raises(InvalidTransition):
        p.transition(ProcessState.RUNNING, 0)  # NEW -> RUNNING skips READY


def test_dispatch_validation():
    sim = Simulator([ProcessSpec(0, 0, (2,)), ProcessSpec(1, 5, (2,))])
    with pytest.raises(Exception):
        sim.dispatch(1)  # not ready yet
    with pytest.raises(ValueError):
        sim.dispatch(0, quantum=0)


def test_max_time_guard():
    from neuralos.sim import SimulationError

    with pytest.raises(SimulationError):
        run([ProcessSpec(0, 0, (50,))], make_baseline("fcfs"), SimConfig(max_time=10))
