"""Scheduler correctness against hand-calculated examples.

Every expected number in this file was worked out by hand (see the comments)
rather than produced by the code under test.
"""

from __future__ import annotations

import pytest

from neuralos.sim import ProcessSpec, SimConfig, make_baseline, run
from neuralos.sim.schedulers import PriorityScheduler, RoundRobin


def P(pid: int, arrival: int, *bursts: int, prio: int = 0) -> ProcessSpec:
    return ProcessSpec(pid=pid, arrival=arrival, bursts=tuple(bursts), priority=prio)


# Classic 4-process example: (arrival, burst) = (0,8) (1,4) (2,9) (3,5)
CLASSIC = [P(1, 0, 8), P(2, 1, 4), P(3, 2, 9), P(4, 3, 5)]


def runs(result) -> list[tuple[int, int, int | None]]:
    return [(s.start, s.end, s.pid) for s in result.timeline if s.kind == "run"]


def by_pid(result):
    return {p.pid: p for p in result.processes}


def test_fcfs_classic():
    r = run(CLASSIC, make_baseline("fcfs"))
    # P1 0-8, P2 8-12, P3 12-21, P4 21-26
    assert runs(r) == [(0, 8, 1), (8, 12, 2), (12, 21, 3), (21, 26, 4)]
    ps = by_pid(r)
    assert [ps[i].waiting for i in (1, 2, 3, 4)] == [0, 7, 10, 18]
    assert [ps[i].turnaround for i in (1, 2, 3, 4)] == [8, 11, 19, 23]
    assert r.metrics.avg_waiting == pytest.approx(8.75)
    assert r.metrics.avg_turnaround == pytest.approx(15.25)
    assert r.metrics.avg_response == pytest.approx(8.75)
    assert r.metrics.context_switches == 3
    assert r.metrics.cpu_utilization == pytest.approx(1.0)


def test_sjf_nonpreemptive_classic():
    r = run(CLASSIC, make_baseline("sjf"))
    # Only P1 is ready at 0, so it runs to 8; then P2(4), P4(5), P3(9).
    assert runs(r) == [(0, 8, 1), (8, 12, 2), (12, 17, 4), (17, 26, 3)]
    assert r.metrics.avg_waiting == pytest.approx((0 + 7 + 15 + 9) / 4)


def test_srtf_classic():
    r = run(CLASSIC, make_baseline("srtf"))
    # P1 0-1, P2 1-5, P4 5-10, P1 10-17, P3 17-26 -> waits 9,0,15,2 (avg 6.5)
    assert runs(r) == [(0, 1, 1), (1, 5, 2), (5, 10, 4), (10, 17, 1), (17, 26, 3)]
    ps = by_pid(r)
    assert [ps[i].waiting for i in (1, 2, 3, 4)] == [9, 0, 15, 2]
    assert r.metrics.avg_waiting == pytest.approx(6.5)
    assert ps[1].response == 0 and ps[3].response == 15
    assert r.metrics.context_switches == 4


def test_round_robin_same_arrival():
    procs = [P(1, 0, 5), P(2, 0, 3), P(3, 0, 1)]
    r = run(procs, RoundRobin(quantum=2))
    # P1 0-2, P2 2-4, P3 4-5, P1 5-7, P2 7-8, P1 8-9
    assert runs(r) == [(0, 2, 1), (2, 4, 2), (4, 5, 3), (5, 7, 1), (7, 8, 2), (8, 9, 1)]
    ps = by_pid(r)
    assert [ps[i].waiting for i in (1, 2, 3)] == [4, 5, 4]
    assert [ps[i].response for i in (1, 2, 3)] == [0, 2, 4]
    assert r.metrics.context_switches == 5


def test_round_robin_arrival_beats_expired_process():
    # P1's quantum expires at t=2 exactly when P2 arrives: P2 is queued first.
    r = run([P(1, 0, 4), P(2, 2, 2)], RoundRobin(quantum=2))
    assert runs(r) == [(0, 2, 1), (2, 4, 2), (4, 6, 1)]


def test_round_robin_single_process_has_no_switches():
    r = run([P(1, 0, 7)], RoundRobin(quantum=2), SimConfig(context_switch=3))
    assert runs(r) == [(0, 7, 1)]  # contiguous segments are merged
    assert r.metrics.context_switches == 0
    assert r.decisions == 4  # 2+2+2+1


PRIO = [
    P(1, 0, 10, prio=3),
    P(2, 1, 1, prio=1),
    P(3, 2, 2, prio=4),
    P(4, 3, 1, prio=5),
    P(5, 4, 5, prio=2),
]


def test_priority_nonpreemptive():
    r = run(PRIO, make_baseline("priority"))
    # P1 0-10, then P2 10-11, P5 11-16, P3 16-18, P4 18-19
    assert runs(r) == [(0, 10, 1), (10, 11, 2), (11, 16, 5), (16, 18, 3), (18, 19, 4)]
    assert r.metrics.avg_waiting == pytest.approx((0 + 9 + 14 + 15 + 7) / 5)


def test_priority_preemptive():
    r = run(PRIO, make_baseline("priority_p"))
    # P1 0-1, P2 1-2, P1 2-4, P5 4-9, P1 9-16, P3 16-18, P4 18-19
    assert runs(r) == [
        (0, 1, 1), (1, 2, 2), (2, 4, 1), (4, 9, 5), (9, 16, 1), (16, 18, 3), (18, 19, 4)
    ]
    ps = by_pid(r)
    assert [ps[i].waiting for i in (1, 2, 3, 4, 5)] == [6, 0, 14, 15, 0]


def test_simultaneous_arrivals_ordered_by_pid():
    # Specs supplied out of order; FCFS must still run pid order on ties.
    procs = [P(3, 0, 1), P(1, 0, 1), P(2, 0, 1)]
    r = run(procs, make_baseline("fcfs"))
    assert [pid for _, _, pid in runs(r)] == [1, 2, 3]


def test_sjf_tie_breaks_fcfs():
    procs = [P(1, 0, 2), P(2, 1, 3), P(3, 1, 3), P(4, 1, 3)]
    r = run(procs, make_baseline("sjf"))
    assert [pid for _, _, pid in runs(r)] == [1, 2, 3, 4]


def test_idle_period():
    r = run([P(1, 0, 2), P(2, 5, 3)], make_baseline("fcfs"))
    kinds = [(s.start, s.end, s.kind) for s in r.timeline]
    assert kinds == [(0, 2, "run"), (2, 5, "idle"), (5, 8, "run")]
    assert r.metrics.idle_time == 3
    assert r.metrics.cpu_utilization == pytest.approx(5 / 8)
    assert r.metrics.avg_waiting == 0
    # Same process resumed after idle would not be a switch; a different one is.
    assert r.metrics.context_switches == 1


def test_late_first_arrival_excluded_from_utilization():
    r = run([P(1, 10, 4)], make_baseline("fcfs"))
    assert r.timeline[0].kind == "idle" and r.timeline[0].end == 10
    assert r.metrics.cpu_utilization == pytest.approx(1.0)
    assert r.metrics.makespan == 4


def test_context_switch_overhead():
    r = run([P(1, 0, 3), P(2, 0, 2)], make_baseline("fcfs"), SimConfig(context_switch=1))
    assert [(s.start, s.end, s.kind, s.pid) for s in r.timeline] == [
        (0, 3, "run", 1),
        (3, 4, "switch", 2),
        (4, 6, "run", 2),
    ]
    ps = by_pid(r)
    assert ps[2].waiting == 4 and ps[2].response == 4
    assert r.metrics.context_switches == 1 and r.metrics.switch_overhead == 1
    assert r.metrics.cpu_utilization == pytest.approx(5 / 6)


def test_srtf_preemption_during_context_switch():
    # cs=2. P2 preempts P1 at t=1; while switching to P2 (1-3) P3 (burst 1)
    # arrives, so the dispatch of P2 is aborted and P3 runs next.
    procs = [P(1, 0, 10), P(2, 1, 5), P(3, 2, 1)]
    r = run(procs, make_baseline("srtf"), SimConfig(context_switch=2))
    tl = [(s.start, s.end, s.kind, s.pid) for s in r.timeline]
    assert tl == [
        (0, 1, "run", 1),
        (1, 3, "switch", 2),
        (3, 5, "switch", 3),
        (5, 6, "run", 3),
        (6, 8, "switch", 2),
        (8, 13, "run", 2),
        (13, 15, "switch", 1),
        (15, 24, "run", 1),
    ]
    ps = by_pid(r)
    assert ps[3].response == 3
    assert r.metrics.context_switches == 4 and r.metrics.switch_overhead == 8


def test_io_blocking_and_state_transitions():
    procs = [P(1, 0, 2, 3, 2), P(2, 0, 4)]
    r = run(procs, make_baseline("fcfs"))
    # P1 0-2, blocked 2-5; P2 2-6; P1 6-8
    assert runs(r) == [(0, 2, 1), (2, 6, 2), (6, 8, 1)]
    ps = by_pid(r)
    assert ps[1].waiting == 1 and ps[1].turnaround == 8 and ps[1].total_io == 3
    assert ps[2].waiting == 2
    p1 = [(t.time, t.from_state.value, t.to_state.value) for t in r.transitions if t.pid == 1]
    assert p1 == [
        (0, "NEW", "READY"),
        (0, "READY", "RUNNING"),
        (2, "RUNNING", "BLOCKED"),
        (5, "BLOCKED", "READY"),
        (6, "READY", "RUNNING"),
        (8, "RUNNING", "TERMINATED"),
    ]


def test_io_return_while_cpu_idle():
    r = run([P(1, 0, 1, 4, 1)], make_baseline("fcfs"))
    assert [(s.start, s.end, s.kind) for s in r.timeline] == [
        (0, 1, "run"), (1, 5, "idle"), (5, 6, "run")
    ]
    assert r.metrics.context_switches == 0


def test_priority_starvation_and_aging():
    # A low-priority long job (prio 9) competes with a stream of 1-tick
    # high-priority jobs arriving every tick (100% load from the stream alone).
    procs = [P(0, 0, 1, prio=0), P(1, 0, 20, prio=9)]
    procs += [P(i, i - 1, 1, prio=0) for i in range(2, 62)]
    cfg = SimConfig(starvation_threshold=30)
    starve = run(procs, make_baseline("priority"), cfg)
    aged = run(procs, PriorityScheduler(aging=2), cfg)
    # Without aging the long job waits until the stream dries up (t=61).
    assert by_pid(starve)[1].max_wait == 61
    assert starve.metrics.starvation_count == 1
    # With aging=2 it reaches priority 0 after 18 ticks and, being the oldest
    # entry in the queue, wins the tie.
    assert by_pid(aged)[1].max_wait == 18
    assert aged.metrics.starvation_count == 0


def test_rejects_invalid_specs():
    with pytest.raises(ValueError):
        ProcessSpec(pid=1, arrival=0, bursts=(3, 2))  # must end with CPU
    with pytest.raises(ValueError):
        ProcessSpec(pid=1, arrival=0, bursts=(0,))
    with pytest.raises(ValueError):
        ProcessSpec(pid=1, arrival=-1, bursts=(1,))
    with pytest.raises(ValueError):
        run([P(1, 0, 1), P(1, 0, 2)], make_baseline("fcfs"))
    with pytest.raises(ValueError):
        RoundRobin(0)
    with pytest.raises(ValueError):
        make_baseline("lottery")
