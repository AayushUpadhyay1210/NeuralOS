# Simulation model, schedulers and metrics

## Processes

A process (`sim.process.ProcessSpec`) has a pid, an arrival time, a priority (lower number = higher priority) and a burst list that alternates CPU and I/O and starts and ends with CPU: `(cpu, io, cpu, …, cpu)`. All durations are integer ticks ≥ 1.

Runtime state (`sim.process.Process`) tracks the remaining time in the current burst, total remaining CPU, first-run time, completion time, total and longest uninterrupted wait, and a history of state transitions.

Allowed transitions (anything else raises `InvalidTransition`):

```
NEW ──▶ READY ──▶ RUNNING ──▶ TERMINATED
          ▲          │
          │          ├──▶ READY    (quantum expired or preempted)
          │          └──▶ BLOCKED  (CPU burst done, I/O next)
          └────────────── BLOCKED  (I/O done)
```

## Time and events

Time is integer ticks. The engine (`sim.engine.Simulator`) is event-driven: it jumps directly to the next arrival, I/O completion or slice end instead of iterating tick by tick.

Ordering at a single instant `t`:

1. the slice ending at `t` stops executing;
2. arrivals at `t` join the ready queue in ascending pid order;
3. I/O completions at `t` join in ascending pid order;
4. a process whose quantum expired, or that was preempted, at `t` joins the tail.

The ready queue keeps insertion order (`enqueue_seq`). Schedulers that rank processes (SJF, SRTF, priority) break ties by that order, which amounts to FCFS and then lower pid.

**Idle periods.** When nothing is ready, the CPU idles until the next event; the gap appears as an `idle` segment.

**I/O.** Every blocked process does its I/O in parallel (unlimited devices). Only the CPU is contended.

**Context switches.** A switch is counted when the dispatcher loads a process different from the last one that ran. Neither the first dispatch nor re-dispatching the same process counts as a switch (for example, Round Robin with a single ready process). Resuming the same process after an idle gap is not a switch either. With `context_switch = c > 0`, each switch costs `c` ticks: the timeline shows a `switch` segment, and the incoming process stays READY, so the switch time counts as its waiting time. For preemptive schedulers, if a process arrives during a switch and should preempt the one being loaded, the dispatch is aborted after the switch completes and the dispatcher switches again.

**Safety.** A run that exceeds `max_time` (default 5,000,000 ticks) raises `SimulationError`.

## Baseline schedulers (`sim/schedulers.py`)

| Name | Rule | Preemptive |
|---|---|---|
| `fcfs` | Head of the ready queue runs its whole CPU burst | no |
| `sjf` | Shortest current CPU burst; ties FCFS | no |
| `srtf` | Shortest remaining time in current burst. A ready process preempts only with *strictly* less remaining time, so equal times never thrash | yes |
| `rr` | Head of queue runs up to `quantum` ticks, then goes to the tail | by quantum |
| `priority` | Lowest priority number; runs whole burst | no |
| `priority_p` | Lowest priority number; preempts on strictly higher priority | yes |
| `random` | Seeded uniform choice among ready processes | no |

Priority supports optional **aging**: a waiting process's effective priority improves by one level per `aging` ticks of its current wait. The tests show the effect. With a 100%-load stream of high-priority jobs, a priority-9 job waits 61 ticks without aging and 18 ticks with `aging=2`.

SJF, SRTF and the RL agent use the true burst lengths, which is the textbook oracle assumption. A real kernel would have to estimate them.

## Workloads (`sim/workload.py`)

All randomness comes from `numpy.random.default_rng(seed)`, so the same configuration always produces the same trace.

| Kind | Arrivals (mean gap) | CPU bursts | I/O |
|---|---|---|---|
| `cpu_bound` | 11 | exponential, mean 10 | none |
| `io_bound` | 7 | exponential, mean 3, 2–5 bursts | every process, mean 12 |
| `short_jobs` | 3.5 | uniform, mean 3 | none |
| `long_jobs` | 28 | log-normal, mean 25 | 10% of processes |
| `mixed_priority` | 7 | exponential, mean 4; 20% are 6× longer and skew to low priority | 20% |
| `bursty` | clusters of 3–8 near-simultaneous arrivals | exponential, mean 5; 15% long | 20% |
| `mixed` | 7 | log-normal, mean 4; 15% long | 40% |

Every preset value can be overridden (mean gap, mean burst, distribution, I/O share, mean I/O burst, priority levels). Limits: at most 500 processes, bursts capped at 500 ticks.

## Metrics (`sim/metrics.py`)

Per process: response = first run − arrival; turnaround = completion − arrival; waiting = turnaround − CPU − I/O, which the engine cross-checks against the time it actually spent READY; longest single wait; slowdown = turnaround ÷ (CPU + I/O).

Aggregate:

| Metric | Definition |
|---|---|
| `avg_waiting`, `avg_turnaround`, `avg_response` | Means, with population standard deviations; `p95_turnaround` |
| `throughput` | Completed processes per 100 ticks over `[first arrival, last completion]` |
| `cpu_utilization` | Ticks executing processes ÷ that span (switch time and idle time excluded) |
| `context_switches`, `switch_overhead` | Count of switches and ticks spent switching |
| `idle_time` | Idle ticks after the first arrival |
| `starvation_count` | Processes whose longest uninterrupted wait exceeds the threshold (default 10 × mean CPU burst of the workload) |
| `max_wait` | Longest uninterrupted wait of any process |
| `jain_fairness` | Jain's index over slowdowns, (Σx)² / (n·Σx²); 1 = everyone slowed down equally |
| `objective` | The RL episode cost (see `rl-design.md`), computed identically for every scheduler |

## Verification

`tests/test_schedulers.py` checks every algorithm against hand-worked textbook examples. One example is the classic four-process SRTF case, with average waiting time 6.5. Other cases cover Round Robin tie rules, both priority modes, idle gaps, context-switch overhead, I/O blocking with its exact state-transition sequence, and an SRTF preemption that happens during a context switch. `tests/test_engine_invariants.py` runs every scheduler on every workload type, with and without switch cost, and checks these invariants:

* the timeline is contiguous;
* each process receives exactly its CPU demand;
* turnaround = waiting + CPU + I/O;
* every transition is legal;
* switch overhead = cost × switches.
