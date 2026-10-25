# Reinforcement-learning design

## Algorithm

`MaskablePPO` from sb3-contrib: standard PPO with invalid-action masking, which is the "correct masking" the action space needs. Policy and value networks are MLPs with two hidden layers of 128 units. CPU only.

| Hyper-parameter | Value |
|---|---|
| parallel envs | 4 (`DummyVecEnv`) |
| `n_steps` × envs | 512 × 4 = 2048 per update |
| batch size / epochs | 256 / 10 |
| learning rate | 3e-4 |
| γ / GAE λ | 0.995 / 0.95 |
| entropy coefficient | 0.01 |

## Episode

One episode is one complete workload, from the first arrival to the last completion. On `reset` the env samples a workload type, size (10–40 processes), switch cost (0 or 1) and seed. It then advances the engine to the first moment a process is ready. Each `step` dispatches the chosen process and runs the engine to the next decision point, skipping idle gaps. The episode is `terminated` when every process has finished. `truncated` is only a safety net and cannot fire: every step runs a process for at least one tick, so an episode needs at most total-CPU steps.

## Observation (`rl/spaces.py`)

A float32 vector of `10 × 10 + 9 = 109` values, each clipped to `[0, 10]`.

The 10 slots describe the **oldest** 10 ready processes in queue order. Using the oldest rather than, say, the shortest has two effects. A long-waiting process can never be hidden from the agent. And the agent must learn to compare jobs, rather than getting a pre-sorted list that hard-codes SJF.

Times are divided by the workload's mean CPU burst `S` and passed through `log1p`, which keeps heavy-tailed workloads in range.

| Per slot | Global |
|---|---|
| present flag | ready count ÷ 10 |
| log1p(remaining in current burst / S) | hidden ready processes ÷ 10 |
| log1p(total remaining CPU / S) | blocked, not-yet-arrived, finished fractions |
| log1p(current wait / S) | log1p(time to next event / S) |
| log1p(total wait / S) | switch cost ÷ S |
| log1p(time since arrival / S) | log1p(mean remaining burst of hidden processes / S) |
| priority ÷ max priority | log1p(max current wait among hidden processes / S) |
| never-run flag | |
| CPU bursts left ÷ 5 | |
| is-last-run flag (no switch cost) | |

## Action

`Discrete(10 × 3)`. Action `a` dispatches the process in slot `a // 3` for quantum `(2, 6, rest of burst)[a % 3]`. The agent therefore decides both *who* runs and *for how long*, and can preempt by choosing a short quantum. Actions pointing at empty slots are masked. If an invalid action is passed anyway, `step` raises instead of silently remapping it.

## Reward

The env tracks a cumulative cost `C(t)` and pays `r = −(C(t₁) − C(t₀)) / (N · S)` after each decision:

```
C = wait   × Σᵢ Wᵢ                         total waiting so far
  + fair   × Σ over READY stretches L² / (2S)  quadratic in each uninterrupted wait
  + resp   × Σᵢ Rᵢ                         response time (running total for unstarted processes)
  + switch × context_switches × S
```

Defaults: `wait=1, fair=0.1, resp=0.25, switch=0.1`.

**Why this shape.**

* **It telescopes.** The undiscounted return is exactly `−C(end)/(N·S)`, so the agent cannot game the reward by moving cost between steps. `tests/test_rl_env.py` checks this identity and recomputes the episode cost independently from the finished metrics.
* **Anti-starvation.** A pure waiting-time objective is minimised by SRTF-like policies that leave long jobs waiting indefinitely. The quadratic term makes one 100-tick wait cost 10× more than ten 10-tick waits, so starving a process is expensive.
* **Turnaround** is waiting plus fixed CPU and I/O work, so the waiting term covers it.
* **Throughput** only changes through idle time, which is set by arrivals because the agent must always dispatch, and through switch overhead, which the switch term covers.
* **Normalisation** by `N · S` puts episodes of different sizes and burst scales on comparable footing.

**Calibration.** `fair = 0.1` was chosen because it puts the classic baselines on the validation set within about 15% of each other (FCFS 23.5, SJF 20.3, RR 21.6, SRTF 21.9). At `fair = 0` SJF and SRTF win outright (8.4 vs FCFS 14.4). At `fair = 0.25` Round Robin wins (32.2 vs SJF 38.1). The default therefore leaves no textbook policy already optimal.

**Not included.** Process priority is visible to the agent but is not part of the objective, so the agent is not rewarded for respecting priorities.

## Data splits

| Split | Seeds | Workload types | Used for |
|---|---|---|---|
| train | sampled from 0–4,999,999 | cpu_bound, io_bound, short_jobs, mixed_priority, bursty | gradient updates |
| validation | 5,000,000+ (20 fixed traces, 30 processes) | the five training types | choosing `best.zip` |
| test | 10,000,000+ | all seven, including **long_jobs** and **mixed**, which training never sees | every reported result |

## Training cost

On a 2-core CPU (no GPU), the measured rate is about 800–1,000 environment steps per second, including validation every 20k steps.

| Steps | Wall time |
|---|---|
| 100k | ~2 min |
| 400k (`ppo-main`) | 8.4 min (504 s measured) |
| 1M | ~20 min |

Memory stays under 1 GB. Checkpoints are about 0.8 MB.

## Reproducibility

Workloads, validation sets, test sets and benchmarks are bit-for-bit reproducible from their seeds; the experiments page can re-run any benchmark and verify every row. PPO training is seeded (`--seed`). However, PyTorch does not guarantee identical floating-point results across CPUs, thread counts or library versions, so a retrained model may score slightly differently. The library versions used are recorded in each run's `metadata.json`.

## Honest limitations

* The agent optimises a *weighted* objective. It should not be expected to beat SRTF on average waiting or Round Robin on worst-case wait, and on held-out tests it does not. See `results.md`.
* Trace-to-trace variance is large, because workload types differ by an order of magnitude in load. Compare head-to-head counts on identical traces rather than raw means alone.
* An MLP over a fixed number of slots only sees aggregates of very long queues.
