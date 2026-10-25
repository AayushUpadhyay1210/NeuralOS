# Results

Every number on this page was produced by the commands listed under each table, on the code in this repository, on 25 October 2026. Hardware: a 2-core x86-64 Linux VM with no GPU. Python 3.13.17, torch 2.14.1, gymnasium 1.4.0, stable-baselines3 / sb3-contrib 2.9.0.

## Setup

* **Model:** `ppo-main`, MaskablePPO, 400k environment steps (401,408 with the final rollout), seed 0, default reward (`wait 1, fair 0.1, resp 0.25, switch 0.1`). Training took 504 s. The checkpoint used is `best.zip`, which scored best on the validation set, at 400k steps.
* **Test traces:** seeds 10,000,000–10,000,009, which training and validation never used. All 7 workload types, 30 processes each, context-switch cost 1: 70 traces, and every scheduler ran on the identical 70.
* **Workload types** used in training are `cpu_bound`, `io_bound`, `short_jobs`, `mixed_priority` and `bursty`. **Held out entirely** from training are `long_jobs` and `mixed`.
* **Objective** is the RL episode cost (see `rl-design.md`), computed with identical weights for every scheduler. Lower is better.

Reproduce with:

```bash
python -m neuralos.rl.train --run-id ppo-main --timesteps 400000 --seed 0
python -m neuralos.bench.benchmark --config experiments/configs/main.json --model ppo-main
python -m neuralos.bench.paired <experiment-id>
```

Experiment `main-20261025-191229-b1d813` (config fingerprint `b1d81357abed`). The UI's Reproduce button re-runs it and confirms every row matches.

## All 70 test traces

Mean ± 95% confidence half-width across traces. These are unpaired intervals; trace-to-trace variance is large, so read them together with the paired tests below.

| Scheduler | Avg wait | Avg response | Avg turnaround | Worst wait | Starved | Fairness (Jain) | Switches | Objective |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FCFS | 87.1 ± 14.0 | 55.5 ± 10.2 | 107.4 ± 16.7 | 121.0 ± 19.6 | 11.4 ± 2.3 | 0.468 ± 0.034 | 48.6 ± 5.8 | 27.8 ± 7.8 |
| SJF | 51.7 ± 8.5 | 35.1 ± 5.6 | 72.0 ± 11.4 | 237.5 ± 38.9 | 5.9 ± 1.3 | 0.607 ± 0.026 | 48.4 ± 5.8 | 26.8 ± 10.1 |
| SRTF | 52.1 ± 9.0 | 30.4 ± 4.7 | 72.4 ± 11.7 | 257.8 ± 39.6 | 6.4 ± 1.4 | 0.636 ± 0.026 | 62.9 ± 7.4 | 32.0 ± 12.6 |
| Round Robin (q=4) | 110.0 ± 18.8 | 25.3 ± 2.6 | 130.3 ± 21.3 | 57.6 ± 5.0 | 7.0 ± 2.3 | 0.642 ± 0.025 | 109.2 ± 13.6 | 27.9 ± 8.0 |
| Priority | 82.7 ± 12.9 | 69.2 ± 11.3 | 103.0 ± 15.3 | 249.5 ± 39.7 | 9.6 ± 1.4 | 0.326 ± 0.031 | 48.0 ± 5.7 | 39.7 ± 12.0 |
| Priority (preemptive) | 87.4 ± 13.0 | 63.6 ± 10.1 | 107.7 ± 15.5 | 264.1 ± 41.4 | 10.0 ± 1.4 | 0.312 ± 0.032 | 59.6 ± 6.7 | 44.3 ± 13.8 |
| Random | 87.9 ± 14.5 | 59.4 ± 11.4 | 108.2 ± 17.1 | 246.8 ± 42.8 | 11.0 ± 1.8 | 0.361 ± 0.037 | 48.3 ± 5.8 | 34.2 ± 10.0 |
| **RL agent** | 69.2 ± 13.3 | 30.1 ± 4.6 | 89.5 ± 15.8 | 149.0 ± 28.3 | 6.7 ± 1.7 | 0.598 ± 0.030 | 71.1 ± 10.1 | **23.2 ± 7.6** |

## Paired comparison on identical traces

`RL − baseline` per trace. "Wins" counts traces where the agent was better. *p* is an exact two-sided sign test. The CI is a 95% normal interval on the mean paired difference.

**Objective** (the quantity the agent was trained on)

| vs | Wins / losses | Sign-test p | Mean diff [95% CI] |
|---|---:|---:|---|
| FCFS | 57 / 13 | 1e-07 | −4.6 [−6.4, −2.8] |
| SJF | 40 / 30 | 0.28 | −3.6 [−6.4, −0.8] |
| SRTF | 34 / 36 | 0.90 | −8.8 [−13.9, −3.6] |
| Round Robin | 66 / 4 | 2e-15 | −4.6 [−5.8, −3.5] |
| Priority | 67 / 3 | 1e-16 | −16.5 [−21.5, −11.5] |
| Priority (preemptive) | 70 / 0 | 2e-21 | −21.1 [−27.9, −14.3] |
| Random | 66 / 4 | 2e-15 | −11.0 [−14.0, −8.0] |

**Average waiting time**

| vs | Wins / losses | Sign-test p | Mean diff [95% CI] |
|---|---:|---:|---|
| FCFS | 53 / 17 | 2e-05 | −17.9 [−25.0, −10.7] |
| SJF | 15 / 55 | 2e-06 | **+17.5** [+11.2, +23.9] |
| SRTF | 4 / 66 | 2e-15 | **+17.2** [+12.1, +22.3] |
| Round Robin | 63 / 7 | 2e-12 | −40.7 [−52.9, −28.6] |

**Worst single wait**

| vs | Wins / losses | Sign-test p | Mean diff [95% CI] |
|---|---:|---:|---|
| FCFS | 22 / 48 | 0.003 | **+28.0** [+15.5, +40.4] |
| SJF | 63 / 7 | 2e-12 | −88.5 [−109.6, −67.5] |
| SRTF | 67 / 3 | 1e-16 | −108.8 [−131.5, −86.2] |
| Round Robin | 5 / 65 | 2e-14 | **+91.4** [+64.6, +118.1] |

### What this shows

1. **On its training objective, the agent has the lowest mean cost of all eight schedulers.** For every baseline, the paired mean difference favours it with a CI that excludes zero. It beats FCFS, Round Robin, both Priority variants and Random on the large majority of traces.
2. **Against SJF and SRTF it is not uniformly better.** It wins 40 of 70 and 34 of 70 traces respectively, and neither sign test is significant. Its lower mean cost comes from avoiding their bad cases: its worst waits are 88–109 ticks shorter on average.
3. **It is a compromise, not a dominant policy.** Its average waiting is 17.5 ticks worse than SJF/SRTF, and its worst-case wait is 91 ticks worse than Round Robin's. Each textbook policy still wins the single metric it is designed for: SJF/SRTF on average waiting, Round Robin on response and worst wait.
4. **It makes more context switches** (71 vs 48 for SJF), because it often picks short quanta. The switch penalty (0.1) tolerates that.

## Seen vs held-out workload types

Same 10 test seeds, split by whether the workload type was used in training.

**Types seen in training** (unseen seeds), 50 traces

| Scheduler | Avg wait | Avg response | Worst wait | Starved | Objective |
|---|---:|---:|---:|---:|---:|
| FCFS | 74.9 ± 15.9 | 44.0 ± 8.6 | 95.8 ± 15.8 | 11.9 ± 2.8 | 30.8 ± 10.5 |
| SJF | 45.8 ± 10.2 | 29.0 ± 4.8 | 185.7 ± 31.9 | 6.3 ± 1.6 | 31.7 ± 13.8 |
| SRTF | 47.5 ± 11.6 | 27.8 ± 5.3 | 203.1 ± 34.4 | 6.8 ± 1.8 | 38.2 ± 17.2 |
| Round Robin | 87.5 ± 17.7 | 24.6 ± 3.2 | 55.8 ± 6.3 | 7.4 ± 2.9 | 31.0 ± 10.9 |
| RL agent | 66.1 ± 17.6 | 25.7 ± 4.0 | 109.2 ± 17.9 | 7.7 ± 2.3 | 27.1 ± 10.3 |

**Types never seen in training** (`long_jobs`, `mixed`), 20 traces

| Scheduler | Avg wait | Avg response | Worst wait | Starved | Objective |
|---|---:|---:|---:|---:|---:|
| FCFS | 117.7 ± 24.6 | 84.3 ± 24.5 | 184.2 ± 46.0 | 10.2 ± 3.9 | 20.3 ± 6.5 |
| SJF | 66.5 ± 13.8 | 50.2 ± 13.9 | 367.0 ± 89.2 | 5.0 ± 1.6 | 14.7 ± 5.3 |
| SRTF | 63.5 ± 10.8 | 37.0 ± 9.3 | 394.7 ± 83.9 | 5.4 ± 1.7 | 16.4 ± 6.3 |
| Round Robin | 166.0 ± 39.7 | 27.2 ± 4.8 | 62.2 ± 7.3 | 6.0 ± 3.7 | 19.9 ± 5.2 |
| RL agent | 77.1 ± 15.1 | 41.1 ± 11.3 | 248.4 ± 73.0 | 4.3 ± 1.6 | 13.4 ± 4.6 |

On held-out types the agent keeps the lowest mean objective (13.4 vs SJF's 14.7). With 20 traces, though, the intervals overlap heavily, so this is evidence that the policy transfers, not proof that it is better there. Its worst-case wait grows sharply on these unfamiliar, heavier workloads (248 vs 109 on seen types).

## Reward ablation

Three models trained identically except for the fairness weight `fair`. All were scored on the same 70 test traces with the **default** objective weights, so the columns are comparable. SJF and Round Robin are shown as anchors.

```bash
python -m neuralos.bench.ablation ablation-fair0 ppo-main ablation-fair025
```

| Model (training `fair`) | Avg wait | Avg response | Worst wait | Starved | Jain | Switches | Objective (default) |
|---|---:|---:|---:|---:|---:|---:|---:|
| SJF | 51.7 | 35.1 | 237.5 | 5.9 | 0.607 | 48.4 | 26.8 |
| `ablation-fair0` (0) | 54.0 | 30.2 | 220.7 | 5.5 | 0.606 | 59.7 | 26.4 |
| `ppo-main` (0.1) | 69.2 | 30.1 | 149.0 | 6.7 | 0.598 | 71.1 | 23.2 |
| `ablation-fair025` (0.25) | 72.9 | 26.8 | 137.3 | 5.7 | 0.612 | 85.7 | 22.9 |
| Round Robin | 110.0 | 25.3 | 57.6 | 7.0 | 0.642 | 109.2 | 27.9 |

* **Without the fairness term**, the agent rediscovers SJF-like behaviour. Its average wait (54.0) is within about 2 ticks of SJF's, and so are its long worst-case waits.
* **Raising the fairness weight** shifts the agent monotonically towards fairness: worst wait drops from 221 to 149 to 137, while average wait rises from 54 to 69 to 73.
* The 0.25 model scores marginally better than `ppo-main` even on the default objective (22.9 vs 23.2). That difference is within run-to-run noise; it would take several training seeds per setting to separate them, and this ablation used one seed each.

## Training curve

Validation objective for `ppo-main` (20 fixed validation traces; lower is better). On the same set, the baselines score SJF 20.3, Round Robin 21.6, SRTF 21.9 and FCFS 23.5.

| Steps | 0 | 60k | 120k | 200k | 220k | 300k | 400k |
|---|---:|---:|---:|---:|---:|---:|---:|
| Objective | 28.9 | 21.8 | 21.2 | 19.8 | 18.9 | 21.6 | **17.6** |

The curve is noisy: PPO on a 20-trace validation set fluctuates by several points between evaluations. The final checkpoint, at 401k steps, scored 18.2. That is why the best checkpoint is kept rather than the last one. The untrained policy (28.9) is worse than every baseline; the agent first beats FCFS around 60k steps, and first reaches SJF's level around 200k.

## Longer training: `ppo-long`

A second run, `ppo-long`, used 1,000,000 steps with seed 1 and the same default reward. It reached a best validation objective of 14.8, at step 1,000,000, and was still improving. That beats `ppo-main`'s 17.6, so validation alone would choose `ppo-long`; the test results below played no part in that choice. Training took 2,080 s, but this machine was running other jobs at the time. At the measured unshared rate of about 900 steps/s, expect roughly 20 minutes on an idle 2-core CPU.

```bash
python -m neuralos.rl.train --run-id ppo-long --timesteps 1000000 --seed 1 --eval-every 25000
python -m neuralos.bench.benchmark --config experiments/configs/main.json --model ppo-long --name main-long
```

Same 70 test traces (experiment `main-long-20261025-192817-287170`):

| Scheduler | Avg wait | Avg response | Avg turnaround | Worst wait | Starved | Jain | Switches | Objective |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SJF | 51.7 | 35.1 | 72.0 | 237.5 | 5.9 | 0.607 | 48.4 | 26.8 |
| Round Robin | 110.0 | 25.3 | 130.3 | 57.6 | 7.0 | 0.642 | 109.2 | 27.9 |
| `ppo-main` (400k) | 69.2 | 30.1 | 89.5 | 149.0 | 6.7 | 0.598 | 71.1 | 23.2 |
| `ppo-long` (1M) | 60.0 | 25.9 | 80.3 | 155.2 | 6.1 | 0.620 | 65.6 | **20.1** |

Paired, `ppo-long` − baseline:

| vs | Objective: wins / losses (p), mean diff | Avg wait: wins / losses, mean diff | Worst wait: wins / losses, mean diff |
|---|---|---|---|
| FCFS | 67 / 3 (1e-16), −7.7 | 67 / 3, −27.2 | 11 / 56, **+34.1** |
| SJF | 46 / 24 (0.012), −6.8 | 15 / 55, **+8.3** | 57 / 13, −82.3 |
| SRTF | 47 / 23 (0.006), −11.9 | 9 / 61, **+7.9** | 65 / 5, −102.6 |
| Round Robin | 70 / 0 (2e-21), −7.8 | 70 / 0, −50.0 | 1 / 69, **+97.5** |

More training helped on every axis except worst-case wait, which held steady. With 1M steps the agent's objective advantage holds trace by trace against SJF and SRTF as well (sign tests p < 0.05), and its average-wait gap to SJF halves from +17.5 to +8.3 ticks. The overall shape of the result does not change: SJF/SRTF still minimise average waiting, and Round Robin still minimises worst-case waiting. The two runs also differ in seed as well as length, so this compares two single runs, not a controlled study of training length.


## Limitations of these results

* One training seed per configuration. Seed-to-seed spread for PPO was not measured, so small differences between models are not meaningful.
* All workloads are synthetic, with 30 processes per test trace; behaviour at other scales is untested.
* Burst lengths are known exactly, as the textbook SJF/SRTF assumption requires; the agent benefits from that too.
* The objective is a design choice. A different weighting would change the ranking, as the ablation shows.
