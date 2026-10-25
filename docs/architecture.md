# Architecture

NeuralOS is a simulator, not an operating-system kernel. It has three layers, and every one of them uses the same simulation engine.

```
frontend/ (React)  ──HTTP /api──▶  backend/neuralos/api  (FastAPI)
                                          │
              ┌───────────────────────────┼─────────────────────────────┐
              ▼                           ▼                             ▼
     neuralos/sim  ◀──────────── neuralos/rl ──────────────▶  neuralos/bench
     engine, schedulers,          Gymnasium env, MaskablePPO     multi-seed benchmarks,
     workloads, metrics           training, checkpoints          ablations, experiments
     (numpy only)                 (torch, SB3; lazy imports)     (no ML needed for baselines)
```

## Module boundaries

| Package | Depends on | Responsibility |
|---|---|---|
| `neuralos.sim` | numpy | Process model, discrete-event engine, baseline schedulers, workload generator, metrics. Has no web or ML imports. |
| `neuralos.rl.spaces` | `sim`, numpy | Observation builder, action decoding, action masks, reward cost. Shared by the training env and inference. |
| `neuralos.rl.env` | `spaces`, gymnasium | `SchedulingEnv`, the Gymnasium wrapper around `sim.Simulator`. |
| `neuralos.rl.train` / `evaluate` / `loader` / `policy` | SB3, sb3-contrib, torch | Training with validation-based checkpoint selection; loading a checkpoint as a `Scheduler`. |
| `neuralos.rl.registry` | stdlib | Reads and writes `models/<run_id>/`. Imports no ML libraries, so the API can list models even without torch. |
| `neuralos.bench` | `sim`, `rl.spaces` | Runs every scheduler on identical traces across seeds; saves raw rows and summaries. |
| `neuralos.api` | all of the above | Pydantic schemas, routes, background training process, in-memory simulation cache. |

There is one simulation engine. Baselines call `Simulator.dispatch` through `sim.runner.run`; the RL environment calls the same `dispatch` from `step()`; the trained policy is wrapped as an ordinary `Scheduler` (`rl.policy.RLScheduler`) and runs through `sim.runner.run` too. No scheduling logic is duplicated.

## Scheduler interface

```python
class Scheduler:
    preemptive: bool
    def reset(self) -> None
    def attach(self, sim: Simulator) -> None             # RL policy needs global state
    def select(self, ready, now) -> Decision(pid, quantum)  # quantum None = run the whole burst
    def should_preempt(self, running, ready, now) -> bool  # only called if preemptive
```

## Data flow for a simulation request

1. `POST /api/simulations` validates the body with Pydantic (`api/schemas.py`).
2. The workload is generated from its seed (`sim.workload.generate`) or built from explicit processes.
3. `sim.runner.run` drives the scheduler to completion and returns the timeline, transition log, per-process stats and metrics.
4. The result is cached in memory under an id (`GET /api/simulations/{id}`, `/timeline`).
5. The frontend replays the transition log client-side for play/pause/step. Playback moves a cursor through recorded results; it never recomputes or invents states.

## Long-running work

* **Training** runs in a separate process started with the `spawn` multiprocessing context, so PyTorch never blocks the API's event loop or GIL. Only one training job runs at a time. Progress is written atomically to `models/<run_id>/progress.json` and read back by `GET /api/training/{run_id}`. Cancellation creates `models/<run_id>/CANCEL`; the training callback checks for it, stops cleanly and saves `final.zip`.
* **Simulations and benchmarks** are CPU-bound but short. They are plain `def` endpoints, so FastAPI runs them in its worker thread pool. Benchmarks are limited to 200,000 process-runs (kinds × seeds × schedulers × processes).
* The API exposes no shell execution. All ids used as directory names are validated against `^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$` (`paths.safe_id`).

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Status, version, whether RL packages are installed |
| GET | `/api/config` | Schedulers, workload presets, limits, reward and action-space defaults, seed ranges |
| POST | `/api/workloads/generate` | Seeded workload |
| POST | `/api/simulations` | Run one scheduler on one workload |
| GET | `/api/simulations/{id}` and `/timeline` | Retrieve a stored result |
| POST | `/api/compare` | Several schedulers on one identical workload |
| GET | `/api/models`, `/api/models/status`, `/api/models/{id}` | Checkpoints, active model, training state |
| POST | `/api/models/{id}/load`, `/api/models/unload` | Choose the checkpoint used for `rl` |
| POST | `/api/models/{id}/evaluate` | Benchmark a checkpoint against baselines on test seeds |
| POST | `/api/training`, GET `/api/training/{id}`, POST `/api/training/{id}/cancel` | Background training |
| POST/GET | `/api/experiments`, `/api/experiments/{id}` | Run and read benchmarks |
| GET | `/api/experiments/{id}/export?format=csv\|json` | Raw rows |
| POST | `/api/experiments/{id}/reproduce` | Re-run from stored seeds and compare row by row |

Interactive documentation is at `http://127.0.0.1:8000/docs` while the API is running.

Error codes: 422 for invalid input, 404 for unknown ids, 409 when a model is required but unavailable or a training job is already running, and 500 (with the exception type) for unexpected failures.

## Model state in the UI

`GET /api/models/status` returns one of:

* `loaded`: a checkpoint is in memory and used for `rl` requests;
* `training`: a run is in progress and no checkpoint is loaded;
* `unavailable`: nothing loaded, so `rl` requests return 409;
* `dependencies_missing`: torch / gymnasium / SB3 are not installed.

The UI disables the RL agent everywhere unless the state is `loaded`. Nothing is ever substituted for the agent.
