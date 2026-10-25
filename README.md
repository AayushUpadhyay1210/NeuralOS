# NeuralOS

An educational CPU-scheduling lab. It generates seeded process workloads, simulates them with a deterministic discrete-event engine, and compares classic schedulers (FCFS, SJF, SRTF, Round Robin, Priority) against a reinforcement-learning policy trained with action-masked PPO. A React dashboard shows Gantt traces, queue states, training curves and multi-seed benchmarks.

NeuralOS simulates scheduling. It is not an operating-system kernel.

```
backend/      Python package `neuralos`: sim/ (engine, schedulers, workloads, metrics),
              rl/ (Gymnasium env, training, checkpoints), bench/ (benchmarks), api/ (FastAPI)
frontend/     React + TypeScript + Vite + Tailwind dashboard
tests/        pytest: scheduler correctness, engine invariants, RL env, training, API, benchmarks
models/       trained checkpoints (git-ignored; see models/README.md)
experiments/  benchmark configs (configs/) and saved results (results/, git-ignored)
docs/         architecture.md, algorithms.md, rl-design.md, results.md
scripts/      setup and check scripts for Linux/macOS (.sh) and Windows (.ps1)
```

## Requirements

* Python 3.10 or newer (verified on 3.13.17)
* Node.js 20 or newer (verified on 22.23)
* No GPU. Everything, including training, runs on CPU.

## Setup

**Linux / macOS**

```bash
bash scripts/setup.sh            # add --no-rl to skip PyTorch; baselines and UI still work
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1     # add -NoRL to skip PyTorch
.\.venv\Scripts\Activate.ps1
```

<details>
<summary>The same steps by hand</summary>

```bash
python3 -m venv .venv && source .venv/bin/activate        # Windows: py -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu   # Linux only: CPU wheel, skips ~3 GB of CUDA
pip install -e "backend[rl,dev]"                          # or: backend[dev] without RL
cd frontend && npm ci && cd ..
```
</details>

## Run

Use two terminals, both from the repository root with the virtual environment active.

```bash
# 1. API on http://127.0.0.1:8000  (interactive docs at /docs)
uvicorn neuralos.api.app:app --port 8000

# 2. Dashboard on http://localhost:5173  (proxies /api to port 8000)
cd frontend && npm run dev
```

PowerShell uses the same commands. Set `NEURALOS_API` before `npm run dev` if the API runs elsewhere.

## Train, evaluate, benchmark

```bash
# Train the reference model (about 8 minutes on a 2-core CPU)
python -m neuralos.rl.train --run-id ppo-main --timesteps 400000 --seed 0

# Evaluate a checkpoint against every baseline on held-out test seeds
python -m neuralos.rl.evaluate --run-id ppo-main --seeds 10

# Reproducible benchmark from a config file; --model adds the RL agent
python -m neuralos.bench.benchmark --config experiments/configs/main.json --model ppo-main

# Reward ablation: score several models on the same traces and objective
python -m neuralos.rl.train --run-id ablation-fair0   --timesteps 400000 --reward-fair 0
python -m neuralos.rl.train --run-id ablation-fair025 --timesteps 400000 --reward-fair 0.25
python -m neuralos.bench.ablation ablation-fair0 ppo-main ablation-fair025
```

Training writes `models/<run_id>/` (checkpoints, metadata, live progress, log). Benchmarks write `experiments/results/<id>/` (raw rows as CSV and JSON, plus a summary). You can also train, load checkpoints, evaluate and run benchmarks from the dashboard.

Until a checkpoint is loaded, the RL agent is disabled in the UI and `rl` requests return HTTP 409. Nothing is ever substituted for it.

## Tests and checks

```bash
bash scripts/check.sh                        # Windows: powershell -File scripts\check.ps1
```

This runs, in order:

* `pytest`: 186 tests, including small real training runs (about 1 minute). Use `pytest -m "not slow"` to skip them.
* `ruff check backend tests` and `mypy` (strict).
* In the frontend: `npm run typecheck`, `npm run lint`, `npm test` (Vitest) and `npm run build`.

## Results

These are measured on held-out test seeds, which training never saw, with 30 processes per trace and a context-switch cost of 1. Full tables and the ablation are in [docs/results.md](docs/results.md).

| Scheduler | Avg wait | Avg response | Worst wait | Fairness (Jain) | Objective |
|---|---:|---:|---:|---:|---:|
| FCFS | 87.1 | 55.5 | 121.0 | 0.468 | 27.8 |
| SJF | **51.7** | 35.1 | 237.5 | 0.607 | 26.8 |
| SRTF | 52.1 | 30.4 | 257.8 | 0.636 | 32.0 |
| Round Robin (q=4) | 110.0 | **25.3** | **57.6** | **0.642** | 27.9 |
| Priority | 82.7 | 69.2 | 249.5 | 0.326 | 39.7 |
| RL agent (`ppo-main`, 400k steps) | 69.2 | 30.1 | 149.0 | 0.598 | **23.2** |

Means over 70 traces (7 workload types × 10 seeds). The objective is the weighted cost the agent was trained on, combining waiting, a quadratic long-wait penalty, response time and context switches. Lower is better for every column except fairness.

* On paired traces, the agent's mean objective is lower than every baseline's, with a 95% CI that excludes zero. It wins 57–70 of 70 traces against FCFS, Round Robin, Priority and Random.
* Against SJF and SRTF it wins only 40 and 34 of 70 traces. Its advantage over them comes from much shorter worst-case waits, not from being better everywhere.
* It does **not** beat SJF/SRTF on average waiting (+17.5 ticks), nor Round Robin on worst-case wait (+91 ticks). It is a trade-off policy, and the reward ablation shows the fairness weight moves it predictably between those extremes.
* Training longer helps. A 1M-step run (`ppo-long`, about 20 minutes) lowers the objective to 20.1 and average waiting to 60.0, and it wins on the objective trace by trace against SJF (46/70) and SRTF (47/70). It still does not beat SJF on average waiting or Round Robin on worst-case wait.

## How it works

* [docs/algorithms.md](docs/algorithms.md): time model, tie-breaking rules, context switches, every scheduler, workloads, metric definitions.
* [docs/rl-design.md](docs/rl-design.md): observation, action masking, reward and why it resists starvation, data splits, training cost.
* [docs/architecture.md](docs/architecture.md): module boundaries, API endpoints, background training, model-state handling.

## Compatibility notes

* Linux: `pip install torch` from PyPI pulls CUDA libraries (about 3 GB) even though the CPU is used. The setup script installs the CPU-only wheel first.
* Windows and macOS: the default PyPI torch wheel is CPU-only.
* The core simulator, baselines, benchmarks and API need only numpy, FastAPI and Pydantic. Without the RL extras, the API reports `dependencies_missing` and every other feature works.
* Benchmarks are bit-for-bit reproducible. Retraining with the same seed gives a very similar model, but not necessarily a bit-identical one: PyTorch floating-point results can differ across CPUs and thread counts.
