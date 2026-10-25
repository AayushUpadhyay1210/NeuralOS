"""Reproducible multi-seed benchmarks.

Every scheduler is run on the *same* generated traces: for each
``(kind, seed)`` the workload is generated once and handed to every scheduler.
Results are saved as raw per-run rows (JSON + CSV) plus a summary with mean,
standard deviation and a 95% confidence half-width.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..paths import experiments_dir, safe_id
from ..rl.spaces import RewardConfig, cumulative_cost, reward_normaliser
from ..sim.engine import SimConfig
from ..sim.runner import simulate
from ..sim.schedulers import BASELINES, Scheduler, make_baseline
from ..sim.workload import WORKLOAD_KINDS, WorkloadConfig, WorkloadKind, generate

TEST_SEED_BASE = 10_000_000  # mirrors neuralos.rl.env (kept here to avoid importing gym)

SUMMARY_METRICS = (
    "avg_waiting",
    "avg_turnaround",
    "avg_response",
    "throughput",
    "cpu_utilization",
    "context_switches",
    "starvation_count",
    "max_wait",
    "jain_fairness",
    "max_slowdown",
    "objective",
)
LOWER_IS_BETTER = {
    "avg_waiting": True,
    "avg_turnaround": True,
    "avg_response": True,
    "throughput": False,
    "cpu_utilization": False,
    "context_switches": True,
    "starvation_count": True,
    "max_wait": True,
    "jain_fairness": False,
    "max_slowdown": True,
    "objective": True,
}

SchedulerFactory = Callable[[str], Scheduler]


@dataclass
class BenchmarkConfig:
    name: str = "benchmark"
    kinds: list[WorkloadKind] = field(default_factory=lambda: list(WORKLOAD_KINDS))
    seeds: list[int] = field(default_factory=lambda: [TEST_SEED_BASE + i for i in range(10)])
    n_processes: int = 30
    context_switch: int = 1
    rr_quantum: int = 4
    priority_aging: int | None = None
    schedulers: list[str] = field(default_factory=lambda: list(BASELINES))
    model_id: str | None = None
    model_checkpoint: str = "best"
    reward: dict[str, float] = field(default_factory=lambda: RewardConfig().to_dict())

    def validate(self) -> None:
        safe_id(self.name)
        if not self.kinds:
            raise ValueError("choose at least one workload kind")
        bad = [k for k in self.kinds if k not in WORKLOAD_KINDS]
        if bad:
            raise ValueError(f"unknown workload kinds: {bad}")
        if not 1 <= len(self.seeds) <= 50:
            raise ValueError("between 1 and 50 seeds")
        if any(s < 0 for s in self.seeds):
            raise ValueError("seeds must be >= 0")
        if not 1 <= self.n_processes <= 300:
            raise ValueError("n_processes must be in [1, 300]")
        if not 0 <= self.context_switch <= 20:
            raise ValueError("context_switch must be in [0, 20]")
        if not self.schedulers:
            raise ValueError("choose at least one scheduler")
        unknown = [s for s in self.schedulers if s not in (*BASELINES, "rl")]
        if unknown:
            raise ValueError(f"unknown schedulers: {unknown}")
        if "rl" in self.schedulers and not self.model_id:
            raise ValueError("scheduler 'rl' needs model_id")
        runs = len(self.kinds) * len(self.seeds) * len(self.schedulers)
        if runs * self.n_processes > 200_000:
            raise ValueError("benchmark too large (kinds x seeds x schedulers x processes)")
        RewardConfig(**self.reward)

    def fingerprint(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def _stats(xs: list[float]) -> dict[str, float]:
    n = len(xs)
    mean = sum(xs) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": 1.96 * std / math.sqrt(n) if n > 1 else 0.0, "n": n}


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["kind"], r["scheduler"]), []).append(r)
        groups.setdefault(("ALL", r["scheduler"]), []).append(r)
    by_group = [
        {
            "kind": kind,
            "scheduler": sched,
            **{m: _stats([float(r[m]) for r in rs]) for m in SUMMARY_METRICS},
        }
        for (kind, sched), rs in sorted(groups.items())
    ]
    # Head-to-head: on how many identical traces does the RL agent beat each baseline?
    head: list[dict[str, Any]] = []
    rl_rows = {(r["kind"], r["seed"]): r for r in rows if r["scheduler"] == "rl"}
    if rl_rows:
        for sched in sorted({r["scheduler"] for r in rows} - {"rl"}):
            for metric in ("avg_waiting", "avg_response", "max_wait", "objective"):
                wins = ties = total = 0
                for r in rows:
                    if r["scheduler"] != sched:
                        continue
                    rl = rl_rows.get((r["kind"], r["seed"]))
                    if rl is None:
                        continue
                    total += 1
                    a, b = float(rl[metric]), float(r[metric])
                    if math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9):
                        ties += 1
                    elif a < b:
                        wins += 1
                head.append(
                    {"baseline": sched, "metric": metric, "rl_wins": wins, "ties": ties,
                     "rl_losses": total - wins - ties, "traces": total}
                )
    return {"groups": by_group, "head_to_head": head}


def run_benchmark(
    cfg: BenchmarkConfig,
    rl_factory: SchedulerFactory | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    cfg.validate()
    rc = RewardConfig(**cfg.reward)
    sim_cfg = SimConfig(context_switch=cfg.context_switch)
    schedulers: dict[str, Scheduler] = {}
    for name in cfg.schedulers:
        if name == "rl":
            if rl_factory is None:
                raise ValueError("scheduler 'rl' requested but no trained model was provided")
            schedulers[name] = rl_factory(cfg.model_id or "")
        else:
            schedulers[name] = make_baseline(
                name, quantum=cfg.rr_quantum, aging=cfg.priority_aging, seed=0
            )
    rows: list[dict[str, Any]] = []
    total = len(cfg.kinds) * len(cfg.seeds)
    done = 0
    t0 = time.perf_counter()
    for kind in cfg.kinds:
        for seed in cfg.seeds:
            specs = generate(WorkloadConfig(kind=kind, n_processes=cfg.n_processes, seed=seed))
            for name, sched in schedulers.items():
                sim = simulate(specs, sched, sim_cfg)
                from ..sim.metrics import compute_metrics

                m = compute_metrics(sim).to_dict()
                rows.append(
                    {
                        "kind": kind,
                        "seed": seed,
                        "scheduler": name,
                        **m,
                        "objective": cumulative_cost(sim, rc) / reward_normaliser(sim),
                    }
                )
            done += 1
            if progress:
                progress(done, total)
    return {
        "config": asdict(cfg),
        "fingerprint": cfg.fingerprint(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_s": round(time.perf_counter() - t0, 3),
        "rows": rows,
        "summary": summarise(rows),
    }


def save_experiment(result: dict[str, Any], exp_id: str | None = None) -> str:
    cfg = result["config"]
    if exp_id is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        exp_id = f"{cfg['name']}-{stamp}-{result['fingerprint'][:6]}"
    safe_id(exp_id)
    d = experiments_dir() / exp_id
    d.mkdir(parents=True, exist_ok=True)
    meta = {k: v for k, v in result.items() if k not in ("rows", "summary")}
    meta["id"] = exp_id
    (d / "experiment.json").write_text(json.dumps(meta, indent=2))
    (d / "summary.json").write_text(json.dumps(result["summary"], indent=2))
    (d / "raw.json").write_text(json.dumps(result["rows"], indent=1))
    write_csv(result["rows"], d / "raw.csv")
    return exp_id


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def load_experiment(exp_id: str) -> dict[str, Any] | None:
    d = experiments_dir() / safe_id(exp_id)
    if not (d / "experiment.json").exists():
        return None
    meta = json.loads((d / "experiment.json").read_text())
    meta["summary"] = json.loads((d / "summary.json").read_text())
    meta["rows"] = json.loads((d / "raw.json").read_text())
    return dict(meta)


def list_experiments() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for d in experiments_dir().iterdir():
        f = d / "experiment.json"
        if d.is_dir() and f.exists():
            out.append(json.loads(f.read_text()))
    out.sort(key=lambda m: str(m.get("created_at", "")), reverse=True)
    return out


def format_table(summary: dict[str, Any], kind: str = "ALL") -> str:
    cols = ("avg_waiting", "avg_response", "avg_turnaround", "max_wait", "starvation_count",
            "jain_fairness", "context_switches", "objective")
    lines = ["scheduler    " + " ".join(f"{c[:14]:>16}" for c in cols)]
    for g in summary["groups"]:
        if g["kind"] != kind:
            continue
        cells = " ".join(f"{g[c]['mean']:>9.2f} ±{g[c]['std']:>5.2f}" for c in cols)
        lines.append(f"{g['scheduler']:<12} {cells}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run a reproducible NeuralOS benchmark")
    ap.add_argument("--config", type=Path, help="JSON file with BenchmarkConfig fields")
    ap.add_argument("--name", default=None)
    ap.add_argument("--model", default=None, help="run id of a trained model (adds 'rl')")
    ap.add_argument("--checkpoint", default="best", choices=["best", "final"])
    ap.add_argument("--seeds", type=int, default=None, help="number of test seeds")
    ap.add_argument("--n-processes", type=int, default=None)
    ap.add_argument("--kinds", nargs="*", default=None)
    args = ap.parse_args(argv)

    data: dict[str, Any] = json.loads(args.config.read_text()) if args.config else {}
    if args.name:
        data["name"] = args.name
    if args.seeds:
        data["seeds"] = [TEST_SEED_BASE + i for i in range(args.seeds)]
    if args.n_processes:
        data["n_processes"] = args.n_processes
    if args.kinds:
        data["kinds"] = args.kinds
    cfg = BenchmarkConfig(**data)
    if args.model:
        cfg.model_id = args.model
        cfg.model_checkpoint = args.checkpoint
        if "rl" not in cfg.schedulers:
            cfg.schedulers.append("rl")

    factory: SchedulerFactory | None = None
    if "rl" in cfg.schedulers:
        from ..rl.loader import scheduler_factory

        factory = scheduler_factory(cfg.model_checkpoint)

    def show(done: int, total: int) -> None:
        print(f"\r  {done}/{total} traces", end="", flush=True)

    print(f"Benchmark '{cfg.name}': {len(cfg.kinds)} kinds x {len(cfg.seeds)} seeds x "
          f"{len(cfg.schedulers)} schedulers, {cfg.n_processes} processes each")
    result = run_benchmark(cfg, factory, show)
    print()
    exp_id = save_experiment(result)
    print(format_table(result["summary"]))
    print(f"\nSaved to {experiments_dir() / exp_id}")


if __name__ == "__main__":
    main()
