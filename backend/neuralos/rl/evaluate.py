"""Evaluate a saved checkpoint against every baseline on held-out test seeds.

Usage::

    python -m neuralos.rl.evaluate --run-id ppo-main [--checkpoint best] [--seeds 10]

Reports in-distribution workload types (seen in training, unseen seeds) and
out-of-distribution types (never seen in training) separately, saves the run
as an experiment, and writes ``models/<run_id>/evaluation.json``.
"""

from __future__ import annotations

import argparse
from typing import Any

from ..bench.benchmark import (
    TEST_SEED_BASE,
    BenchmarkConfig,
    format_table,
    run_benchmark,
    save_experiment,
    summarise,
)
from ..paths import experiments_dir
from . import registry
from .env import HELD_OUT_KINDS, TRAIN_KINDS
from .loader import load_scheduler

BASELINE_SET = ["fcfs", "sjf", "srtf", "rr", "priority", "priority_p", "random"]


def evaluate(run_id: str, checkpoint: str = "best", n_seeds: int = 10, n_processes: int = 30,
             context_switch: int = 1) -> dict[str, Any]:
    load_scheduler(run_id, checkpoint)  # fail fast with a readable error
    cfg = BenchmarkConfig(
        name=f"eval-{run_id}"[:40],
        kinds=[*TRAIN_KINDS, *HELD_OUT_KINDS],
        seeds=[TEST_SEED_BASE + i for i in range(n_seeds)],
        n_processes=n_processes,
        context_switch=context_switch,
        schedulers=[*BASELINE_SET, "rl"],
        model_id=run_id,
        model_checkpoint=checkpoint,
    )
    result = run_benchmark(cfg, lambda rid: load_scheduler(rid, checkpoint))
    exp_id = save_experiment(result)
    rows = result["rows"]
    seen = summarise([r for r in rows if r["kind"] in TRAIN_KINDS])
    held = summarise([r for r in rows if r["kind"] in HELD_OUT_KINDS])
    out = {"experiment_id": exp_id, "checkpoint": checkpoint, "seen_types": seen,
           "held_out_types": held}
    registry.write_json(registry.run_dir(run_id) / "evaluation.json", out)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate a trained NeuralOS model on test seeds")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--checkpoint", default="best", choices=["best", "final"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n-processes", type=int, default=30)
    ap.add_argument("--context-switch", type=int, default=1)
    a = ap.parse_args(argv)
    out = evaluate(a.run_id, a.checkpoint, a.seeds, a.n_processes, a.context_switch)
    print(f"\nWorkload types seen in training ({', '.join(TRAIN_KINDS)}), unseen test seeds:")
    print(format_table(out["seen_types"]))
    print(f"\nHeld-out workload types ({', '.join(HELD_OUT_KINDS)}):")
    print(format_table(out["held_out_types"]))
    print("\nHead to head (RL wins / ties / losses on identical traces, all types):")
    from ..bench.benchmark import load_experiment

    exp = load_experiment(out["experiment_id"]) or {}
    for h in exp.get("summary", {}).get("head_to_head", []):
        if h["metric"] in ("objective", "avg_waiting"):
            print(f"  vs {h['baseline']:<11} {h['metric']:<12} "
                  f"{h['rl_wins']:>3} / {h['ties']:>2} / {h['rl_losses']:>3}")
    print(f"\nSaved experiment {out['experiment_id']} in {experiments_dir()}")


if __name__ == "__main__":
    main()
