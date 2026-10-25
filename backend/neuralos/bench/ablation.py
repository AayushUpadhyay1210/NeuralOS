"""Reward ablation: compare models trained with different reward weights.

Usage::

    python -m neuralos.bench.ablation ablation-fair0 ppo-main ablation-fair025

Every model is evaluated on the same test traces and scored with the same
*default* objective, plus the raw metrics, so models trained on different
objectives remain comparable. SJF and Round Robin are included as anchors.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

from ..paths import experiments_dir
from ..rl import registry
from ..rl.env import HELD_OUT_KINDS, TRAIN_KINDS
from ..rl.loader import load_scheduler
from .benchmark import TEST_SEED_BASE, BenchmarkConfig, run_benchmark, summarise, write_csv

METRICS = ("avg_waiting", "avg_response", "max_wait", "starvation_count", "jain_fairness",
           "context_switches", "objective")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_ids", nargs="+")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n-processes", type=int, default=30)
    a = ap.parse_args(argv)

    kinds = [*TRAIN_KINDS, *HELD_OUT_KINDS]
    seeds = [TEST_SEED_BASE + i for i in range(a.seeds)]
    table: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    anchors_done = False
    for run_id in a.run_ids:
        meta = registry.metadata(run_id) or {}
        schedulers = ["rl"] if anchors_done else ["sjf", "rr", "rl"]
        cfg = BenchmarkConfig(name=f"abl-{run_id}"[:40], kinds=kinds, seeds=seeds,
                              n_processes=a.n_processes, schedulers=schedulers, model_id=run_id)
        res = run_benchmark(cfg, lambda rid: load_scheduler(rid, "best"))
        for r in res["rows"]:
            name = run_id if r["scheduler"] == "rl" else r["scheduler"]
            raw.append({**r, "scheduler": name})
        anchors_done = True
        for g in summarise(res["rows"])["groups"]:
            if g["kind"] != "ALL" or (g["scheduler"] != "rl" and run_id != a.run_ids[0]):
                continue
            label = run_id if g["scheduler"] == "rl" else g["scheduler"]
            reward = meta.get("reward") if g["scheduler"] == "rl" else None
            row: dict[str, Any] = {"model": label, "train_reward": reward}
            for m in METRICS:
                row[m] = round(g[m]["mean"], 4)
                row[m + "_ci95"] = round(g[m]["ci95"], 4)
            table.append(row)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    d = experiments_dir() / f"ablation-{stamp}"
    d.mkdir(parents=True)
    (d / "ablation.json").write_text(json.dumps({"seeds": seeds, "kinds": kinds,
                                                 "n_processes": a.n_processes, "table": table},
                                                indent=2))
    write_csv(raw, d / "raw.csv")
    print(f"{'model':<20}" + "".join(f"{m[:13]:>15}" for m in METRICS))
    for row in table:
        print(f"{row['model']:<20}" + "".join(f"{row[m]:>15.3f}" for m in METRICS))
    print(f"\nScored with the default objective. Saved to {d}")


if __name__ == "__main__":
    main()
