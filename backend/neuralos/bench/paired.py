"""Paired comparison of one scheduler against the others on identical traces.

Usage::

    python -m neuralos.bench.paired <experiment_id> [--target rl]

For each baseline and metric it reports wins/losses on identical traces, an
exact two-sided sign test, and the mean paired difference (target - baseline)
with a normal-approximation 95% confidence interval. Paired statistics remove
the large trace-to-trace variance that dominates unpaired means.
"""

from __future__ import annotations

import argparse
import math
from statistics import mean, stdev
from typing import Any

from .benchmark import LOWER_IS_BETTER, load_experiment


def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail: float = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired(rows: list[dict[str, Any]], target: str, metric: str) -> list[dict[str, Any]]:
    tgt = {(r["kind"], r["seed"]): r for r in rows if r["scheduler"] == target}
    lower = LOWER_IS_BETTER.get(metric, True)
    out = []
    for base in sorted({r["scheduler"] for r in rows} - {target}):
        diffs = [float(tgt[(r["kind"], r["seed"])][metric]) - float(r[metric])
                 for r in rows if r["scheduler"] == base and (r["kind"], r["seed"]) in tgt]
        if not diffs:
            continue
        better = [d < -1e-9 if lower else d > 1e-9 for d in diffs]
        worse = [d > 1e-9 if lower else d < -1e-9 for d in diffs]
        m = mean(diffs)
        ci = 1.96 * stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
        out.append({"baseline": base, "metric": metric, "wins": sum(better),
                    "losses": sum(worse), "p_sign": sign_test(sum(better), sum(worse)),
                    "mean_diff": m, "ci_low": m - ci, "ci_high": m + ci, "n": len(diffs)})
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment_id")
    ap.add_argument("--target", default="rl")
    ap.add_argument("--metrics", nargs="*",
                    default=["objective", "avg_waiting", "avg_response", "max_wait"])
    a = ap.parse_args(argv)
    exp = load_experiment(a.experiment_id)
    if exp is None:
        raise SystemExit(f"experiment '{a.experiment_id}' not found")
    for metric in a.metrics:
        print(f"\n{metric} ({a.target} - baseline; "
              f"{'lower' if LOWER_IS_BETTER.get(metric, True) else 'higher'} is better)")
        print(f"  {'baseline':<11} {'wins/losses':>11} {'sign p':>9} {'mean diff':>10}  95% CI")
        for r in paired(exp["rows"], a.target, metric):
            print(f"  {r['baseline']:<11} {r['wins']:>5}/{r['losses']:<5} {r['p_sign']:>9.2g} "
                  f"{r['mean_diff']:>+10.2f}  [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]")


if __name__ == "__main__":
    main()
