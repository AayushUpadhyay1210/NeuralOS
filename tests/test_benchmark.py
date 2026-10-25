"""Benchmark reproducibility and summary statistics."""

from __future__ import annotations

import csv
import json

import pytest

from neuralos.bench.benchmark import (
    BenchmarkConfig,
    load_experiment,
    run_benchmark,
    save_experiment,
    summarise,
)


def small(**kw) -> BenchmarkConfig:
    base = dict(name="t", kinds=["short_jobs", "bursty"], seeds=[1, 2, 3], n_processes=12,
                schedulers=["fcfs", "sjf", "rr"])
    base.update(kw)
    return BenchmarkConfig(**base)


def test_benchmark_is_deterministic():
    a = run_benchmark(small())
    b = run_benchmark(small())
    assert a["rows"] == b["rows"]
    assert a["fingerprint"] == b["fingerprint"]
    assert len(a["rows"]) == 2 * 3 * 3


def test_all_schedulers_see_identical_traces():
    rows = run_benchmark(small())["rows"]
    by_trace: dict[tuple, set] = {}
    for r in rows:
        # identical traces => identical total demand => same process count
        by_trace.setdefault((r["kind"], r["seed"]), set()).add(r["n_processes"])
    assert all(len(v) == 1 for v in by_trace.values())


def test_summary_statistics():
    rows = [
        {"kind": "k", "seed": s, "scheduler": "x", **{m: v for m in (
            "avg_waiting", "avg_turnaround", "avg_response", "throughput", "cpu_utilization",
            "context_switches", "starvation_count", "max_wait", "jain_fairness", "max_slowdown",
            "objective")}}
        for s, v in ((1, 2.0), (2, 4.0), (3, 6.0))
    ]
    g = next(x for x in summarise(rows)["groups"] if x["kind"] == "k")
    assert g["avg_waiting"]["mean"] == pytest.approx(4.0)
    assert g["avg_waiting"]["std"] == pytest.approx(2.0)  # sample std
    assert g["avg_waiting"]["ci95"] == pytest.approx(1.96 * 2.0 / 3**0.5)


def test_save_and_load_roundtrip(isolated_dirs):
    res = run_benchmark(small())
    exp_id = save_experiment(res)
    _, exps = isolated_dirs
    d = exps / exp_id
    assert {p.name for p in d.iterdir()} == {"experiment.json", "summary.json", "raw.json",
                                            "raw.csv"}
    with (d / "raw.csv").open() as f:
        assert len(list(csv.DictReader(f))) == len(res["rows"])
    loaded = load_experiment(exp_id)
    assert loaded["rows"] == json.loads(json.dumps(res["rows"]))
    # Re-running from the stored config reproduces every row exactly.
    again = run_benchmark(BenchmarkConfig(**loaded["config"]))
    assert again["rows"] == loaded["rows"]


@pytest.mark.parametrize("kw", [
    {"kinds": []}, {"kinds": ["nope"]}, {"seeds": []}, {"n_processes": 0},
    {"schedulers": ["lottery"]}, {"schedulers": ["rl"]}, {"name": "../x"},
    # 7 kinds x 50 seeds x 4 schedulers x 300 processes = 420k > the 200k limit
    {"kinds": ["cpu_bound", "io_bound", "short_jobs", "long_jobs", "mixed_priority", "bursty",
               "mixed"], "seeds": list(range(50)), "n_processes": 300,
     "schedulers": ["fcfs", "sjf", "rr", "srtf"]},
])
def test_validation(kw):
    with pytest.raises(ValueError):
        small(**kw).validate()


def test_rl_requires_a_model_factory():
    with pytest.raises(ValueError):
        run_benchmark(small(schedulers=["fcfs", "rl"], model_id="m"))


def test_sign_test_and_paired_differences():
    from neuralos.bench.paired import paired, sign_test

    assert sign_test(5, 5) == 1.0
    assert sign_test(10, 0) == pytest.approx(2 / 1024)
    rows = [
        {"kind": "k", "seed": s, "scheduler": name, "avg_waiting": v}
        for s, (a, b) in enumerate([(1, 3), (2, 2), (5, 4)])
        for name, v in (("rl", a), ("sjf", b))
    ]
    (r,) = paired(rows, "rl", "avg_waiting")
    assert (r["wins"], r["losses"], r["n"]) == (1, 1, 3)
    assert r["mean_diff"] == pytest.approx((-2 + 0 + 1) / 3)
