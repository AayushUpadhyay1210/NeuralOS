from __future__ import annotations

import itertools

import pytest

from neuralos.sim import WORKLOAD_KINDS, WorkloadConfig, generate
from neuralos.sim.workload import workload_summary


@pytest.mark.parametrize("kind", WORKLOAD_KINDS)
def test_seeded_generation_is_reproducible(kind):
    a = generate(WorkloadConfig(kind=kind, n_processes=40, seed=123))
    b = generate(WorkloadConfig(kind=kind, n_processes=40, seed=123))
    c = generate(WorkloadConfig(kind=kind, n_processes=40, seed=124))
    assert a == b
    assert a != c
    assert len(a) == 40
    assert [s.pid for s in a] == list(range(40))
    assert all(s.arrival >= 0 for s in a)
    assert sorted(s.arrival for s in a) == [s.arrival for s in a]


def test_kinds_have_expected_character():
    io = workload_summary(generate(WorkloadConfig(kind="io_bound", n_processes=100, seed=1)))
    cpu = workload_summary(generate(WorkloadConfig(kind="cpu_bound", n_processes=100, seed=1)))
    short = workload_summary(generate(WorkloadConfig(kind="short_jobs", n_processes=100, seed=1)))
    long = workload_summary(generate(WorkloadConfig(kind="long_jobs", n_processes=100, seed=1)))
    assert io["io_processes"] == 100 and cpu["io_processes"] == 0
    assert long["mean_cpu_burst"] > 3 * short["mean_cpu_burst"]


def test_bursty_arrivals_cluster():
    specs = generate(WorkloadConfig(kind="bursty", n_processes=60, seed=5))
    arrivals = [s.arrival for s in specs]
    gaps = [b - a for a, b in itertools.pairwise(arrivals)]
    # Most gaps are tiny (inside clusters) and a few are large.
    assert sum(1 for g in gaps if g <= 2) > len(gaps) * 0.6
    assert max(gaps) > 10


def test_overrides_apply():
    specs = generate(
        WorkloadConfig(kind="cpu_bound", n_processes=200, seed=2, mean_cpu_burst=50,
                       burst_distribution="uniform")
    )
    mean = sum(s.total_cpu for s in specs) / len(specs)
    assert 40 < mean < 60


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_processes": 0},
        {"n_processes": 10_000},
        {"mean_cpu_burst": 0},
        {"io_fraction": 2.0},
        {"seed": -1},
        {"kind": "nope"},
    ],
)
def test_validation(kwargs):
    with pytest.raises(ValueError):
        WorkloadConfig(**kwargs)
