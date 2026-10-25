"""Seeded synthetic workload generation.

All randomness flows through a single ``numpy.random.Generator`` created from
the seed, so ``generate(cfg)`` returns the identical trace for the same config.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Literal

import numpy as np

from .process import ProcessSpec

WorkloadKind = Literal[
    "cpu_bound", "io_bound", "short_jobs", "long_jobs", "mixed_priority", "bursty", "mixed"
]
Distribution = Literal["exponential", "uniform", "lognormal"]

WORKLOAD_KINDS: tuple[WorkloadKind, ...] = (
    "cpu_bound",
    "io_bound",
    "short_jobs",
    "long_jobs",
    "mixed_priority",
    "bursty",
    "mixed",
)

MAX_PROCESSES = 500
MAX_BURST = 500


@dataclass(frozen=True)
class WorkloadConfig:
    kind: WorkloadKind = "mixed"
    n_processes: int = 20
    seed: int = 0
    # Optional overrides; ``None`` keeps the preset value for ``kind``.
    mean_interarrival: float | None = None
    mean_cpu_burst: float | None = None
    burst_distribution: Distribution | None = None
    priority_levels: int = 5
    io_fraction: float | None = None  # fraction of processes that do I/O
    mean_io_burst: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in WORKLOAD_KINDS:
            raise ValueError(f"unknown workload kind '{self.kind}'")
        if not 1 <= self.n_processes <= MAX_PROCESSES:
            raise ValueError(f"n_processes must be between 1 and {MAX_PROCESSES}")
        if self.mean_interarrival is not None and not 0 < self.mean_interarrival <= 1000:
            raise ValueError("mean_interarrival must be in (0, 1000]")
        if self.mean_cpu_burst is not None and not 1 <= self.mean_cpu_burst <= 200:
            raise ValueError("mean_cpu_burst must be in [1, 200]")
        if self.mean_io_burst is not None and not 1 <= self.mean_io_burst <= 200:
            raise ValueError("mean_io_burst must be in [1, 200]")
        if not 1 <= self.priority_levels <= 20:
            raise ValueError("priority_levels must be between 1 and 20")
        if self.io_fraction is not None and not 0.0 <= self.io_fraction <= 1.0:
            raise ValueError("io_fraction must be in [0, 1]")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _Preset:
    mean_interarrival: float
    mean_cpu_burst: float
    burst_distribution: Distribution
    io_fraction: float
    mean_io_burst: float
    long_fraction: float = 0.0  # bimodal mix: fraction of jobs drawn 6x longer
    bursty: bool = False


PRESETS: dict[WorkloadKind, _Preset] = {
    "cpu_bound": _Preset(11.0, 10.0, "exponential", 0.0, 8.0),
    "io_bound": _Preset(7.0, 3.0, "exponential", 1.0, 12.0),
    "short_jobs": _Preset(3.5, 3.0, "uniform", 0.0, 6.0),
    "long_jobs": _Preset(28.0, 25.0, "lognormal", 0.1, 15.0),
    "mixed_priority": _Preset(7.0, 4.0, "exponential", 0.2, 10.0, long_fraction=0.2),
    "bursty": _Preset(8.0, 5.0, "exponential", 0.2, 10.0, long_fraction=0.15, bursty=True),
    "mixed": _Preset(7.0, 4.0, "lognormal", 0.4, 10.0, long_fraction=0.15),
}


def _draw(rng: np.random.Generator, dist: Distribution, mean: float) -> int:
    if dist == "exponential":
        x = rng.exponential(mean)
    elif dist == "uniform":
        x = rng.uniform(1.0, 2.0 * mean - 1.0) if mean > 1 else 1.0
    else:  # lognormal with sigma 0.8, scaled so the mean matches
        sigma = 0.8
        mu = np.log(mean) - sigma * sigma / 2
        x = rng.lognormal(mu, sigma)
    return int(min(MAX_BURST, max(1, round(float(x)))))


def resolved(cfg: WorkloadConfig) -> WorkloadConfig:
    """Return ``cfg`` with every ``None`` override filled from its preset."""
    p = PRESETS[cfg.kind]
    return replace(
        cfg,
        mean_interarrival=cfg.mean_interarrival or p.mean_interarrival,
        mean_cpu_burst=cfg.mean_cpu_burst or p.mean_cpu_burst,
        burst_distribution=cfg.burst_distribution or p.burst_distribution,
        io_fraction=p.io_fraction if cfg.io_fraction is None else cfg.io_fraction,
        mean_io_burst=cfg.mean_io_burst or p.mean_io_burst,
    )


def generate(cfg: WorkloadConfig) -> list[ProcessSpec]:
    preset = PRESETS[cfg.kind]
    r = resolved(cfg)
    assert r.mean_interarrival is not None and r.mean_cpu_burst is not None
    assert r.burst_distribution is not None and r.io_fraction is not None
    assert r.mean_io_burst is not None
    rng = np.random.default_rng(cfg.seed)

    # Arrival times
    arrivals: list[int] = []
    t = 0.0
    if preset.bursty:
        # Clusters of 3-8 near-simultaneous arrivals separated by long gaps.
        while len(arrivals) < cfg.n_processes:
            size = int(rng.integers(3, 9))
            for _ in range(size):
                arrivals.append(round(t + rng.uniform(0, 2)))
            t += rng.exponential(r.mean_interarrival * size)
        arrivals = sorted(arrivals[: cfg.n_processes])
    else:
        for i in range(cfg.n_processes):
            if i > 0:
                t += rng.exponential(r.mean_interarrival)
            arrivals.append(round(t))

    specs: list[ProcessSpec] = []
    for pid, arrival in enumerate(arrivals):
        long_job = preset.long_fraction > 0 and rng.random() < preset.long_fraction
        mean = r.mean_cpu_burst * (6.0 if long_job else 1.0)
        does_io = rng.random() < r.io_fraction
        if does_io:
            n_cpu = int(rng.integers(2, 6))
            bursts: list[int] = []
            for j in range(n_cpu):
                bursts.append(_draw(rng, r.burst_distribution, mean / 2 if long_job else mean))
                if j < n_cpu - 1:
                    bursts.append(_draw(rng, "exponential", r.mean_io_burst))
        else:
            bursts = [_draw(rng, r.burst_distribution, mean)]
        if cfg.kind == "mixed_priority":
            # Long jobs skew towards low priority (high number) - a classic
            # setup where pure priority scheduling starves them.
            hi = cfg.priority_levels
            prio = int(rng.integers(hi // 2, hi)) if long_job else int(rng.integers(0, hi))
        else:
            prio = int(rng.integers(0, cfg.priority_levels))
        specs.append(ProcessSpec(pid=pid, arrival=arrival, bursts=tuple(bursts), priority=prio))
    return specs


def workload_summary(specs: list[ProcessSpec]) -> dict[str, float | int]:
    cpu = [b for s in specs for b in s.cpu_bursts]
    io = [b for s in specs for b in s.io_bursts]
    total_cpu = sum(s.total_cpu for s in specs)
    span = max(1, max(s.arrival for s in specs) - min(s.arrival for s in specs))
    return {
        "n_processes": len(specs),
        "total_cpu": total_cpu,
        "total_io": sum(io),
        "mean_cpu_burst": sum(cpu) / len(cpu),
        "max_cpu_burst": max(cpu),
        "io_processes": sum(1 for s in specs if len(s.bursts) > 1),
        "arrival_span": span,
        "offered_load": total_cpu / span,
    }
