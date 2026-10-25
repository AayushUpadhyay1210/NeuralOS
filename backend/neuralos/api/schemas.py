"""Pydantic request/response schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..sim.workload import MAX_BURST, MAX_PROCESSES

KindLiteral = Literal[
    "cpu_bound", "io_bound", "short_jobs", "long_jobs", "mixed_priority", "bursty", "mixed"
]
SchedulerName = Literal["fcfs", "sjf", "srtf", "rr", "priority", "priority_p", "random", "rl"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorkloadGenerate(StrictModel):
    kind: KindLiteral = "mixed"
    n_processes: int = Field(20, ge=1, le=MAX_PROCESSES)
    seed: int = Field(0, ge=0, le=2**31 - 1)
    mean_interarrival: float | None = Field(None, gt=0, le=1000)
    mean_cpu_burst: float | None = Field(None, ge=1, le=200)
    burst_distribution: Literal["exponential", "uniform", "lognormal"] | None = None
    priority_levels: int = Field(5, ge=1, le=20)
    io_fraction: float | None = Field(None, ge=0, le=1)
    mean_io_burst: float | None = Field(None, ge=1, le=200)


class ProcessIn(StrictModel):
    pid: int = Field(ge=0, le=100_000)
    arrival: int = Field(ge=0, le=1_000_000)
    bursts: list[int] = Field(min_length=1, max_length=41)
    priority: int = Field(0, ge=0, le=99)

    @model_validator(mode="after")
    def _check(self) -> ProcessIn:
        if len(self.bursts) % 2 == 0:
            raise ValueError("bursts must alternate CPU, I/O, ..., CPU (odd length)")
        if any(b < 1 or b > MAX_BURST for b in self.bursts):
            raise ValueError(f"each burst must be between 1 and {MAX_BURST}")
        return self


class WorkloadIn(StrictModel):
    """Either ``generate`` (seeded generator) or explicit ``processes``."""

    generate: WorkloadGenerate | None = None
    processes: list[ProcessIn] | None = Field(None, min_length=1, max_length=MAX_PROCESSES)

    @model_validator(mode="after")
    def _one_of(self) -> WorkloadIn:
        if (self.generate is None) == (self.processes is None):
            raise ValueError("provide exactly one of 'generate' or 'processes'")
        if self.processes is not None:
            pids = [p.pid for p in self.processes]
            if len(set(pids)) != len(pids):
                raise ValueError("process ids must be unique")
        return self


class SchedulerIn(StrictModel):
    name: SchedulerName = "fcfs"
    quantum: int = Field(4, ge=1, le=100)
    aging: int | None = Field(None, ge=1, le=1000)
    seed: int = Field(0, ge=0)
    model_id: str | None = Field(None, description="Run id; defaults to the loaded model")
    checkpoint: Literal["best", "final"] = "best"


class SimulationRequest(StrictModel):
    workload: WorkloadIn
    scheduler: SchedulerIn = Field(default_factory=lambda: SchedulerIn(name="fcfs"))
    context_switch: int = Field(0, ge=0, le=20)
    starvation_threshold: int | None = Field(None, ge=1, le=100_000)


class CompareRequest(StrictModel):
    workload: WorkloadIn
    schedulers: list[SchedulerIn] = Field(min_length=1, max_length=10)
    context_switch: int = Field(0, ge=0, le=20)


class TrainRequest(StrictModel):
    run_id: str | None = Field(None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    timesteps: int = Field(100_000, ge=1_000, le=1_000_000)
    seed: int = Field(0, ge=0, le=2**31 - 1)
    n_envs: int = Field(4, ge=1, le=8)
    eval_every: int = Field(20_000, ge=1_000, le=200_000)
    reward: dict[Literal["wait", "fair", "resp", "switch"], float] | None = None


class LoadModelRequest(StrictModel):
    checkpoint: Literal["best", "final"] = "best"


def _all_kinds() -> list[KindLiteral]:
    return ["cpu_bound", "io_bound", "short_jobs", "mixed_priority", "bursty", "long_jobs",
            "mixed"]


def _default_schedulers() -> list[SchedulerName]:
    return ["fcfs", "sjf", "srtf", "rr", "priority", "random"]


class BenchmarkRequest(StrictModel):
    name: str = Field("benchmark", pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,40}$")
    kinds: list[KindLiteral] = Field(default_factory=_all_kinds, min_length=1)
    seeds: list[int] | None = Field(None, min_length=1, max_length=50)
    n_seeds: int = Field(5, ge=1, le=50)
    n_processes: int = Field(30, ge=1, le=300)
    context_switch: int = Field(1, ge=0, le=20)
    rr_quantum: int = Field(4, ge=1, le=100)
    priority_aging: int | None = Field(None, ge=1, le=1000)
    schedulers: list[SchedulerName] = Field(default_factory=_default_schedulers, min_length=1)
    model_id: str | None = None
    model_checkpoint: Literal["best", "final"] = "best"


class EvaluateRequest(StrictModel):
    checkpoint: Literal["best", "final"] = "best"
    n_seeds: int = Field(5, ge=1, le=30)
    n_processes: int = Field(30, ge=1, le=300)
    context_switch: int = Field(1, ge=0, le=20)


class Message(BaseModel):
    detail: str


def as_dict(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(exclude_none=True)
