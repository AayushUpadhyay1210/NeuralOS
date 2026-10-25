"""HTTP routes. Handlers are plain ``def`` where they do CPU work, so FastAPI runs
them in its worker thread pool instead of blocking the event loop. Training
runs in a separate process (see ``state.TrainingManager``)."""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from .. import __version__
from ..bench import benchmark as bench
from ..rl import registry
from ..rl.loader import ModelUnavailable
from ..rl.spaces import RewardConfig, SpaceConfig
from ..sim.engine import SimConfig
from ..sim.process import ProcessSpec
from ..sim.runner import run
from ..sim.schedulers import BASELINES, Scheduler, make_baseline
from ..sim.workload import (
    MAX_BURST,
    MAX_PROCESSES,
    PRESETS,
    WORKLOAD_KINDS,
    WorkloadConfig,
    generate,
    resolved,
    workload_summary,
)
from . import schemas as S
from .state import models, simulations, training

router = APIRouter(prefix="/api")

SCHEDULER_INFO = {
    "fcfs": "First Come First Served (non-preemptive)",
    "sjf": "Shortest Job First on the current CPU burst (non-preemptive)",
    "srtf": "Shortest Remaining Time First (preemptive)",
    "rr": "Round Robin with configurable quantum",
    "priority": "Priority, non-preemptive (lower number = higher priority)",
    "priority_p": "Priority, preemptive",
    "random": "Seeded uniform random choice (sanity floor)",
    "rl": "Trained MaskablePPO policy loaded from a checkpoint",
}


# ------------------------------------------------------------------ helpers
def _specs(w: S.WorkloadIn) -> tuple[list[ProcessSpec], dict[str, Any] | None]:
    if w.generate is not None:
        cfg = WorkloadConfig(**w.generate.model_dump())
        return generate(cfg), resolved(cfg).to_dict()
    assert w.processes is not None
    return [ProcessSpec(p.pid, p.arrival, tuple(p.bursts), p.priority) for p in w.processes], None


def _scheduler(s: S.SchedulerIn) -> Scheduler:
    if s.name == "rl":
        return models.scheduler(s.model_id, s.checkpoint)
    return make_baseline(s.name, quantum=s.quantum, aging=s.aging, seed=s.seed)


def _spec_dicts(specs: list[ProcessSpec]) -> list[dict[str, Any]]:
    return [
        {"pid": s.pid, "arrival": s.arrival, "bursts": list(s.bursts), "priority": s.priority,
         "total_cpu": s.total_cpu, "total_io": s.total_io}
        for s in specs
    ]


# ------------------------------------------------------------- health/config
@router.get("/health")
def health() -> dict[str, Any]:
    missing = registry.missing_rl_dependencies()
    return {"status": "ok", "version": __version__, "rl_available": not missing,
            "missing_dependencies": missing}


@router.get("/config")
def config() -> dict[str, Any]:
    return {
        "schedulers": [{"name": n, "description": SCHEDULER_INFO[n]} for n in (*BASELINES, "rl")],
        "workload_kinds": [
            {"name": k, **{f: getattr(PRESETS[k], f) for f in
                           ("mean_interarrival", "mean_cpu_burst", "burst_distribution",
                            "io_fraction", "mean_io_burst", "long_fraction", "bursty")}}
            for k in WORKLOAD_KINDS
        ],
        "limits": {"max_processes": MAX_PROCESSES, "max_burst": MAX_BURST,
                   "max_train_timesteps": 1_000_000, "max_context_switch": 20},
        "reward_defaults": RewardConfig().to_dict(),
        "space_defaults": SpaceConfig().to_dict(),
        "seed_ranges": {"train": [0, 4_999_999], "validation": 5_000_000,
                        "test": bench.TEST_SEED_BASE},
    }


# --------------------------------------------------------------- workloads
@router.post("/workloads/generate")
def generate_workload(req: S.WorkloadGenerate) -> dict[str, Any]:
    cfg = WorkloadConfig(**req.model_dump())
    specs = generate(cfg)
    return {"config": resolved(cfg).to_dict(), "processes": _spec_dicts(specs),
            "summary": workload_summary(specs)}


# ------------------------------------------------------------- simulations
@router.post("/simulations")
def create_simulation(req: S.SimulationRequest) -> dict[str, Any]:
    specs, wcfg = _specs(req.workload)
    sched = _scheduler(req.scheduler)
    result = run(specs, sched, SimConfig(context_switch=req.context_switch,
                                         starvation_threshold=req.starvation_threshold))
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "workload": {"config": wcfg, "processes": _spec_dicts(specs),
                     "summary": workload_summary(specs)},
        "request": req.model_dump(),
        **result.to_dict(),
    }
    sid = simulations.put(payload)
    return {"id": sid, **payload}


@router.get("/simulations/{sid}")
def get_simulation(sid: str) -> dict[str, Any]:
    data = simulations.get(sid)
    if data is None:
        raise HTTPException(404, f"simulation '{sid}' not found (results are kept in memory)")
    return {"id": sid, **data}


@router.get("/simulations/{sid}/timeline")
def get_timeline(sid: str) -> dict[str, Any]:
    data = get_simulation(sid)
    return {"id": sid, "timeline": data["timeline"], "transitions": data["transitions"]}


@router.post("/compare")
def compare(req: S.CompareRequest) -> dict[str, Any]:
    specs, wcfg = _specs(req.workload)
    cfg = SimConfig(context_switch=req.context_switch)
    results = []
    for s in req.schedulers:
        r = run(specs, _scheduler(s), cfg)
        results.append({"scheduler": r.scheduler, "metrics": r.metrics.to_dict(),
                        "timeline": [vars(x) for x in r.timeline]})
    return {"workload": {"config": wcfg, "summary": workload_summary(specs),
                         "processes": _spec_dicts(specs)},
            "context_switch": req.context_switch, "results": results}


# ------------------------------------------------------------------ models
@router.get("/models")
def list_models() -> dict[str, Any]:
    return {"models": registry.list_runs(), "active": models.active_info()}


@router.get("/models/status")
def model_status() -> dict[str, Any]:
    missing = registry.missing_rl_dependencies()
    running = training.any_running()
    active = models.active_info()
    if missing:
        state = "dependencies_missing"
    elif active is not None:
        state = "loaded"
    elif running:
        state = "training"
    else:
        state = "unavailable"
    return {
        "state": state,
        "active": active,
        "training": running,
        "missing_dependencies": missing,
        "available_checkpoints": [m["run_id"] for m in registry.list_runs()
                                  if m.get("has_best") or m.get("has_final")],
        "message": {
            "dependencies_missing": "RL packages are not installed; baselines still work.",
            "loaded": "A trained checkpoint is loaded and used for 'rl' simulations.",
            "training": "A model is training; load a checkpoint when it finishes.",
            "unavailable": "No trained model is loaded. 'rl' simulations are disabled.",
        }[state],
    }


@router.get("/models/{run_id}")
def get_model(run_id: str) -> dict[str, Any]:
    meta = _meta_or_404(run_id)
    meta["has_best"] = registry.checkpoint_path(run_id, "best") is not None
    meta["has_final"] = registry.checkpoint_path(run_id, "final") is not None
    return {"metadata": meta, "progress": registry.progress(run_id),
            "training": training.status(run_id)}


def _meta_or_404(run_id: str) -> dict[str, Any]:
    try:
        meta = registry.metadata(run_id)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    if meta is None:
        raise HTTPException(404, f"model '{run_id}' not found")
    return meta


@router.post("/models/{run_id}/load")
def load_model(run_id: str, req: S.LoadModelRequest | None = None) -> dict[str, Any]:
    _meta_or_404(run_id)
    models.activate(run_id, (req or S.LoadModelRequest()).checkpoint)
    return {"active": models.active_info()}


@router.post("/models/unload")
def unload_model() -> dict[str, Any]:
    models.unload()
    return {"active": None}


@router.post("/models/{run_id}/evaluate")
def evaluate_model(run_id: str, req: S.EvaluateRequest) -> dict[str, Any]:
    _meta_or_404(run_id)
    models.load(run_id, req.checkpoint)  # fail fast with a clear message
    cfg = bench.BenchmarkConfig(
        name=f"eval-{run_id}"[:40],
        seeds=[bench.TEST_SEED_BASE + i for i in range(req.n_seeds)],
        n_processes=req.n_processes,
        context_switch=req.context_switch,
        schedulers=["fcfs", "sjf", "srtf", "rr", "priority", "random", "rl"],
        model_id=run_id,
        model_checkpoint=req.checkpoint,
    )
    return _run_and_save(cfg)


# ---------------------------------------------------------------- training
@router.post("/training", status_code=202)
def start_training(req: S.TrainRequest) -> dict[str, Any]:
    run_id = req.run_id or f"ppo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    reward = RewardConfig(**(req.reward or {})).to_dict()
    eval_every = min(req.eval_every, max(1_000, req.timesteps // 2))
    try:
        training.start({"run_id": run_id, "timesteps": req.timesteps, "seed": req.seed,
                        "n_envs": req.n_envs, "eval_every": eval_every, "reward": reward,
                        "torch_threads": 1})
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    except FileExistsError as e:
        raise HTTPException(409, str(e)) from e
    return {"run_id": run_id, "state": "starting"}


@router.get("/training/{run_id}")
def training_status(run_id: str) -> dict[str, Any]:
    try:
        return training.status(run_id)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@router.post("/training/{run_id}/cancel")
def cancel_training(run_id: str) -> dict[str, Any]:
    st = training_status(run_id)
    if st["state"] not in ("running", "starting"):
        raise HTTPException(409, f"run '{run_id}' is not running (state: {st['state']})")
    training.cancel(run_id)
    return {"run_id": run_id, "state": "cancelling"}


# -------------------------------------------------------------- experiments
def _run_and_save(cfg: bench.BenchmarkConfig) -> dict[str, Any]:
    factory = None
    if "rl" in cfg.schedulers:
        def factory(run_id: str) -> Scheduler:
            return models.scheduler(run_id, cfg.model_checkpoint)
    result = bench.run_benchmark(cfg, factory)
    exp_id = bench.save_experiment(result)
    return {"id": exp_id, **result}


@router.post("/experiments")
def run_experiment(req: S.BenchmarkRequest) -> dict[str, Any]:
    seeds = req.seeds or [bench.TEST_SEED_BASE + i for i in range(req.n_seeds)]
    model_id: str | None = req.model_id
    checkpoint: str = req.model_checkpoint
    if "rl" in req.schedulers and not model_id:
        if models.active is None:
            raise ModelUnavailable("Benchmark includes 'rl' but no model_id was given and "
                                   "no model is loaded.")
        model_id, checkpoint = models.active
    cfg = bench.BenchmarkConfig(
        name=req.name, kinds=list(req.kinds), seeds=seeds, n_processes=req.n_processes,
        context_switch=req.context_switch, rr_quantum=req.rr_quantum,
        priority_aging=req.priority_aging, schedulers=list(req.schedulers),
        model_id=model_id, model_checkpoint=checkpoint,
    )
    return _run_and_save(cfg)


@router.get("/experiments")
def list_experiments() -> dict[str, Any]:
    return {"experiments": bench.list_experiments()}


def _exp_or_404(exp_id: str) -> dict[str, Any]:
    try:
        exp = bench.load_experiment(exp_id)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    if exp is None:
        raise HTTPException(404, f"experiment '{exp_id}' not found")
    return exp


@router.get("/experiments/{exp_id}")
def get_experiment(exp_id: str) -> dict[str, Any]:
    return _exp_or_404(exp_id)


@router.get("/experiments/{exp_id}/export")
def export_experiment(exp_id: str, format: str = Query("csv", pattern="^(csv|json)$")) -> Response:
    exp = _exp_or_404(exp_id)
    if format == "json":
        import json

        body = json.dumps(exp, indent=2)
        return Response(body, media_type="application/json", headers={
            "Content-Disposition": f'attachment; filename="{exp_id}.json"'})
    buf = io.StringIO()
    import csv

    rows = exp["rows"]
    if rows:
        w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return Response(buf.getvalue(), media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="{exp_id}.csv"'})


@router.post("/experiments/{exp_id}/reproduce")
def reproduce_experiment(exp_id: str) -> dict[str, Any]:
    exp = _exp_or_404(exp_id)
    cfg = bench.BenchmarkConfig(**exp["config"])
    factory = None
    if "rl" in cfg.schedulers:
        def factory(run_id: str) -> Scheduler:
            return models.scheduler(run_id, cfg.model_checkpoint)
    fresh = bench.run_benchmark(cfg, factory)
    old = {(r["kind"], r["seed"], r["scheduler"]): r for r in exp["rows"]}
    max_diff = 0.0
    mismatched = 0
    for r in fresh["rows"]:
        o = old.get((r["kind"], r["seed"], r["scheduler"]))
        if o is None:
            mismatched += 1
            continue
        for k, v in r.items():
            if isinstance(v, (int, float)) and k != "seed":
                d = abs(float(v) - float(o[k]))
                max_diff = max(max_diff, d)
    return {
        "id": exp_id,
        "identical": mismatched == 0 and max_diff < 1e-9 and len(fresh["rows"]) == len(old),
        "rows": len(fresh["rows"]),
        "max_abs_difference": max_diff,
        "missing_rows": mismatched,
    }
