"""Train a MaskablePPO scheduling policy.

Usage::

    python -m neuralos.rl.train --timesteps 300000 --seed 0 --run-id ppo-default

Model selection uses a fixed *validation* set (seed range 5,000,000+), which
is disjoint from training seeds (0-4,999,999) and from the test seeds used by
benchmarks (10,000,000+).
"""

from __future__ import annotations

import argparse
import logging
import platform
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..paths import safe_id
from ..sim.engine import SimConfig
from ..sim.metrics import compute_metrics
from ..sim.runner import simulate
from ..sim.schedulers import make_baseline
from ..sim.workload import WorkloadConfig, WorkloadKind, generate
from . import registry
from .env import (
    VALIDATION_SEED_BASE,
    EnvSettings,
    SchedulingEnv,
    TrainDistribution,
)
from .spaces import RewardConfig, SpaceConfig, cumulative_cost, reward_normaliser

MAX_TIMESTEPS = 3_000_000
log = logging.getLogger("neuralos.train")


@dataclass
class TrainConfig:
    run_id: str
    timesteps: int = 300_000
    seed: int = 0
    n_envs: int = 4
    learning_rate: float = 3e-4
    n_steps: int = 512
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.995
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    net_arch: list[int] = field(default_factory=lambda: [128, 128])
    eval_every: int = 20_000
    validation_per_kind: int = 4
    validation_n_processes: int = 30
    torch_threads: int = 2
    space: SpaceConfig = field(default_factory=SpaceConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    distribution: TrainDistribution = field(default_factory=TrainDistribution)

    def validate(self) -> None:
        safe_id(self.run_id)
        if not 1_000 <= self.timesteps <= MAX_TIMESTEPS:
            raise ValueError(f"timesteps must be in [1000, {MAX_TIMESTEPS}]")
        if not 1 <= self.n_envs <= 16:
            raise ValueError("n_envs must be in [1, 16]")
        if self.batch_size > self.n_steps * self.n_envs:
            raise ValueError("batch_size must be <= n_steps * n_envs")
        if not 0 < self.learning_rate < 1:
            raise ValueError("learning_rate must be in (0, 1)")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["space"] = self.space.to_dict()
        d["reward"] = self.reward.to_dict()
        d["distribution"] = self.distribution.to_dict()
        return d


def validation_set(
    kinds: tuple[WorkloadKind, ...], per_kind: int, n_processes: int
) -> list[tuple[str, list[Any], SimConfig]]:
    out: list[tuple[str, list[Any], SimConfig]] = []
    for ki, kind in enumerate(kinds):
        for j in range(per_kind):
            seed = VALIDATION_SEED_BASE + ki * 1000 + j
            specs = generate(WorkloadConfig(kind=kind, n_processes=n_processes, seed=seed))
            out.append((kind, specs, SimConfig(context_switch=j % 2)))
    return out


def score(scheduler: Any, vset: list[tuple[str, list[Any], SimConfig]],
          rc: RewardConfig) -> dict[str, float]:
    objs, waits, resps, maxw = [], [], [], []
    for _, specs, cfg in vset:
        sim = simulate(specs, scheduler, cfg)
        m = compute_metrics(sim)
        objs.append(cumulative_cost(sim, rc) / reward_normaliser(sim))
        waits.append(m.avg_waiting)
        resps.append(m.avg_response)
        maxw.append(m.max_wait)
    return {
        "objective": float(np.mean(objs)),
        "avg_waiting": float(np.mean(waits)),
        "avg_response": float(np.mean(resps)),
        "max_wait": float(np.mean(maxw)),
    }


def _versions() -> dict[str, str]:
    import gymnasium
    import sb3_contrib
    import stable_baselines3
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "sb3_contrib": sb3_contrib.__version__,
    }


def train(cfg: TrainConfig) -> Path:
    cfg.validate()
    missing = registry.missing_rl_dependencies()
    if missing:
        raise RuntimeError(
            f"missing RL dependencies: {', '.join(missing)} "
            "(pip install -r backend/requirements-rl.txt)"
        )
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    from .policy import RLScheduler

    torch.set_num_threads(cfg.torch_threads)
    d = registry.run_dir(cfg.run_id)
    if (d / "metadata.json").exists():
        raise FileExistsError(f"run '{cfg.run_id}' already exists in {d}")
    d.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(d / "train.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)
    log.setLevel(logging.INFO)

    meta: dict[str, Any] = {
        "run_id": cfg.run_id,
        "algorithm": "MaskablePPO",
        "policy": "MlpPolicy",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "running",
        "seed": cfg.seed,
        "timesteps_requested": cfg.timesteps,
        "timesteps_done": 0,
        "config": cfg.to_dict(),
        "space": cfg.space.to_dict(),
        "reward": cfg.reward.to_dict(),
        "versions": _versions(),
        "best": None,
        "error": None,
    }
    registry.write_json(d / "metadata.json", meta)

    settings = EnvSettings(space=cfg.space, reward=cfg.reward, train=cfg.distribution)
    vset = validation_set(cfg.distribution.kinds, cfg.validation_per_kind,
                          cfg.validation_n_processes)
    baseline_scores = {
        name: score(make_baseline(name), vset, cfg.reward) for name in ("fcfs", "sjf", "srtf", "rr")
    }
    log.info("validation set: %d workloads; baselines: %s", len(vset),
             {k: round(v["objective"], 3) for k, v in baseline_scores.items()})

    def make_env(rank: int) -> Any:
        def _f() -> Any:
            env = Monitor(SchedulingEnv(settings))
            env.reset(seed=cfg.seed * 1000 + rank)
            return env
        return _f

    venv = DummyVecEnv([make_env(i) for i in range(cfg.n_envs)])
    model = MaskablePPO(
        "MlpPolicy",
        venv,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        ent_coef=cfg.ent_coef,
        policy_kwargs={"net_arch": list(cfg.net_arch)},
        seed=cfg.seed,
        device="cpu",
        verbose=0,
    )

    state: dict[str, Any] = {
        "curve": [],
        "validation": [],
        "best": None,
        "t0": time.time(),
        "last_log": 0,
        "next_eval": cfg.eval_every,
    }

    def write_progress(status: str, timesteps: int) -> None:
        elapsed = time.time() - state["t0"]
        registry.write_json(
            d / "progress.json",
            {
                "run_id": cfg.run_id,
                "status": status,
                "timesteps": timesteps,
                "total_timesteps": cfg.timesteps,
                "elapsed_s": round(elapsed, 1),
                "fps": round(timesteps / elapsed, 1) if elapsed > 0 else 0.0,
                "curve": state["curve"],
                "validation": state["validation"],
                "baselines": baseline_scores,
                "best": state["best"],
            },
        )

    def validate(timesteps: int) -> None:
        sched = RLScheduler(model, cfg.space, model_id=f"{cfg.run_id}@{timesteps}")
        s = score(sched, vset, cfg.reward)
        s["timesteps"] = timesteps
        state["validation"].append(s)
        improved = state["best"] is None or s["objective"] < state["best"]["objective"]
        if improved:
            state["best"] = s
            model.save(d / "best.zip")
        log.info("t=%d validation objective=%.3f wait=%.2f%s", timesteps, s["objective"],
                 s["avg_waiting"], " (best)" if improved else "")

    class Progress(BaseCallback):  # type: ignore[misc]
        def _on_step(self) -> bool:
            t = self.num_timesteps
            if t - state["last_log"] >= 2_000:
                state["last_log"] = t
                buf = list(self.model.ep_info_buffer or [])
                if buf:
                    state["curve"].append(
                        {
                            "timesteps": t,
                            "ep_reward_mean": float(np.mean([e["r"] for e in buf])),
                            "ep_len_mean": float(np.mean([e["l"] for e in buf])),
                        }
                    )
                write_progress("running", t)
            if t >= state["next_eval"]:
                state["next_eval"] += cfg.eval_every
                validate(t)
                write_progress("running", t)
            if registry.cancel_requested(cfg.run_id):
                log.info("cancellation requested at t=%d", t)
                return False
            return True

    status = "completed"
    try:
        log.info("training %s for %d timesteps (seed %d)", cfg.run_id, cfg.timesteps, cfg.seed)
        validate(0)  # score of the untrained policy, for reference
        model.learn(total_timesteps=cfg.timesteps, callback=Progress(), progress_bar=False)
        if registry.cancel_requested(cfg.run_id):
            status = "cancelled"
        else:
            validate(int(model.num_timesteps))
        model.save(d / "final.zip")
    except Exception as exc:
        status = "failed"
        meta["error"] = f"{type(exc).__name__}: {exc}"
        log.error("training failed:\n%s", traceback.format_exc())
        raise
    finally:
        meta.update(
            status=status,
            timesteps_done=int(model.num_timesteps),
            best=state["best"],
            baselines=baseline_scores,
            finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            duration_s=round(time.time() - state["t0"], 1),
        )
        registry.write_json(d / "metadata.json", meta)
        write_progress(status, int(model.num_timesteps))
        log.info("finished with status %s", status)
        log.removeHandler(fh)
        fh.close()
    return d


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train a NeuralOS MaskablePPO scheduler")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=20_000)
    ap.add_argument("--reward-wait", type=float, default=RewardConfig.wait)
    ap.add_argument("--reward-fair", type=float, default=RewardConfig.fair)
    ap.add_argument("--reward-resp", type=float, default=RewardConfig.resp)
    ap.add_argument("--reward-switch", type=float, default=RewardConfig.switch)
    args = ap.parse_args(argv)
    run_id = args.run_id or f"ppo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = TrainConfig(
        run_id=run_id,
        timesteps=args.timesteps,
        seed=args.seed,
        n_envs=args.n_envs,
        eval_every=args.eval_every,
        reward=RewardConfig(args.reward_wait, args.reward_fair, args.reward_resp,
                            args.reward_switch),
    )
    out = train(cfg)
    meta = registry.metadata(run_id) or {}
    print(f"\nRun '{run_id}' {meta.get('status')} in {meta.get('duration_s')}s -> {out}")
    print(f"Best validation: {meta.get('best')}")
    print(f"Validation baselines: {meta.get('baselines')}")


if __name__ == "__main__":
    main()
