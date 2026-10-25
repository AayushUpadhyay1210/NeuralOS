"""Real (tiny) training runs: checkpointing, loading, inference, API training path.

These train actual MaskablePPO models for a few thousand steps. They verify
the pipeline works, not that the policy is good.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("sb3_contrib")

from neuralos.rl import registry
from neuralos.rl.loader import ModelUnavailable, load_scheduler
from neuralos.rl.train import TrainConfig, train
from neuralos.sim import SimConfig, WorkloadConfig, generate, run

pytestmark = [pytest.mark.rl, pytest.mark.slow]


@pytest.fixture()
def tiny_model(isolated_dirs):
    cfg = TrainConfig(run_id="tiny", timesteps=2_048, n_envs=2, n_steps=256, batch_size=128,
                      eval_every=1_024, validation_per_kind=1, validation_n_processes=10,
                      torch_threads=1)
    train(cfg)
    return "tiny"


def test_training_writes_checkpoints_and_metadata(tiny_model):
    meta = registry.metadata(tiny_model)
    assert meta["status"] == "completed"
    assert meta["algorithm"] == "MaskablePPO"
    assert meta["timesteps_done"] >= 2_048
    assert registry.checkpoint_path(tiny_model, "best") is not None
    assert registry.checkpoint_path(tiny_model, "final") is not None
    prog = registry.progress(tiny_model)
    assert prog["status"] == "completed"
    assert len(prog["validation"]) >= 2  # t=0 plus at least one more
    assert {"fcfs", "sjf", "srtf", "rr"} <= set(prog["baselines"])
    assert (registry.run_dir(tiny_model) / "train.log").read_text()


def test_checkpoint_inference_is_deterministic(tiny_model):
    specs = generate(WorkloadConfig(kind="mixed", n_processes=25, seed=99))
    a = run(specs, load_scheduler(tiny_model, "best"), SimConfig(context_switch=1))
    b = run(specs, load_scheduler(tiny_model, "best"), SimConfig(context_switch=1))
    assert a.metrics == b.metrics
    assert a.scheduler["name"] == "rl"
    assert a.metrics.n_processes == 25


def test_duplicate_run_id_rejected(tiny_model):
    with pytest.raises(FileExistsError):
        train(TrainConfig(run_id=tiny_model, timesteps=1_000))


def test_unknown_model_is_reported(isolated_dirs):
    with pytest.raises(ModelUnavailable):
        load_scheduler("missing-model")


def test_api_load_simulate_evaluate(client, tiny_model):
    st = client.get("/api/models/status").json()
    assert st["state"] == "unavailable" and "tiny" in st["available_checkpoints"]
    meta = client.get(f"/api/models/{tiny_model}").json()["metadata"]
    assert meta["has_best"] and meta["has_final"]
    r = client.post(f"/api/models/{tiny_model}/load", json={"checkpoint": "best"})
    assert r.status_code == 200, r.text
    st = client.get("/api/models/status").json()
    assert st["state"] == "loaded" and st["active"]["run_id"] == tiny_model
    sim = client.post("/api/simulations", json={
        "workload": {"generate": {"kind": "io_bound", "n_processes": 12, "seed": 1}},
        "scheduler": {"name": "rl"},
    })
    assert sim.status_code == 200, sim.text
    assert sim.json()["scheduler"]["model_id"] == f"{tiny_model}:best"
    ev = client.post(f"/api/models/{tiny_model}/evaluate",
                     json={"n_seeds": 1, "n_processes": 10})
    assert ev.status_code == 200, ev.text
    assert any(r["scheduler"] == "rl" for r in ev.json()["rows"])
    assert ev.json()["summary"]["head_to_head"]


def test_api_training_process_and_cancel(client):
    r = client.post("/api/training", json={"run_id": "apirun", "timesteps": 200_000,
                                           "n_envs": 1, "eval_every": 1_000})
    assert r.status_code == 202, r.text
    # A second concurrent job is refused.
    assert client.post("/api/training", json={"run_id": "other"}).status_code == 409
    deadline = time.time() + 120
    while time.time() < deadline:
        st = client.get("/api/training/apirun").json()
        if st["state"] == "running" and (st.get("progress") or {}).get("timesteps", 0) > 0:
            break
        assert st["state"] in ("starting", "running", "unknown"), st
        time.sleep(1)
    else:
        pytest.fail("training never reported progress")
    assert client.post("/api/training/apirun/cancel").status_code == 200
    while time.time() < deadline:
        st = client.get("/api/training/apirun").json()
        if st["state"] in ("cancelled", "completed", "failed"):
            break
        time.sleep(1)
    assert st["state"] == "cancelled", st
    assert registry.checkpoint_path("apirun", "final") is not None
