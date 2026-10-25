"""API validation, error handling and end-to-end integration."""

from __future__ import annotations

import pytest

CLASSIC = [
    {"pid": 1, "arrival": 0, "bursts": [8]},
    {"pid": 2, "arrival": 1, "bursts": [4]},
    {"pid": 3, "arrival": 2, "bursts": [9]},
    {"pid": 4, "arrival": 3, "bursts": [5]},
]


def test_health_and_config(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok"
    cfg = client.get("/api/config").json()
    names = [s["name"] for s in cfg["schedulers"]]
    assert {"fcfs", "sjf", "srtf", "rr", "priority", "rl"} <= set(names)
    assert len(cfg["workload_kinds"]) == 7


def test_integration_srtf_metrics_through_api(client):
    """Run a hand-calculable workload through HTTP and verify exact metrics."""
    r = client.post("/api/simulations", json={
        "workload": {"processes": CLASSIC},
        "scheduler": {"name": "srtf"},
    })
    assert r.status_code == 200, r.text
    body = r.json()
    m = body["metrics"]
    assert m["avg_waiting"] == pytest.approx(6.5)
    assert m["avg_turnaround"] == pytest.approx((17 + 4 + 24 + 7) / 4)
    assert m["context_switches"] == 4
    runs = [(s["start"], s["end"], s["pid"]) for s in body["timeline"] if s["kind"] == "run"]
    assert runs == [(0, 1, 1), (1, 5, 2), (5, 10, 4), (10, 17, 1), (17, 26, 3)]

    # Retrieval endpoints return the same stored result.
    sid = body["id"]
    again = client.get(f"/api/simulations/{sid}").json()
    assert again["metrics"] == m
    tl = client.get(f"/api/simulations/{sid}/timeline").json()
    assert tl["timeline"] == body["timeline"]
    assert tl["transitions"][0] == {"time": 0, "pid": 1, "from": "NEW", "to": "READY"}


def test_round_robin_with_context_switch_through_api(client):
    r = client.post("/api/simulations", json={
        "workload": {"processes": [{"pid": 1, "arrival": 0, "bursts": [3]},
                                   {"pid": 2, "arrival": 0, "bursts": [2]}]},
        "scheduler": {"name": "rr", "quantum": 2},
        "context_switch": 1,
    }).json()
    # P1 0-2, switch 2-3, P2 3-5, switch 5-6, P1 6-7
    kinds = [(s["start"], s["end"], s["kind"]) for s in r["timeline"]]
    assert kinds == [(0, 2, "run"), (2, 3, "switch"), (3, 5, "run"), (5, 6, "switch"),
                     (6, 7, "run")]
    assert r["metrics"]["switch_overhead"] == 2


def test_generate_workload_is_seeded(client):
    a = client.post("/api/workloads/generate", json={"kind": "bursty", "n_processes": 15,
                                                     "seed": 9}).json()
    b = client.post("/api/workloads/generate", json={"kind": "bursty", "n_processes": 15,
                                                     "seed": 9}).json()
    assert a["processes"] == b["processes"] and len(a["processes"]) == 15
    assert a["config"]["mean_cpu_burst"] is not None  # preset resolved


def test_compare_uses_same_workload(client):
    r = client.post("/api/compare", json={
        "workload": {"generate": {"kind": "mixed", "n_processes": 20, "seed": 4}},
        "schedulers": [{"name": "fcfs"}, {"name": "sjf"}, {"name": "rr", "quantum": 3}],
    }).json()
    assert [x["scheduler"]["name"] for x in r["results"]] == ["fcfs", "sjf", "rr"]
    # Same workload => same total CPU executed by every scheduler.
    totals = {sum(s["end"] - s["start"] for s in x["timeline"] if s["kind"] == "run")
              for x in r["results"]}
    assert len(totals) == 1


@pytest.mark.parametrize("payload", [
    {"workload": {}},  # neither generate nor processes
    {"workload": {"processes": [{"pid": 1, "arrival": 0, "bursts": [2, 3]}]}},  # even bursts
    {"workload": {"processes": [{"pid": 1, "arrival": 0, "bursts": [0]}]}},
    {"workload": {"processes": [{"pid": 1, "arrival": 0, "bursts": [2]},
                                {"pid": 1, "arrival": 1, "bursts": [2]}]}},
    {"workload": {"generate": {"n_processes": 0}}},
    {"workload": {"generate": {"n_processes": 100000}}},
    {"workload": {"generate": {"kind": "nope"}}},
    {"workload": {"generate": {}}, "scheduler": {"name": "lottery"}},
    {"workload": {"generate": {}}, "scheduler": {"name": "rr", "quantum": 0}},
    {"workload": {"generate": {}}, "context_switch": -1},
    {"workload": {"generate": {}}, "unknown_field": 1},
])
def test_invalid_simulation_requests(client, payload):
    r = client.post("/api/simulations", json=payload)
    assert r.status_code == 422, r.text


def test_rl_without_model_is_refused(client):
    status = client.get("/api/models/status").json()
    assert status["active"] is None
    r = client.post("/api/simulations", json={
        "workload": {"generate": {}}, "scheduler": {"name": "rl"}})
    assert r.status_code == 409
    assert "No trained model" in r.json()["detail"]


def test_missing_resources(client):
    assert client.get("/api/simulations/doesnotexist").status_code == 404
    assert client.get("/api/models/nope").status_code == 404
    assert client.post("/api/models/nope/load", json={}).status_code == 404
    assert client.get("/api/experiments/nope").status_code == 404
    assert client.get("/api/models/..%2F..%2Fetc").status_code in (404, 422)
    assert client.post("/api/training/nope/cancel").status_code == 409


def test_experiment_lifecycle_and_reproducibility(client):
    r = client.post("/api/experiments", json={
        "name": "apitest", "kinds": ["short_jobs", "io_bound"], "n_seeds": 3,
        "n_processes": 15, "schedulers": ["fcfs", "sjf", "rr"],
    })
    assert r.status_code == 200, r.text
    exp = r.json()
    assert len(exp["rows"]) == 2 * 3 * 3
    groups = {(g["kind"], g["scheduler"]) for g in exp["summary"]["groups"]}
    assert ("ALL", "sjf") in groups
    eid = exp["id"]
    listed = client.get("/api/experiments").json()["experiments"]
    assert any(e["id"] == eid for e in listed)
    csv_text = client.get(f"/api/experiments/{eid}/export?format=csv").text
    assert csv_text.splitlines()[0].startswith("kind,seed,scheduler,")
    assert len(csv_text.strip().splitlines()) == 1 + 18
    js = client.get(f"/api/experiments/{eid}/export?format=json").json()
    assert js["fingerprint"] == exp["fingerprint"]
    rep = client.post(f"/api/experiments/{eid}/reproduce").json()
    assert rep["identical"] is True and rep["max_abs_difference"] == 0.0


def test_experiment_rl_without_model(client):
    r = client.post("/api/experiments", json={"schedulers": ["fcfs", "rl"], "n_seeds": 1})
    assert r.status_code == 409


def test_training_request_validation(client):
    assert client.post("/api/training", json={"timesteps": 10}).status_code == 422
    assert client.post("/api/training", json={"run_id": "../x"}).status_code == 422
    assert client.post("/api/training",
                       json={"reward": {"wait": -1}}).status_code == 422
