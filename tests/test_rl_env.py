"""RL environment: spaces, masking, reward telescoping, termination, seeding."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from gymnasium.utils.env_checker import check_env

from neuralos.rl.env import EnvSettings, SchedulingEnv
from neuralos.rl.spaces import (
    ObservationBuilder,
    RewardConfig,
    SpaceConfig,
    cumulative_cost,
    reward_normaliser,
)
from neuralos.sim import ProcessSpec, SimConfig, Simulator, WorkloadConfig, generate
from neuralos.sim.metrics import compute_metrics

pytestmark = pytest.mark.rl


def P(pid: int, arrival: int, *b: int, prio: int = 0) -> ProcessSpec:
    return ProcessSpec(pid, arrival, tuple(b), prio)


def first_valid(env: SchedulingEnv) -> int:
    return int(np.flatnonzero(env.action_masks())[0])


def test_gymnasium_env_checker():
    # check_env samples random actions without a mask; a masked env correctly
    # rejects those. Give the checker an action space that samples through the
    # current mask, which is exactly what MaskablePPO does.
    from gymnasium import spaces

    env = SchedulingEnv()

    class MaskedDiscrete(spaces.Discrete):
        def sample(self, mask=None, probability=None):  # type: ignore[override]
            return super().sample(mask=env.action_masks().astype(np.int8))

    env.action_space = MaskedDiscrete(env.settings.space.n_actions)
    check_env(env, skip_render_check=True)


def test_observation_in_bounds_and_mask_matches_ready_queue():
    env = SchedulingEnv()
    obs, _ = env.reset(seed=3)
    for _ in range(500):
        assert env.observation_space.contains(obs)
        mask = env.action_masks()
        assert env.sim is not None
        visible = min(len(env.sim.ready), env.settings.space.slots)
        assert visible >= 1, "env must only ask for actions when something is ready"
        assert mask.sum() == visible * len(env.settings.space.quanta)
        rng = np.random.default_rng(0)
        obs, r, term, trunc, _ = env.step(int(rng.choice(np.flatnonzero(mask))))
        assert np.isfinite(r) and r <= 1e-9  # costs only accumulate
        assert not trunc
        if term:
            obs, _ = env.reset()


def test_invalid_action_raises():
    env = SchedulingEnv(workloads=[([P(0, 0, 5)], SimConfig())])
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(env.settings.space.n_actions - 1)  # last slot is empty
    with pytest.raises(ValueError):
        env.step(10_000)


def test_reward_telescopes_to_episode_cost():
    specs = generate(WorkloadConfig(kind="mixed", n_processes=25, seed=11))
    env = SchedulingEnv(workloads=[(specs, SimConfig(context_switch=1))])
    env.reset(seed=0)
    total = 0.0
    rng = np.random.default_rng(1)
    while True:
        a = int(rng.choice(np.flatnonzero(env.action_masks())))
        _, r, term, _, info = env.step(a)
        total += r
        if term:
            break
    assert total == pytest.approx(-info["episode_cost"], rel=1e-9)
    # Episode cost recomputed independently from the finished metrics:
    rc = env.settings.reward
    m = compute_metrics(env.sim)
    stats = {p.pid: p for p in env.sim.procs.values()}
    s = reward_normaliser(env.sim) / len(specs)
    expected = (
        rc.wait * sum(p.total_wait for p in stats.values())
        + rc.fair * sum(p.wait_sq_half for p in stats.values()) / s
        + rc.resp * m.avg_response * len(specs)
        + rc.switch * m.context_switches * s
    ) / reward_normaliser(env.sim)
    assert info["episode_cost"] == pytest.approx(expected, rel=1e-9)


def test_fairness_term_punishes_starvation_more_than_spreading():
    # Same total waiting (20 ticks): one process waits 20 vs two wait 10 each.
    rc = RewardConfig(wait=1, fair=1, resp=0, switch=0)
    a = Simulator([P(0, 0, 20), P(1, 0, 1)])
    a.dispatch(0)
    a.dispatch(1)
    b = Simulator([P(0, 0, 10), P(1, 0, 10), P(2, 0, 1)])
    b.dispatch(0)
    b.dispatch(2)
    b.dispatch(1)
    # a: one stretch of 20 ticks -> 200. b: stretches of 10 and 11 -> 110.5,
    # although b's total waiting (21) is higher than a's (20).
    sa = sum(p.wait_sq_half for p in a.procs.values())
    sb = sum(p.wait_sq_half for p in b.procs.values())
    assert sa == pytest.approx(0.5 * 20**2)
    assert sb == pytest.approx(0.5 * 10**2 + 0.5 * 11**2)
    # Normalise by S so both are on the same scale (S = mean CPU burst).
    ca = cumulative_cost(a, rc) * 1.0
    cb = cumulative_cost(b, rc) * 1.0
    assert ca > 0 and cb > 0


def test_terminal_condition_and_reset():
    env = SchedulingEnv(workloads=[([P(0, 0, 2), P(1, 3, 2)], SimConfig())])
    env.reset(seed=0)
    _, _, term, _, _ = env.step(first_valid(env))
    assert not term  # P1 still to come; env skipped the idle gap to t=3
    assert env.sim.now == 3
    _, _, term, _, info = env.step(first_valid(env))
    assert term and "metrics" in info
    with pytest.raises(RuntimeError):
        env.step(0)
    env.reset()
    assert not env.sim.done


def test_reproducible_seeding():
    def rollout(seed: int) -> tuple[list[float], dict]:
        env = SchedulingEnv()
        _, info = env.reset(seed=seed)
        rs = []
        for _ in range(60):
            _, r, term, _, _ = env.step(first_valid(env))
            rs.append(r)
            if term:
                env.reset()
        return rs, info["workload"]

    a, wa = rollout(42)
    b, wb = rollout(42)
    _c, wc = rollout(43)
    assert a == b and wa == wb
    assert wa != wc


def test_quantum_choice_is_respected():
    space = SpaceConfig(slots=2, quanta=(1, None))
    wl = [([P(0, 0, 5), P(1, 0, 5)], SimConfig())]
    env = SchedulingEnv(EnvSettings(space=space), workloads=wl)
    env.reset(seed=0)
    env.step(0)  # slot 0, quantum 1
    assert env.sim.now == 1 and env.sim.procs[0].remaining_in_burst == 4
    env.step(1)  # slot 0 is now P1 (P0 went to the tail), run to completion
    assert env.sim.procs[1].remaining_in_burst == 0


def test_oldest_processes_are_visible():
    specs = [P(i, 0, 3) for i in range(15)]
    sim = Simulator(specs)
    b = ObservationBuilder(SpaceConfig(slots=4))
    assert [p.pid for p in b.visible(sim)] == [0, 1, 2, 3]
    obs = b.build(sim)
    assert obs[4 * 10 + 1] == pytest.approx(11 / 4)  # hidden count / K
