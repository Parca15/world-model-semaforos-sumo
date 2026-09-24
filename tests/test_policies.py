"""Pruebas de las políticas de recolección (P1–P4)."""
import numpy as np
import pytest

from wm.control.policies import make_policy
from wm.data.collect import EpisodeSpec, run_episode

SHORT = 200


def spec(kind, policy="PX", seed=1, **params):
    return EpisodeSpec(episode="ep_test", demand="D2", policy=policy, policy_kind=kind, seed=seed, split="train",
                       policy_params=params)


@pytest.mark.parametrize("kind,params", [("fixed_time", {}), ("actuated", {}),
                                         ("random_restricted", {"p": 0.5}), ("max_pressure", {"epsilon": 0.1})])
def test_policy_runs_and_records(kind, params):
    rec = run_episode(spec(kind, **params), episode_seconds=SHORT)
    T = SHORT // 5
    assert rec.states.shape == (T + 1, 7, 13)
    assert rec.actions.shape == rec.rewards.shape == (T, 7)
    assert rec.raw_metrics.shape == (T + 1, 7, 4)
    assert rec.actions.sum() > 0


def test_policy_is_reproducible():
    a = run_episode(spec("max_pressure", epsilon=0.1), episode_seconds=SHORT)
    b = run_episode(spec("max_pressure", epsilon=0.1), episode_seconds=SHORT)
    assert np.array_equal(a.states, b.states) and np.array_equal(a.actions, b.actions)


def test_policy_rng_depends_on_seed_policy_and_demand():
    draws = {s.policy_rng().integers(1 << 30) for s in (spec("x", "P2", 1), spec("x", "P2", 2), spec("x", "P4", 1))}
    assert len(draws) == 3


def test_max_pressure_beats_random_on_waiting():
    mp = run_episode(spec("max_pressure", epsilon=0.0), episode_seconds=600)
    rnd = run_episode(spec("random_restricted", p=0.5), episode_seconds=600)
    assert mp.raw_metrics[..., 3].mean() < rnd.raw_metrics[..., 3].mean()


def test_unknown_policy():
    with pytest.raises(ValueError):
        make_policy("nope")
