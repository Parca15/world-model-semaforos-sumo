"""Pruebas de las políticas de recolección (P1–P4)."""
import numpy as np
import pytest

from wm.control.policies import make_policy
from wm.data.collect import EpisodeSpec, run_episode

SHORT = 200


def spec(kind, policy="PX", seed=1, demand="D2", **params):
    return EpisodeSpec(episode="ep_test", demand=demand, policy=policy, policy_kind=kind, seed=seed, split="train",
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


def test_pressure_sensor_covers_upstream_queue():
    """La cola de entrada incluye el tramo aguas arriba del bolsillo (regresión: antes solo se contaban 60 m)."""
    from wm.env.pressure import PressureSensor
    from wm.env.traffic_env import make_env

    env = make_env(demand="D2", episode_seconds=50)
    env.reset(seed=1)
    sensor = PressureSensor(env.sim, env.tls_ids)
    through = [ins for per in sensor.phase_links for mv in per for ins, _ in mv if len(ins) == 2]
    env.close()
    # p. ej. el carril "W_J0_in_1" (bolsillo) se complementa con "W_J0_1" (tramo aguas arriba)
    assert through and all(up == pocket.replace("_in_", "_") for pocket, up in through)


def test_max_pressure_beats_fixed_time_under_high_demand():
    """Regresión: una heurística de presión razonable debe superar al tiempo fijo en D3."""
    mp = run_episode(spec("max_pressure", demand="D3", epsilon=0.0), episode_seconds=1200)
    fx = run_episode(spec("fixed_time", demand="D3"), episode_seconds=1200)
    assert mp.raw_metrics[..., 3].mean() < fx.raw_metrics[..., 3].mean()
