"""Pruebas unitarias de la Etapa 2 (TrafficEnvironment + CustomStateBuilder).

Usan episodios cortos (300 s = 60 pasos) para que corran rápido.
"""
import numpy as np
import pytest

from wm.env.state_builder import FEATURE_NAMES, N_FEATURES
from wm.env.traffic_env import make_env

F = {n: i for i, n in enumerate(FEATURE_NAMES)}
SHORT = 300


def rollout(env, policy, seed=1):
    obs, info = env.reset(seed=seed)
    O, A, R, RV = [obs], [], [], []
    for _ in range(env.max_steps):
        obs, r, term, trunc, info = env.step(policy(obs))
        O.append(obs), A.append(info["action"]), R.append(r), RV.append(info["reward_vec"])
        if trunc:
            break
    env.close()
    return np.stack(O), np.stack(A), np.array(R), np.stack(RV)


def check_physical(O, speed_cap=30.0):
    assert np.isfinite(O).all()
    assert (O >= 0).all()
    assert (O[..., F["halting"]] <= O[..., F["vehicles"]]).all()
    assert (O[..., F["mean_speed"]] <= speed_cap).all()
    assert (O[..., F["occupancy"]] <= 100).all()
    onehot = O[..., F["phase_NS_straight"]:F["phase_EW_left"] + 1]
    assert np.allclose(onehot.sum(-1), 1)
    assert set(np.unique(O[..., F["yellow"]])) <= {0.0, 1.0}


def test_spaces_and_shapes():
    env = make_env(demand="D1", episode_seconds=SHORT)
    assert env.max_steps == SHORT // 5
    obs, info = env.reset(seed=1)
    assert obs.shape == (7, N_FEATURES) == env.observation_space.shape
    assert env.action_space.nvec.tolist() == [2] * 7
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert isinstance(r, float) and np.isclose(r, info["reward_vec"].sum(), atol=1e-4)
    assert info["raw_metrics"].shape == (7, 4)
    assert info["sim_time"] == 300 + 5
    env.close()


def test_random_policy_physical_ranges():
    rng = np.random.default_rng(0)
    O, A, R, _ = rollout(make_env(demand="D3", episode_seconds=SHORT), lambda o: rng.integers(0, 2, 7))
    assert len(A) == SHORT // 5
    check_physical(O)
    assert O[..., F["time_in_phase"]].max() <= 60
    assert A.sum() > 0


def test_min_green_respected():
    """Pidiendo cambiar siempre, cada verde dura 11 s (Δt múltiplo de 5 y 10 s de mínimo) -> 1 cambio cada 3 pasos."""
    O, A, _, _ = rollout(make_env(demand="D2", episode_seconds=SHORT), lambda o: np.ones(7, int))
    for i in range(7):
        idx = np.flatnonzero(A[:, i])
        assert (np.diff(idx) >= 3).all()
    greens = O[:-1, :, F["time_in_phase"]][A == 1]  # tiempo en verde observado al decidir el cambio
    assert (greens >= 10).all()


def test_max_green_forced():
    """Sin pedir nunca cambio, el entorno fuerza el cambio al llegar a 60 s de verde."""
    env = make_env(demand="D2", episode_seconds=SHORT)
    env.reset(seed=1)
    forced, tip = 0, []
    for _ in range(env.max_steps):
        obs, _, _, trunc, info = env.step(np.zeros(7, int))
        forced += info["forced"].sum()
        tip.append(obs[:, F["time_in_phase"]])
    env.close()
    assert forced > 0
    assert np.max(tip) <= 60


def test_determinism_same_seed():
    def run():
        rng = np.random.default_rng(42)
        return rollout(make_env(demand="D2", episode_seconds=SHORT), lambda o: rng.integers(0, 2, 7), seed=3)[0]

    assert np.array_equal(run(), run())


@pytest.mark.parametrize("control", ["fixed", "actuated"])
def test_external_control_modes(control):
    O, A, _, _ = rollout(make_env(demand="D2", control=control, episode_seconds=SHORT), lambda o: np.zeros(7, int))
    check_physical(O)
    assert A.sum() > 0  # los cambios de SUMO se registran como acción efectiva
    if control == "fixed":  # ciclo de 90 s con 4 verdes -> ~4 cambios por intersección cada 90 s
        per_tls = A.sum(0)
        assert (abs(per_tls - SHORT / 90 * 4) <= 1).all()
