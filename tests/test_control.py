"""Pruebas de las piezas de control: wrapper de PPO en SUMO, DreamEnv, reglas del semáforo y planificador."""
import numpy as np
from stable_baselines3.common.monitor import Monitor

from wm.control.planning import ImaginationPlanner
from wm.control.ppo import SumoPPOEnv, make_ppo
from wm.data.base import FlatWindows
from wm.data.normalization import NormStats
from wm.dream.dream_env import DreamVecEnv
from wm.dream.rules import F, SignalRules
from wm.models.baselines import Persistence

N_TLS, N_FEAT = 7, 13


def unit_norm():
    return NormStats(np.zeros((N_TLS, N_FEAT), np.float32), np.ones((N_TLS, N_FEAT), np.float32),
                     np.zeros(N_TLS, np.float32), np.ones(N_TLS, np.float32))


def raw_state(time_in_phase, yellow=0.0):
    s = np.zeros((N_TLS, N_FEAT), np.float32)
    s[:, F["phase_NS_straight"]] = 1
    s[:, F["time_in_phase"]] = time_in_phase
    s[:, F["yellow"]] = yellow
    return s


def test_rules_match_environment_constraints():
    rules = SignalRules(min_green=10, max_green=60, delta_t=5)
    ask = np.ones(N_TLS)
    assert rules.effective_action(raw_state(5), ask).sum() == 0          # verde mínimo
    assert rules.effective_action(raw_state(10), ask).sum() == N_TLS
    assert rules.effective_action(raw_state(20, yellow=1), ask).sum() == 0  # en transición
    assert rules.effective_action(raw_state(56), np.zeros(N_TLS)).sum() == N_TLS  # verde máximo forzado
    assert rules.effective_action(raw_state(55), np.zeros(N_TLS)).sum() == 0


def test_projection_gives_valid_states():
    rules = SignalRules(10, 60, 5)
    s = np.random.default_rng(0).normal(size=(4, N_TLS, N_FEAT)).astype(np.float32) * 50
    p = rules.project_state(s)
    assert (p >= 0).all()
    assert np.allclose(p[..., 7:11].sum(-1), 1)
    assert set(np.unique(p[..., F["yellow"]])) <= {0.0, 1.0}
    assert (p[..., F["halting"]] <= p[..., F["vehicles"]]).all()


def fake_windows(W=4, rows=50):
    rng = np.random.default_rng(0)
    r = rng.random((rows, N_TLS * N_FEAT + N_TLS)).astype(np.float32)
    return FlatWindows(r, np.zeros((rows, N_TLS * N_FEAT), np.float32), np.zeros((rows, N_TLS), np.float32),
                       np.arange(W - 1, rows), W)


def test_dream_env_truncates_and_bootstraps():
    env = DreamVecEnv(Persistence(N_TLS * N_FEAT, N_TLS), unit_norm(), fake_windows(), n_envs=3, episode_steps=5,
                      reward_scale=10.0)
    obs = env.reset()
    assert obs.shape == (3, N_TLS * N_FEAT)
    for k in range(5):
        env.step_async(np.ones((3, N_TLS), int))
        obs, rew, dones, infos = env.step_wait()
    assert dones.all() and all(i["TimeLimit.truncated"] and "terminal_observation" in i for i in infos)
    model = make_ppo(env, seed=0, n_envs=3)
    model.learn(64)


def test_planner_prefers_the_candidate_with_higher_imagined_return():
    class RewardsSwitching:
        name, state_dim = "fake", N_TLS * N_FEAT

        def predict(self, x):
            acts = x[:, -1, self.state_dim:]
            return np.zeros((len(x), self.state_dim), np.float32), acts.astype(np.float32)

    planner = ImaginationPlanner(RewardsSwitching(), unit_norm(), window=4, n_candidates=4, horizon=2,
                                 rules=SignalRules(10, 60, 5))
    window = np.zeros((4, N_TLS * N_FEAT + N_TLS), np.float32)
    window[:, [k * N_FEAT + F["phase_NS_straight"] for k in range(N_TLS)]] = 1
    window[:, [k * N_FEAT + F["time_in_phase"] for k in range(N_TLS)]] = 30
    cands = np.array([np.zeros(N_TLS, int), np.ones(N_TLS, int)])
    ret = planner.imagine(window, cands)
    assert ret[1] > ret[0]


def test_sumo_ppo_env_works_with_monitor(tmp_path):
    env = Monitor(SumoPPOEnv(unit_norm(), ["D1"], [1], 10.0, episode_seconds=25), str(tmp_path / "m"),
                  info_keywords=("demand", "sim_seed"))
    env.reset(seed=0)
    done = False
    while not done:
        _, _, term, trunc, info = env.step(env.action_space.sample())
        done = term or trunc
    assert info["episode"]["demand"] == "D1" and info["episode"]["l"] == 5
    env.close()


def test_learn_resumable_continues_after_interruption(tmp_path):
    """Un entrenamiento cortado tras el primer tramo se reanuda desde el punto de control hasta el total."""
    import gymnasium as gym
    from stable_baselines3 import PPO

    from wm.control.ppo import EpisodeCurve, learn_resumable

    env = Monitor(gym.make("CartPole-v1"))
    new = lambda: PPO("MlpPolicy", env, n_steps=64, batch_size=32, n_epochs=1, seed=0, device="cpu")

    first = learn_resumable(new, env, 128, tmp_path, EpisodeCurve(), rollouts_per_checkpoint=1)
    assert first.num_timesteps == 128 and (tmp_path / "checkpoint.zip").exists()
    rows_before = EpisodeCurve()
    learn_resumable(new, env, 64, tmp_path, rows_before, rollouts_per_checkpoint=1)   # ya completo: no entrena
    cb = EpisodeCurve()
    model = learn_resumable(new, env, 320, tmp_path, cb, rollouts_per_checkpoint=2)
    assert model.num_timesteps == 320
    assert cb.rows[:len(rows_before.rows)] == rows_before.rows            # la curva previa se conserva
    steps = [r["timesteps"] for r in cb.rows]
    assert steps == sorted(steps) and steps[-1] <= 320 and {"r", "l"} <= cb.rows[0].keys()
