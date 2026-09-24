"""PPO (Stable-Baselines3) con los MISMOS hiperparámetros en el Dream Environment y en SUMO directo.

* `make_ppo`: construye PPO desde configs/ppo.yaml (rollout_size = n_steps × n_envs es igual en ambos casos).
* `SumoPPOEnv`: TrafficEnvironment con la misma observación (estado normalizado aplanado) y la misma escala de
  recompensa que el DreamEnv; cada reinicio elige demanda y semilla de simulación de train.
* `PPOPolicy`: una política PPO entrenada con la interfaz `Policy` (para evaluarla en SUMO).
"""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from wm.control.policies import Policy
from wm.data.normalization import NormStats
from wm.env.traffic_env import TrafficEnvironment, make_env
from wm.utils import load_config


def make_ppo(env, seed: int, n_envs: int, tensorboard_log: str | None = None) -> PPO:
    hp = load_config("ppo")["hyperparams"]
    return PPO("MlpPolicy", env, seed=seed, device="cpu", verbose=0,
               learning_rate=hp["learning_rate"], n_steps=hp["rollout_size"] // n_envs,
               batch_size=hp["batch_size"], n_epochs=hp["n_epochs"], gamma=hp["gamma"],
               gae_lambda=hp["gae_lambda"], clip_range=hp["clip_range"], ent_coef=hp["ent_coef"],
               vf_coef=hp["vf_coef"], max_grad_norm=hp["max_grad_norm"],
               policy_kwargs={"net_arch": {"pi": hp["net_arch"], "vf": hp["net_arch"]}},
               tensorboard_log=tensorboard_log)


class SumoPPOEnv(gym.Env):
    """Envoltura de SUMO para PPO directo (Condición 2)."""

    def __init__(self, norm: NormStats, demands: list[str], seeds: list[int], reward_scale: float,
                 episode_seconds: int | None = None):
        self.env: TrafficEnvironment = make_env(control="agent", episode_seconds=episode_seconds)
        self.norm, self.demands, self.sim_seeds, self.reward_scale = norm, demands, seeds, reward_scale
        n = norm.state_mean.size
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(n,), dtype=np.float32)
        self.action_space = self.env.action_space

    def _obs(self, s: np.ndarray) -> np.ndarray:
        return self.norm.norm_state(s).reshape(-1)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        demand = self.demands[self.np_random.integers(len(self.demands))]
        sim_seed = int(self.sim_seeds[self.np_random.integers(len(self.sim_seeds))])
        obs, info = self.env.reset(seed=sim_seed, options={"demand": demand})
        self.episode_info = {"demand": demand, "sim_seed": sim_seed}
        return self._obs(obs), {**info, **self.episode_info}

    def step(self, action):
        obs, _, term, trunc, info = self.env.step(action)
        reward = float(info["reward_vec"].sum()) / self.reward_scale
        return self._obs(obs), reward, term, trunc, {**info, **self.episode_info}

    def close(self):
        self.env.close()


class PPOPolicy(Policy):
    """Política PPO entrenada (en el sueño o en SUMO), evaluada de forma determinista."""

    def __init__(self, path: Path, norm: NormStats):
        self.model = PPO.load(str(path), device="cpu")
        self.norm = norm

    def act(self, obs, env):
        action, _ = self.model.predict(self.norm.norm_state(obs).reshape(-1), deterministic=True)
        return np.asarray(action, dtype=np.int64)
