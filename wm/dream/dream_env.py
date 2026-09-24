"""Dream Environment: entorno imaginado que avanza con el modelo temporal en lugar de SUMO.

Implementa directamente la interfaz VecEnv de Stable-Baselines3 para simular `n_envs` episodios imaginados
con UNA sola pasada del modelo por paso (lote), que es lo que hace viable entrenar PPO en CPU.

* Reinicio: ventana REAL de W pasos muestreada de train del dataset base (estados + acciones efectivas).
* Paso: acción pedida -> acción efectiva (reglas del semáforo) -> modelo -> ŝ_{t+1} = s_t + Δ̂s (proyectado a
  rangos físicos) y r̂_t. Recompensa para PPO: Σ_i r̂_t^i / reward_scale (misma escala que en SUMO).
* Observación: s_t normalizado y aplanado (91), igual que en el wrapper de SUMO.
* Los episodios imaginados duran `episode_steps` pasos y terminan por truncamiento (PPO hace bootstrap).
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from wm.data.base import FlatWindows
from wm.data.normalization import NormStats
from wm.dream.rules import SignalRules
from wm.models.predictor import Predictor


class DreamVecEnv(VecEnv):
    render_mode = None

    def __init__(self, predictor: Predictor, norm: NormStats, reset_windows: FlatWindows, n_envs: int,
                 episode_steps: int, reward_scale: float, rules: SignalRules | None = None, seed: int = 0):
        self.predictor, self.norm, self.windows = predictor, norm, reset_windows
        self.n_tls, self.n_feat = norm.state_mean.shape
        self.sd = self.n_tls * self.n_feat
        self.episode_steps, self.reward_scale = episode_steps, reward_scale
        self.rules = rules or SignalRules.from_config()
        self.rng = np.random.default_rng(seed)
        obs_space = spaces.Box(-np.inf, np.inf, shape=(self.sd,), dtype=np.float32)
        super().__init__(n_envs, obs_space, spaces.MultiDiscrete([2] * self.n_tls))
        self.x = np.zeros((n_envs, reset_windows.W, reset_windows.rows.shape[1]), np.float32)
        self.t = np.zeros(n_envs, np.int64)
        self.ep_return = np.zeros(n_envs)
        self._actions = None

    # --------------------------------------------------------- utilidades
    def _raw(self, s_norm: np.ndarray) -> np.ndarray:
        return self.norm.denorm_state(s_norm.reshape(-1, self.n_tls, self.n_feat))

    def _reset_env(self, i: int) -> None:
        e = self.windows.ends[self.rng.integers(len(self.windows))]
        self.x[i] = self.windows.rows[e - self.windows.W + 1:e + 1]
        self.t[i] = 0
        self.ep_return[i] = 0.0

    def _obs(self) -> np.ndarray:
        return self.x[:, -1, :self.sd].copy()

    # --------------------------------------------------------- VecEnv
    def reset(self):
        for i in range(self.num_envs):
            self._reset_env(i)
        return self._obs()

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions).reshape(self.num_envs, self.n_tls)

    def step_wait(self):
        s = self.x[:, -1, :self.sd]
        raw = self._raw(s)
        self.x[:, -1, self.sd:] = self.rules.effective_action(raw, self._actions)
        delta, r = self.predictor.predict(self.x)
        raw_next = self.rules.project_state(self._raw(s + delta))
        s_next = self.norm.norm_state(raw_next).reshape(self.num_envs, -1)
        reward = self.norm.denorm_reward(r).sum(1) / self.reward_scale

        self.x = np.concatenate([self.x[:, 1:], np.zeros_like(self.x[:, :1])], axis=1)
        self.x[:, -1, :self.sd] = s_next
        self.t += 1
        self.ep_return += reward
        dones = self.t >= self.episode_steps
        infos = [{} for _ in range(self.num_envs)]
        for i in np.flatnonzero(dones):
            infos[i] = {"terminal_observation": s_next[i].copy(), "TimeLimit.truncated": True,
                        "episode": {"r": float(self.ep_return[i]), "l": int(self.t[i])}}
            self._reset_env(i)
        return self._obs(), reward.astype(np.float32), dones, infos

    def close(self) -> None:
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name)] * len(self._get_indices(indices))

    def set_attr(self, attr_name, value, indices=None) -> None:
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [getattr(self, method_name)(*args, **kwargs)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._get_indices(indices))

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)
        return [seed] * self.num_envs
