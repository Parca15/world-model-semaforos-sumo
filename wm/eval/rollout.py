"""Rollouts autorregresivos a H pasos con cualquier Predictor (las predicciones se reutilizan como entrada)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from wm.data.base import WindowDataset
from wm.models.predictor import Predictor


@dataclass
class HorizonArrays:
    """Ventanas de un conjunto con su continuación real de H pasos (todo normalizado)."""
    x0: np.ndarray            # [N, W, input_dim]  ventana que termina en t (incluye a_t)
    future_states: np.ndarray  # [N, H, state_dim]  s_{t+1..t+H}
    future_actions: np.ndarray # [N, H-1, n_tls]    a_{t+1..t+H-1}
    future_rewards: np.ndarray # [N, H, n_tls]      r_{t..t+H-1}
    episode: np.ndarray       # [N] índice del episodio (para estadística por episodio)

    @property
    def current_state(self) -> np.ndarray:
        return self.x0[:, -1, :self.future_states.shape[2]]


def horizon_arrays(ds: WindowDataset, stride: int = 1) -> HorizonArrays:
    W, H = ds.W, ds.H
    win = ds.windows[::stride]
    x0, fs, fa, fr = [], [], [], []
    for e, t in win:
        ep = ds.episodes[e]
        s = ep.states.reshape(len(ep.states), -1)
        x0.append(np.concatenate([s[t - W + 1:t + 1], ep.actions[t - W + 1:t + 1]], axis=1))
        fs.append(s[t + 1:t + H + 1])
        fa.append(ep.actions[t + 1:t + H])
        fr.append(ep.rewards[t:t + H])
    return HorizonArrays(np.asarray(x0, np.float32), np.asarray(fs, np.float32), np.asarray(fa, np.float32),
                         np.asarray(fr, np.float32), win[:, 0].copy())


def rollout(predictor: Predictor, x0: np.ndarray, future_actions: np.ndarray, H: int,
            project: Callable[[np.ndarray], np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve estados [N, H, state_dim] y recompensas [N, H, n_tls] predichos (normalizados)."""
    sd = predictor.state_dim
    x = x0.copy()
    states, rewards = [], []
    for h in range(H):
        delta, r = predictor.predict(x)
        s_next = x[:, -1, :sd] + delta
        if project is not None:
            s_next = project(s_next)
        states.append(s_next)
        rewards.append(r)
        if h < H - 1:
            row = np.concatenate([s_next, future_actions[:, h]], axis=1)[:, None]
            x = np.concatenate([x[:, 1:], row], axis=1)
    return np.stack(states, 1), np.stack(rewards, 1)
