"""Planificación por imaginación (complementaria, Etapa 7): en cada decisión se imaginan K acciones conjuntas
candidatas con rollouts de horizonte h usando el modelo temporal, y se ejecuta la de mayor retorno imaginado.

Candidatas: mantener todo, la acción de la política PPO (si hay) y el resto aleatorias. Tras el primer paso
los rollouts continúan con "mantener" (con las reglas del semáforo, que fuerzan el cambio en verde máximo).
"""
from __future__ import annotations

import numpy as np

from wm.control.policies import Policy
from wm.data.normalization import NormStats
from wm.dream.rules import SignalRules
from wm.models.predictor import Predictor


class ImaginationPlanner(Policy):
    def __init__(self, predictor: Predictor, norm: NormStats, window: int, n_candidates: int, horizon: int,
                 base_policy: Policy | None = None, rules: SignalRules | None = None):
        self.predictor, self.norm, self.W = predictor, norm, window
        self.K, self.h, self.base = n_candidates, horizon, base_policy
        self.rules = rules or SignalRules.from_config()
        self.n_tls, self.n_feat = norm.state_mean.shape

    def reset(self, env, rng):
        super().reset(env, rng)
        if self.base:
            self.base.reset(env, rng)
        self.hist_s: list[np.ndarray] = []   # estados normalizados aplanados
        self.hist_a: list[np.ndarray] = []   # acciones efectivas

    def observe(self, info: dict) -> None:
        self.hist_a.append(info["action"].astype(np.float32))

    def _window(self, s_now: np.ndarray) -> np.ndarray:
        states = (self.hist_s + [s_now])[-self.W:]
        acts = (self.hist_a[-(self.W - 1):] if self.W > 1 else []) + [np.zeros(self.n_tls, np.float32)]
        pad = self.W - len(states)                 # al inicio del episodio se repite el primer estado
        states = [states[0]] * pad + states
        acts = [np.zeros(self.n_tls, np.float32)] * (self.W - len(acts)) + acts
        return np.concatenate([np.stack(states), np.stack(acts)], axis=1)

    def imagine(self, window: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """Retorno imaginado (unidades originales) de cada candidata en `h` pasos."""
        K, sd = len(candidates), self.n_tls * self.n_feat
        x = np.repeat(window[None], K, axis=0).astype(np.float32)
        ret = np.zeros(K)
        for step in range(self.h):
            raw = self.norm.denorm_state(x[:, -1, :sd].reshape(K, self.n_tls, self.n_feat))
            req = candidates if step == 0 else np.zeros_like(candidates)
            x[:, -1, sd:] = self.rules.effective_action(raw, req)
            delta, r = self.predictor.predict(x)
            ret += self.norm.denorm_reward(r).sum(1)
            raw_next = self.rules.project_state(
                self.norm.denorm_state((x[:, -1, :sd] + delta).reshape(K, self.n_tls, self.n_feat)))
            row = np.zeros((K, 1, x.shape[2]), np.float32)
            row[:, 0, :sd] = self.norm.norm_state(raw_next).reshape(K, -1)
            x = np.concatenate([x[:, 1:], row], axis=1)
        return ret

    def act(self, obs, env):
        s_now = self.norm.norm_state(obs).reshape(-1)
        cands = [np.zeros(self.n_tls, np.int64)]
        if self.base:
            cands.append(self.base.act(obs, env))
        while len(cands) < self.K:
            cands.append(self.rng.integers(0, 2, self.n_tls))
        cands = np.unique(np.stack(cands), axis=0)
        best = cands[int(np.argmax(self.imagine(self._window(s_now), cands)))]
        self.hist_s.append(s_now)
        return best
