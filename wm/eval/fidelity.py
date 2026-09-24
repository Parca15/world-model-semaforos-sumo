"""Fidelidad del World Model (PLAN_DE_TRABAJO.md, sección 4.4).

1. `logged_fidelity`: en segmentos de 40 pasos de los episodios de test, retorno imaginado (rollout con las
   acciones registradas, como en el DreamEnv) frente al retorno real -> correlación y brecha sueño-realidad.
2. `ranking_fidelity`: en estados reales de SUMO, K acciones candidatas se evalúan con h pasos imaginados y
   con h pasos REALES; se mide la concordancia de rankings con Kendall τ. Los retornos reales se obtienen
   por repetición determinista: misma semilla + mismas acciones hasta el instante de decisión + candidata.
   (`loadState` de SUMO 1.20 no reproduce exactamente la simulación, por eso no se usa.)
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
from scipy import stats

from wm.control.planning import ImaginationPlanner
from wm.control.policies import MaxPressure
from wm.data.base import WindowDataset
from wm.data.normalization import NormStats
from wm.dream.rules import SignalRules
from wm.env.traffic_env import make_env
from wm.eval.rollout import rollout


# ------------------------------------------------------------ 1. trayectorias registradas
def logged_fidelity(predictor, ds40: WindowDataset, norm: NormStats, segments_per_episode: int) -> dict:
    rules = SignalRules.from_config()
    n_tls, n_feat = norm.state_mean.shape
    sd, W, H = n_tls * n_feat, ds40.W, ds40.H
    x0, fa, real = [], [], []
    for e, ep in enumerate(ds40.episodes):
        ts = ds40.windows[ds40.windows[:, 0] == e, 1]
        for t in ts[np.linspace(0, len(ts) - 1, segments_per_episode).astype(int)]:
            s = ep.states.reshape(len(ep.states), -1)
            x0.append(np.concatenate([s[t - W + 1:t + 1], ep.actions[t - W + 1:t + 1]], axis=1))
            fa.append(ep.actions[t + 1:t + H])
            real.append(norm.denorm_reward(ep.rewards[t:t + H]).sum())

    def project(s_norm):
        raw = rules.project_state(norm.denorm_state(s_norm.reshape(-1, n_tls, n_feat)))
        return norm.norm_state(raw).reshape(len(s_norm), sd)

    _, pr = rollout(predictor, np.asarray(x0, np.float32), np.asarray(fa, np.float32), H, project=project)
    imagined = norm.denorm_reward(pr).sum(axis=(1, 2))
    real = np.asarray(real)
    return {"n_segments": len(real), "horizon": H,
            "pearson": float(stats.pearsonr(imagined, real)[0]),
            "spearman": float(stats.spearmanr(imagined, real)[0]),
            "gap_mean": float(np.mean(imagined - real)), "gap_mae": float(np.mean(np.abs(imagined - real))),
            "real_mean": float(real.mean()), "imagined_mean": float(imagined.mean())}


# ------------------------------------------------------------ 2. ranking de acciones
@dataclass
class DecisionPoint:
    demand: str
    seed: int
    prefix: list[np.ndarray]     # acciones pedidas hasta el instante de decisión (se repiten tal cual)
    window: np.ndarray           # ventana normalizada [W, input_dim] en el instante de decisión
    candidates: np.ndarray       # [K, n_tls]


def decision_point(demand: str, seed: int, t_decision: int, n_candidates: int, norm: NormStats,
                   window: int) -> DecisionPoint:
    """Corre Max-Pressure (ε = 0,1) hasta t_decision y guarda las acciones pedidas y la ventana."""
    env = make_env(demand=demand, control="agent")
    policy = MaxPressure(epsilon=0.1)
    rng = np.random.default_rng([seed, t_decision, zlib.crc32(demand.encode())])
    try:
        obs, _ = env.reset(seed=seed)
        policy.reset(env, rng)
        states, acts, prefix = [norm.norm_state(obs).reshape(-1)], [], []
        for _ in range(t_decision):
            a = policy.act(obs, env)
            obs, _, _, _, info = env.step(a)
            prefix.append(a)
            acts.append(info["action"].astype(np.float32))
            states.append(norm.norm_state(obs).reshape(-1))
    finally:
        env.close()
    rows_s = np.stack(states[-window:])
    rows_a = np.stack(acts[-(window - 1):] + [np.zeros_like(acts[0])])
    n_tls = len(acts[0])
    cands = {tuple(np.zeros(n_tls, int)), tuple(np.ones(n_tls, int))}
    while len(cands) < n_candidates:
        cands.add(tuple(rng.integers(0, 2, n_tls)))
    return DecisionPoint(demand, seed, prefix, np.concatenate([rows_s, rows_a], 1).astype(np.float32),
                         np.array(sorted(cands)))


def real_return(job: tuple[str, int, list[np.ndarray], np.ndarray, int]) -> float:
    """Retorno real de h pasos aplicando la candidata tras repetir el prefijo (determinista)."""
    demand, seed, prefix, candidate, h = job
    env = make_env(demand=demand, control="agent")
    try:
        env.reset(seed=seed)
        for a in prefix:
            env.step(a)
        total = 0.0
        for k in range(h):
            _, _, _, _, info = env.step(candidate if k == 0 else np.zeros_like(candidate))
            total += float(info["reward_vec"].sum())
    finally:
        env.close()
    return total


def kendall(imagined: np.ndarray, real: np.ndarray) -> float:
    tau = stats.kendalltau(imagined, real).statistic
    return float(0.0 if np.isnan(tau) else tau)


def imagined_returns(predictor, norm: NormStats, point: DecisionPoint, h: int) -> np.ndarray:
    planner = ImaginationPlanner(predictor, norm, point.window.shape[0], len(point.candidates), h)
    return planner.imagine(point.window, point.candidates)
