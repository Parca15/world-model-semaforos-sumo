"""Métricas de precisión predictiva (PLAN_DE_TRABAJO.md, sección 4.1)."""
from __future__ import annotations

import numpy as np

from wm.data.normalization import NormStats
from wm.eval.rollout import HorizonArrays, rollout
from wm.models.predictor import Predictor

PHASE = slice(7, 11)       # índices de la fase one-hot dentro de las 13 variables
QUEUE_IDX = 2              # queue_m


def _rmse(a, b, axis=None):
    return np.sqrt(np.mean((a - b) ** 2, axis=axis))


def evaluate(predictor: Predictor, arr: HorizonArrays, norm: NormStats, feature_names: list[str],
             congestion_queue_m: float) -> dict:
    """Evalúa un predictor sobre ventanas con continuación H. Devuelve métricas escalares, tablas y
    valores por episodio (para las pruebas estadísticas)."""
    n_tls, n_feat = norm.state_mean.shape
    H = arr.future_states.shape[1]
    pred_s, pred_r = rollout(predictor, arr.x0, arr.future_actions, H)
    true_s, true_r = arr.future_states, arr.future_rewards

    # --- error multi-paso (normalizado, todas las variables)
    rmse_h = np.array([_rmse(pred_s[:, h], true_s[:, h]) for h in range(H)])
    out = {
        "rmse_h": rmse_h.tolist(),
        "auc_rmse": float(np.trapz(rmse_h, dx=1.0)),
        "growth_ratio": float(rmse_h[-1] / rmse_h[0]),
    }

    # --- 1 paso
    p1, y1 = pred_s[:, 0].reshape(-1, n_tls, n_feat), true_s[:, 0].reshape(-1, n_tls, n_feat)
    out["mae_1"] = float(np.mean(np.abs(p1 - y1)))
    out["rmse_1"] = float(_rmse(p1, y1))
    out["rmse_1_by_tls"] = _rmse(p1, y1, axis=(0, 2)).tolist()

    raw_p, raw_y = norm.denorm_state(p1), norm.denorm_state(y1)
    per_var = []
    for f, name in enumerate(feature_names):
        a, b = raw_p[..., f].ravel(), raw_y[..., f].ravel()
        den = np.abs(a) + np.abs(b)
        ok = den > 0
        sst = np.sum((b - b.mean()) ** 2)
        per_var.append({
            "variable": name,
            "mae": float(np.mean(np.abs(a - b))),
            "rmse": float(_rmse(a, b)),
            "nmae": float(np.mean(np.abs(p1[..., f] - y1[..., f]))),
            "smape_pct": float(100 * np.mean(2 * np.abs(a - b)[ok] / den[ok])) if ok.any() else 0.0,
            "r2": float(1 - np.sum((a - b) ** 2) / sst) if sst > 0 else float("nan"),
        })
    out["per_variable"] = per_var

    # --- recompensa (unidades originales)
    rp, ry = norm.denorm_reward(pred_r[:, 0]), norm.denorm_reward(true_r[:, 0])
    out["reward_mae_1"] = float(np.mean(np.abs(rp - ry)))
    out["reward_rmse_1"] = float(_rmse(rp, ry))
    out["reward_rmse_1_by_tls"] = _rmse(rp, ry, axis=0).tolist()

    # --- exactitud de la fase predicha (argmax del one-hot)
    for h in (0, H - 1):
        ph_p = pred_s[:, h].reshape(-1, n_tls, n_feat)[..., PHASE].argmax(-1)
        ph_y = true_s[:, h].reshape(-1, n_tls, n_feat)[..., PHASE].argmax(-1)
        out[f"phase_acc_{h + 1}"] = float(np.mean(ph_p == ph_y))

    # --- régimen congestionado: intersecciones con cola actual > percentil 75 de train
    cur = norm.denorm_state(arr.current_state.reshape(-1, n_tls, n_feat))[..., QUEUE_IDX]
    mask = cur > congestion_queue_m
    out["rmse_1_congested"] = float(_rmse(p1[mask], y1[mask])) if mask.any() else float("nan")
    out["congested_fraction"] = float(mask.mean())

    # --- por episodio (para Wilcoxon pareado)
    eps = np.unique(arr.episode)
    out["per_episode"] = {
        "episode": eps.tolist(),
        "rmse_1": [float(_rmse(pred_s[arr.episode == e, 0], true_s[arr.episode == e, 0])) for e in eps],
        "rmse_H": [float(_rmse(pred_s[arr.episode == e, -1], true_s[arr.episode == e, -1])) for e in eps],
    }
    return out


def skill(model_rmse: float, persistence_rmse: float) -> float:
    """Skill score frente a persistencia: 1 − RMSE_modelo / RMSE_persistencia (>0 = mejor)."""
    return 1.0 - model_rmse / persistence_rmse
