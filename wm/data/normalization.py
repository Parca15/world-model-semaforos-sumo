"""Normalización z-score por variable (intersección × característica), ajustada SOLO con train.

Las variables categóricas (fase one-hot, indicador de amarillo) no se normalizan: media 0 y desviación 1.
Los modelos trabajan en el espacio normalizado y predicen Δs en ese espacio:
    ŝ_{t+1} = s_t + Δ̂s   (normalizado)   <=>   Δs_original = Δ̂s · std.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MIN_STD = 1e-6


@dataclass
class NormStats:
    state_mean: np.ndarray   # [7, 13]
    state_std: np.ndarray    # [7, 13]
    reward_mean: np.ndarray  # [7]
    reward_std: np.ndarray   # [7]

    @classmethod
    def fit(cls, states: list[np.ndarray], rewards: list[np.ndarray], normalize_mask: np.ndarray) -> "NormStats":
        """states: lista de [T+1, 7, 13]; rewards: lista de [T, 7]; normalize_mask: [13] booleano."""
        s = np.concatenate(states).astype(np.float64)
        r = np.concatenate(rewards).astype(np.float64)
        mean, std = s.mean(0), s.std(0)
        std = np.where(std < MIN_STD, 1.0, std)
        mean[:, ~normalize_mask], std[:, ~normalize_mask] = 0.0, 1.0
        r_std = r.std(0)
        return cls(mean.astype(np.float32), std.astype(np.float32),
                   r.mean(0).astype(np.float32), np.where(r_std < MIN_STD, 1.0, r_std).astype(np.float32))

    # --- transformaciones
    def norm_state(self, s: np.ndarray) -> np.ndarray:
        return ((s - self.state_mean) / self.state_std).astype(np.float32)

    def denorm_state(self, z: np.ndarray) -> np.ndarray:
        return (z * self.state_std + self.state_mean).astype(np.float32)

    def denorm_delta(self, dz: np.ndarray) -> np.ndarray:
        return (dz * self.state_std).astype(np.float32)

    def norm_reward(self, r: np.ndarray) -> np.ndarray:
        return ((r - self.reward_mean) / self.reward_std).astype(np.float32)

    def denorm_reward(self, z: np.ndarray) -> np.ndarray:
        return (z * self.reward_std + self.reward_mean).astype(np.float32)

    # --- persistencia
    def save(self, path: Path, feature_names: list[str], tls_ids: list[str]) -> None:
        payload = {
            "fitted_on": "train",
            "shape": list(self.state_mean.shape),
            "tls_ids": tls_ids,
            "features": feature_names,
            **{k: getattr(self, k).tolist() for k in ("state_mean", "state_std", "reward_mean", "reward_std")},
        }
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "NormStats":
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(*(np.asarray(d[k], dtype=np.float32) for k in ("state_mean", "state_std", "reward_mean", "reward_std")))
