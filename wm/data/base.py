"""Cargador común del dataset base: ÚNICO punto de acceso a los datos para AE/VAE, LSTM, TSMixer, Transformer
y el Dream Environment.

    ds = load_base("base_v1", "train", W=12, H=1)
    item = ds[0]      # dict de tensores (ver WindowDataset.__getitem__)
    X, dS, R = ds.arrays()   # matrices numpy para baselines (ridge, MLP)

Cada muestra es una ventana que termina en el paso t de un episodio:
    x              [W, n_tls·F + n_tls]   estados normalizados s_{t-W+1..t} (aplanados) y acciones a_{t-W+1..t}
    delta          [n_tls·F]              Δs_{t+1} = s_{t+1} − s_t (normalizado)
    reward         [n_tls]                r_t (normalizada)
Con H > 1 además (para evaluar a varios pasos de forma autorregresiva):
    future_states  [H, n_tls·F]           s_{t+1..t+H} (normalizados)
    future_actions [H-1, n_tls]           a_{t+1..t+H-1}
    future_rewards [H, n_tls]             r_{t..t+H-1} (normalizadas)
Ninguna ventana cruza el límite de un episodio y los conjuntos se definen por episodio (semilla).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from wm.data.manifest import verify_manifest
from wm.data.normalization import NormStats
from wm.utils import DATA_DIR

SPLITS = ("train", "val", "test", "test_ood")


@lru_cache(maxsize=None)
def _verified(root: str) -> int:
    """Verifica el manifiesto una sola vez por proceso y versión."""
    return verify_manifest(Path(root))


@dataclass
class Episode:
    episode: str
    states: np.ndarray    # [T+1, n_tls, F] normalizado
    actions: np.ndarray   # [T, n_tls]
    rewards: np.ndarray   # [T, n_tls] normalizado

    @property
    def T(self) -> int:
        return len(self.actions)


class WindowDataset(Dataset):
    def __init__(self, episodes: list[Episode], index: pd.DataFrame, norm: NormStats, schema: dict,
                 split: str, W: int, H: int):
        assert W >= 1 and H >= 1
        self.episodes, self.index, self.norm, self.schema = episodes, index, norm, schema
        self.split, self.W, self.H = split, W, H
        n_tls, n_feat = episodes[0].states.shape[1:]
        self.n_tls, self.n_features = n_tls, n_feat
        self.state_dim = n_tls * n_feat
        self.input_dim = self.state_dim + n_tls
        # (episodio, t): la ventana usa s_{t-W+1..t} y necesita s_{t+H}
        self.windows = np.array([(e, t) for e, ep in enumerate(episodes) for t in range(W - 1, ep.T - H + 1)],
                                dtype=np.int64).reshape(-1, 2)

    def __len__(self) -> int:
        return len(self.windows)

    def _flat(self, s: np.ndarray) -> np.ndarray:
        return s.reshape(len(s), -1)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        e, t = self.windows[i]
        ep, W, H = self.episodes[e], self.W, self.H
        s_win = self._flat(ep.states[t - W + 1:t + 1])
        a_win = ep.actions[t - W + 1:t + 1].astype(np.float32)
        item = {
            "x": np.concatenate([s_win, a_win], axis=1),
            "delta": (ep.states[t + 1] - ep.states[t]).reshape(-1),
            "reward": ep.rewards[t],
        }
        if H > 1:
            item["future_states"] = self._flat(ep.states[t + 1:t + H + 1])
            item["future_actions"] = ep.actions[t + 1:t + H].astype(np.float32)
            item["future_rewards"] = ep.rewards[t:t + H]
        return {k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)) for k, v in item.items()}

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """X [N, W, input_dim], ΔS [N, state_dim], R [N, n_tls] como numpy (para baselines)."""
        X = np.empty((len(self), self.W, self.input_dim), np.float32)
        dS = np.empty((len(self), self.state_dim), np.float32)
        R = np.empty((len(self), self.n_tls), np.float32)
        for i, (e, t) in enumerate(self.windows):
            ep = self.episodes[e]
            X[i, :, :self.state_dim] = self._flat(ep.states[t - self.W + 1:t + 1])
            X[i, :, self.state_dim:] = ep.actions[t - self.W + 1:t + 1]
            dS[i] = (ep.states[t + 1] - ep.states[t]).reshape(-1)
            R[i] = ep.rewards[t]
        return X, dS, R

    def flat(self) -> "FlatWindows":
        """Vista compacta para entrenar: las ventanas se arman por indexación, sin copiar W veces cada fila."""
        rows, delta, reward, ends, offset = [], [], [], [], 0
        for e, ep in enumerate(self.episodes):
            s = self._flat(ep.states)
            rows.append(np.concatenate([s[:-1], ep.actions.astype(np.float32)], axis=1))
            delta.append(s[1:] - s[:-1])
            reward.append(ep.rewards)
            ends.append(offset + self.windows[self.windows[:, 0] == e, 1])
            offset += ep.T
        return FlatWindows(np.concatenate(rows), np.concatenate(delta), np.concatenate(reward),
                           np.concatenate(ends), self.W)

    @property
    def feature_names(self) -> list[str]:
        return [f["name"] for f in self.schema["features"]]


@dataclass
class FlatWindows:
    rows: np.ndarray     # [R, input_dim]  fila t = (s_t, a_t) de todos los episodios concatenados
    delta: np.ndarray    # [R, state_dim]  Δs_{t+1}
    reward: np.ndarray   # [R, n_tls]      r_t
    ends: np.ndarray     # [N]             fila final de cada ventana válida (nunca cruza episodios)
    W: int

    def __len__(self) -> int:
        return len(self.ends)

    def batch(self, idx: np.ndarray, k: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x [B, W, input_dim] y los k objetivos siguientes: Δs [B, k, state_dim] (o [B, state_dim] si k = 1),
        r [B, k, n_tls] (o [B, n_tls]). Con k > 1 las filas e+1..e+k-1 aportan las acciones futuras reales;
        la vista debe construirse con H >= k para que nunca crucen el final del episodio."""
        e = self.ends[idx]
        rows = e[:, None] + np.arange(-self.W + 1, 1)
        if k == 1:
            return (torch.from_numpy(self.rows[rows]), torch.from_numpy(self.delta[e]),
                    torch.from_numpy(self.reward[e]))
        fut = e[:, None] + np.arange(k)
        return (torch.from_numpy(self.rows[rows]), torch.from_numpy(self.delta[fut]),
                torch.from_numpy(self.reward[fut]), torch.from_numpy(self.rows[fut]))


def dataset_root(version: str, data_dir: Path = DATA_DIR) -> Path:
    root = Path(data_dir) / version
    if not root.exists():
        raise FileNotFoundError(f"No existe {root}. Constrúyalo con: python -m wm.data.build_base --version {version}")
    return root


def load_norm(version: str, data_dir: Path = DATA_DIR) -> NormStats:
    return NormStats.load(dataset_root(version, data_dir) / "norm_stats.json")


def load_base(version: str, split: str, W: int, H: int = 1, *, normalize: bool = True,
              verify: bool = True, data_dir: Path = DATA_DIR) -> WindowDataset:
    if split not in SPLITS:
        raise ValueError(f"split debe ser uno de {SPLITS}")
    root = dataset_root(version, data_dir)
    if verify:
        _verified(str(root))
    index = pd.read_parquet(root / "index.parquet")
    split_eps = json.loads((root / "splits.json").read_text(encoding="utf-8"))[split]
    index = index.set_index("episode").loc[split_eps].reset_index()
    norm = NormStats.load(root / "norm_stats.json")
    schema = json.loads((root / "feature_schema.json").read_text(encoding="utf-8"))
    episodes = []
    for row in index.itertuples():
        with np.load(root / row.file) as z:
            s, a, r = z["states"], z["actions"], z["rewards"]
        if normalize:
            s, r = norm.norm_state(s), norm.norm_reward(r)
        episodes.append(Episode(row.episode, s.astype(np.float32), a.astype(np.int8), r.astype(np.float32)))
    return WindowDataset(episodes, index, norm, schema, split, W, H)
