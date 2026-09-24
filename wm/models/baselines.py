"""Etapa 4 — baselines predictivos: el piso que el modelo temporal debe superar.

Todos cumplen la interfaz Predictor (wm.models.predictor). En espacio normalizado la recompensa media de train
es 0, así que los baselines sin modelo de recompensa predicen r̂ = 0 (la media de train).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class Persistence:
    """ŝ_{t+1} = s_t."""
    name = "Persistencia"

    def __init__(self, state_dim: int, n_tls: int):
        self.state_dim, self.n_tls = state_dim, n_tls

    def predict(self, x):
        return np.zeros((len(x), self.state_dim), np.float32), np.zeros((len(x), self.n_tls), np.float32)


class MovingAverage:
    """ŝ_{t+1} = media de los estados de la ventana."""
    name = "Media móvil"

    def __init__(self, state_dim: int, n_tls: int):
        self.state_dim, self.n_tls = state_dim, n_tls

    def predict(self, x):
        s = x[:, :, :self.state_dim]
        return (s.mean(1) - s[:, -1]).astype(np.float32), np.zeros((len(x), self.n_tls), np.float32)


class Ridge:
    """Regresión ridge sobre la ventana aplanada -> [Δs, r]. Solución cerrada; alpha elegido en validación."""
    name = "Ridge"

    def __init__(self, state_dim: int, alpha: float = 1.0):
        self.state_dim, self.alpha = state_dim, alpha

    @staticmethod
    def _design(x: np.ndarray) -> np.ndarray:
        flat = x.reshape(len(x), -1).astype(np.float64)
        return np.concatenate([flat, np.ones((len(flat), 1))], axis=1)

    def fit_gram(self, gram: np.ndarray, cross: np.ndarray) -> "Ridge":
        reg = self.alpha * np.eye(len(gram))
        reg[-1, -1] = 0.0  # el sesgo no se regulariza
        self.coef = np.linalg.solve(gram + reg, cross).astype(np.float32)
        return self

    def predict(self, x):
        y = self._design(x).astype(np.float32) @ self.coef
        return y[:, :self.state_dim], y[:, self.state_dim:]


def ridge_gram(flat_windows, batch: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Acumula XᵀX y XᵀY por lotes (sin materializar todas las ventanas)."""
    gram = cross = None
    for i in range(0, len(flat_windows), batch):
        x, d, r = flat_windows.batch(np.arange(i, min(i + batch, len(flat_windows))))
        X = Ridge._design(x.numpy())
        Y = np.concatenate([d.numpy(), r.numpy()], axis=1).astype(np.float64)
        gram = X.T @ X if gram is None else gram + X.T @ X
        cross = X.T @ Y if cross is None else cross + X.T @ Y
    return gram, cross


class WindowMLP(nn.Module):
    """MLP sobre la ventana aplanada (sin estructura temporal explícita)."""

    def __init__(self, window: int, input_dim: int, state_dim: int, n_tls: int, hidden: list[int], dropout: float):
        super().__init__()
        layers, d = [], window * input_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, state_dim + n_tls))
        self.net = nn.Sequential(*layers)
        self.state_dim = state_dim

    def forward(self, x: torch.Tensor):
        out = self.net(x.flatten(1))
        return out[:, :self.state_dim], out[:, self.state_dim:]
