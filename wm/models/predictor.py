"""Interfaz común de predicción a 1 paso, usada por la evaluación, los rollouts, el DreamEnv y la planificación.

    predictor.predict(x) -> (Δŝ, r̂)
        x   [B, W, state_dim + n_tls]  ventana normalizada (estados aplanados + acciones)
        Δŝ  [B, state_dim]             en espacio normalizado
        r̂   [B, n_tls]                 recompensa normalizada
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
from torch import nn


class Predictor(Protocol):
    name: str
    state_dim: int

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


class TorchPredictor:
    """Adapta un nn.Module con forward(x) -> (Δŝ, r̂) a la interfaz Predictor."""

    def __init__(self, model: nn.Module, name: str, state_dim: int, batch_size: int = 4096):
        self.model = model.eval()
        self.name, self.state_dim, self.batch_size = name, state_dim, batch_size

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        outs_d, outs_r = [], []
        for i in range(0, len(x), self.batch_size):
            d, r = self.model(torch.from_numpy(np.ascontiguousarray(x[i:i + self.batch_size], dtype=np.float32)))
            outs_d.append(d.numpy())
            outs_r.append(r.numpy())
        return np.concatenate(outs_d), np.concatenate(outs_r)
