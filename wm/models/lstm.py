"""LSTM como modelo temporal del World Model (determinista, sin MDN).

Entrada [B, W, input_dim] (estados normalizados aplanados + acciones) -> Δŝ_{t+1} [B, state_dim], r̂_t [B, n_tls].

    n_layers × LSTM(hidden) con dropout entre capas
    último estado oculto h_W -> dropout -> cabeza densa (hidden -> state_dim + n_tls)

El tamaño oculto no se busca: se calcula para cumplir el presupuesto de parámetros.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class LSTMConfig:
    input_dim: int = 98
    state_dim: int = 91
    n_tls: int = 7
    window: int = 12
    n_layers: int = 2
    hidden: int = 128
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


class LSTMModel(nn.Module):
    def __init__(self, cfg: LSTMConfig):
        super().__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(cfg.input_dim, cfg.hidden, num_layers=cfg.n_layers, batch_first=True,
                            dropout=cfg.dropout if cfg.n_layers > 1 else 0.0)
        self.drop = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.hidden, cfg.state_dim + cfg.n_tls)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, _ = self.lstm(x)
        out = self.head(self.drop(h[:, -1]))
        return out[:, :self.cfg.state_dim], out[:, self.cfg.state_dim:]


def lstm_params(cfg: LSTMConfig) -> int:
    """Parámetros entrenables: cada capa tiene 4 compuertas con pesos de entrada, recurrentes y 2 sesgos."""
    h, out = cfg.hidden, cfg.state_dim + cfg.n_tls
    layers = 4 * h * (cfg.input_dim + h + 2) + (cfg.n_layers - 1) * 4 * h * (2 * h + 2)
    return layers + h * out + out


def hidden_for_budget(cfg: LSTMConfig, target: int) -> int:
    """Tamaño oculto que deja el número de parámetros lo más cerca posible del objetivo."""
    return min(range(8, 1025), key=lambda h: abs(lstm_params(LSTMConfig(**{**cfg.to_dict(), "hidden": h})) - target))


def build_lstm(target_params: int | None = None, **kwargs) -> LSTMModel:
    cfg = LSTMConfig(**kwargs)
    if target_params is not None:
        cfg.hidden = hidden_for_budget(cfg, target_params)
    return LSTMModel(cfg)
