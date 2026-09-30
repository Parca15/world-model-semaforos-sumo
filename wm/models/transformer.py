"""Transformer (encoder) como modelo temporal del World Model.

Entrada [B, W, input_dim] (estados normalizados aplanados + acciones) -> Δŝ_{t+1} [B, state_dim], r̂_t [B, n_tls].

    embedding lineal (input_dim -> d_model) + codificación posicional sinusoidal
    N × capa encoder (pre-norm): atención multi-cabeza causal + MLP (d_model -> ff_dim -> d_model), residuales
    norma final -> último token -> cabeza densa (d_model -> state_dim + n_tls)

La máscara causal hace que el token t solo vea t' <= t, como la LSTM. ff_dim no se busca: se calcula para
cumplir el presupuesto de parámetros.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

from wm.models.tsmixer import count_params


@dataclass
class TransformerConfig:
    input_dim: int = 98
    state_dim: int = 91
    n_tls: int = 7
    window: int = 12
    n_layers: int = 2
    d_model: int = 64
    n_heads: int = 4
    ff_dim: int = 128
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


def sinusoidal_encoding(length: int, d: int) -> torch.Tensor:
    pos = torch.arange(length, dtype=torch.float32)[:, None]
    freq = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(length, d)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq[:d // 2])
    return pe


class TransformerModel(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Linear(cfg.input_dim, cfg.d_model)
        self.register_buffer("pos", sinusoidal_encoding(cfg.window, cfg.d_model), persistent=False)
        self.register_buffer("mask", nn.Transformer.generate_square_subsequent_mask(cfg.window), persistent=False)
        self.drop = nn.Dropout(cfg.dropout)
        layer = nn.TransformerEncoderLayer(cfg.d_model, cfg.n_heads, cfg.ff_dim, cfg.dropout, batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, norm=nn.LayerNorm(cfg.d_model),
                                             enable_nested_tensor=False)
        self.head = nn.Linear(cfg.d_model, cfg.state_dim + cfg.n_tls)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.drop(self.embed(x) + self.pos)
        h = self.encoder(h, mask=self.mask, is_causal=True)
        out = self.head(h[:, -1])
        return out[:, :self.cfg.state_dim], out[:, self.cfg.state_dim:]


def ff_dim_for_budget(cfg: TransformerConfig, target: int) -> int:
    """ff_dim que deja el número de parámetros lo más cerca posible del objetivo."""
    base = count_params(TransformerModel(TransformerConfig(**{**cfg.to_dict(), "ff_dim": 1})))
    per_unit = 2 * cfg.d_model + 1   # cada unidad de ff añade d (entrada) + d (salida) + 1 (sesgo) por capa
    return max(8, round((target - base) / (cfg.n_layers * per_unit)) + 1)


def build_transformer(target_params: int | None = None, **kwargs) -> TransformerModel:
    cfg = TransformerConfig(**kwargs)
    if target_params is not None:
        cfg.ff_dim = ff_dim_for_budget(cfg, target_params)
    return TransformerModel(cfg)
