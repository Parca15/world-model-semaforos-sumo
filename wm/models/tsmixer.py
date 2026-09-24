"""TSMixer (Chen et al., 2023) como modelo temporal del World Model.

Entrada [B, W, input_dim] (estados normalizados aplanados + acciones) -> Δŝ_{t+1} [B, state_dim], r̂_t [B, n_tls].

    proyección de entrada (input_dim -> d_model)
    N × MixerBlock:
        mezcla temporal:    norm -> Linear(W, W) a lo largo del tiempo -> ReLU -> dropout -> residual
        mezcla de variables: norm -> Linear(d, ff) -> ReLU -> dropout -> Linear(ff, d) -> dropout -> residual
    proyección temporal (W -> 1) y cabeza densa (d_model -> state_dim + n_tls)

Sin recurrencia ni atención: solo capas densas, normalización y conexiones residuales.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class TSMixerConfig:
    input_dim: int = 98
    state_dim: int = 91
    n_tls: int = 7
    window: int = 12
    n_blocks: int = 4
    d_model: int = 96
    ff_dim: int = 128
    dropout: float = 0.1
    norm: str = "layer"   # layer | batch

    def to_dict(self) -> dict:
        return asdict(self)


class _Norm(nn.Module):
    """Normalización sobre la dimensión de variables de un tensor [B, W, d]."""

    def __init__(self, kind: str, d: int):
        super().__init__()
        self.kind = kind
        self.norm = nn.LayerNorm(d) if kind == "layer" else nn.BatchNorm1d(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "layer":
            return self.norm(x)
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class MixerBlock(nn.Module):
    def __init__(self, window: int, d_model: int, ff_dim: int, dropout: float, norm: str):
        super().__init__()
        self.time_norm = _Norm(norm, d_model)
        self.time_mlp = nn.Linear(window, window)
        self.feat_norm = _Norm(norm, d_model)
        self.feat_mlp = nn.Sequential(nn.Linear(d_model, ff_dim), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(ff_dim, d_model), nn.Dropout(dropout))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x [B, W, d]
        h = self.time_norm(x).transpose(1, 2)             # [B, d, W]
        x = x + self.drop(torch.relu(self.time_mlp(h))).transpose(1, 2)
        return x + self.feat_mlp(self.feat_norm(x))


class TSMixer(nn.Module):
    def __init__(self, cfg: TSMixerConfig):
        super().__init__()
        self.cfg = cfg
        self.inp = nn.Linear(cfg.input_dim, cfg.d_model)
        self.blocks = nn.ModuleList(MixerBlock(cfg.window, cfg.d_model, cfg.ff_dim, cfg.dropout, cfg.norm)
                                    for _ in range(cfg.n_blocks))
        self.time_proj = nn.Linear(cfg.window, 1)
        self.head = nn.Linear(cfg.d_model, cfg.state_dim + cfg.n_tls)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.inp(x)
        for block in self.blocks:
            h = block(h)
        h = self.time_proj(h.transpose(1, 2)).squeeze(-1)  # [B, d]
        out = self.head(h)
        return out[:, :self.cfg.state_dim], out[:, self.cfg.state_dim:]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ff_dim_for_budget(cfg: TSMixerConfig, target: int) -> int:
    """ff_dim que deja el número de parámetros lo más cerca posible del objetivo."""
    base = count_params(TSMixer(TSMixerConfig(**{**cfg.to_dict(), "ff_dim": 1})))
    per_unit = 2 * cfg.d_model + 1   # cada unidad de ff añade d (entrada) + d (salida) + 1 (sesgo) por bloque
    return max(8, round((target - base) / (cfg.n_blocks * per_unit)) + 1)


def build_tsmixer(target_params: int | None = None, **kwargs) -> TSMixer:
    cfg = TSMixerConfig(**kwargs)
    if target_params is not None:
        cfg.ff_dim = ff_dim_for_budget(cfg, target_params)
    return TSMixer(cfg)
