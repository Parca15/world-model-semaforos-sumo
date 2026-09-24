"""Bucle de entrenamiento común (Etapa 6): el mismo para cualquier modelo temporal o baseline neuronal.

Pérdida MSE(Δs) + λ·MSE(r) en espacio normalizado, AdamW, batch 256, early stopping sobre la pérdida de
validación. Con `rollout_k` > 1 (pérdida multi-paso, opción del plan) el modelo avanza k pasos reutilizando sus
propias predicciones como entrada (con las acciones reales) y la pérdida promedia el error de estado y de
recompensa en los k pasos; con k = 1 coincide con la pérdida a 1 paso. Una "época" es una muestra aleatoria de `samples_per_epoch` ventanas de train (las ventanas
consecutivas se solapan en W−1 pasos, así que recorrerlas todas por época es redundante en CPU).
"""
from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import psutil
import torch
from torch import nn

from wm.data.base import FlatWindows


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 10
    reward_weight: float = 1.0
    samples_per_epoch: int | None = None   # None = todas las ventanas de train
    val_max_samples: int | None = None     # submuestra fija de validación (None = todas)
    seed: int = 0
    threads: int = 1
    rollout_k: int = 1                     # pasos de la pérdida multi-paso (1 = pérdida a 1 paso)


@dataclass
class FitResult:
    best_epoch: int
    best_val_loss: float
    epochs_run: int
    train_seconds: float
    seconds_per_epoch: float
    peak_rss_mb: float
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _loss(model, flat: FlatWindows, idx, cfg: TrainConfig, mse=nn.functional.mse_loss):
    if cfg.rollout_k == 1:
        x, d, r = flat.batch(idx)
        pd_, pr = model(x)
        ld, lr_ = mse(pd_, d), mse(pr, r)
        return ld + cfg.reward_weight * lr_, ld, lr_
    x, d, r, fut_rows = flat.batch(idx, cfg.rollout_k)
    sd = d.shape[-1]
    s_true = x[:, -1:, :sd] + torch.cumsum(d, dim=1)         # s_{t+1..t+k} reales
    ld = lr_ = 0.0
    for j in range(cfg.rollout_k):
        pd_, pr = model(x)
        s_pred = x[:, -1, :sd] + pd_
        ld = ld + mse(s_pred, s_true[:, j])
        lr_ = lr_ + mse(pr, r[:, j])
        if j < cfg.rollout_k - 1:
            row = torch.cat([s_pred, fut_rows[:, j + 1, sd:]], dim=1)[:, None]
            x = torch.cat([x[:, 1:], row], dim=1)
    ld, lr_ = ld / cfg.rollout_k, lr_ / cfg.rollout_k
    return ld + cfg.reward_weight * lr_, ld, lr_


@torch.no_grad()
def evaluate_loss(model: nn.Module, flat: FlatWindows, idx: np.ndarray, cfg: TrainConfig) -> tuple[float, float, float]:
    model.eval()
    tot = np.zeros(3)
    for i in range(0, len(idx), 2048):
        b = idx[i:i + 2048]
        tot += np.array([v.item() for v in _loss(model, flat, b, cfg)]) * len(b)
    return tuple(tot / len(idx))


def fit(model: nn.Module, train: FlatWindows, val: FlatWindows, cfg: TrainConfig,
        log_path: Path | None = None) -> tuple[nn.Module, FitResult]:
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    proc = psutil.Process(os.getpid())
    peak = proc.memory_info().rss

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n_epoch = min(cfg.samples_per_epoch or len(train), len(train))
    val_idx = np.arange(len(val))
    if cfg.val_max_samples and len(val_idx) > cfg.val_max_samples:
        val_idx = np.sort(np.random.default_rng(12345).choice(val_idx, cfg.val_max_samples, replace=False))

    best, best_state, best_epoch, bad = np.inf, None, 0, 0
    history = []
    t0 = time.time()
    epoch = 0
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        order = rng.permutation(len(train))[:n_epoch]
        tr = np.zeros(3)
        for i in range(0, n_epoch, cfg.batch_size):
            b = order[i:i + cfg.batch_size]
            loss, ld, lr_ = _loss(model, train, b, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tr += np.array([loss.item(), ld.item(), lr_.item()]) * len(b)
        tr /= n_epoch
        va = evaluate_loss(model, val, val_idx, cfg)
        peak = max(peak, proc.memory_info().rss)
        history.append({"epoch": epoch, "train_loss": tr[0], "train_delta": tr[1], "train_reward": tr[2],
                        "val_loss": va[0], "val_delta": va[1], "val_reward": va[2],
                        "seconds": time.time() - t0})
        if log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(history[-1]) + "\n")
        if va[0] < best - 1e-6:
            best, best_epoch, bad = va[0], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    elapsed = time.time() - t0
    model.load_state_dict(best_state)
    model.eval()
    return model, FitResult(best_epoch, float(best), epoch, elapsed, elapsed / epoch, peak / 2**20, history)
