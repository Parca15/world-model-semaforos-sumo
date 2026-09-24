"""Experimento 0 (Etapa 5) — Autoencoder / VAE sobre el estado normalizado (91 -> z -> 91).

`LatentPredictor` envuelve encoder + modelo temporal en z + decoder y expone la misma interfaz Predictor que
el modelo sobre estado crudo, así ambas rutas se evalúan con el mismo código y en el mismo espacio.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn


def _mlp(sizes: list[int]) -> nn.Sequential:
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers[:-1])


class StateAE(nn.Module):
    def __init__(self, state_dim: int, latent_dim: int, hidden: list[int], variational: bool = False):
        super().__init__()
        self.variational = variational
        self.encoder = _mlp([state_dim, *hidden, latent_dim * (2 if variational else 1)])
        self.decoder = _mlp([latent_dim, *hidden[::-1], state_dim])
        self.latent_dim = latent_dim

    def encode(self, s: torch.Tensor) -> torch.Tensor:
        """Código determinista (la media en el VAE)."""
        h = self.encoder(s)
        return h[:, :self.latent_dim] if self.variational else h

    def forward(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(s)
        if not self.variational:
            return self.decoder(h), torch.zeros((), device=s.device)
        mu, logvar = h[:, :self.latent_dim], h[:, self.latent_dim:].clamp(-8, 8)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1))
        return self.decoder(z), kl


def fit_ae(model: StateAE, train_states: np.ndarray, val_states: np.ndarray, lr: float, beta: float,
           max_epochs: int, patience: int, batch_size: int = 256, seed: int = 0) -> tuple[StateAE, dict]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    xt, xv = torch.from_numpy(train_states), torch.from_numpy(val_states)
    best, best_state, bad, hist = np.inf, None, 0, []
    for epoch in range(1, max_epochs + 1):
        model.train()
        xt = xt[torch.from_numpy(rng.permutation(len(xt)))]
        for i in range(0, len(xt), batch_size):
            b = xt[i:i + batch_size]
            rec, kl = model(b)
            loss = nn.functional.mse_loss(rec, b) + beta * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            rec, _ = model(xv)
            val = nn.functional.mse_loss(rec, xv).item()
        hist.append({"epoch": epoch, "val_recon_mse": val})
        if val < best - 1e-6:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model.eval(), {"val_recon_mse": best, "epochs": len(hist), "history": hist}


class LatentPredictor:
    """encoder -> modelo temporal en z -> decoder, con la interfaz Predictor en el espacio de estados."""

    def __init__(self, ae: StateAE, dynamics: nn.Module, name: str, state_dim: int, n_tls: int):
        self.ae, self.dyn = ae.eval(), dynamics.eval()
        self.name, self.state_dim, self.n_tls = name, state_dim, n_tls

    @torch.no_grad()
    def encode_window(self, x: torch.Tensor) -> torch.Tensor:
        B, W, _ = x.shape
        z = self.ae.encode(x[..., :self.state_dim].reshape(B * W, -1)).reshape(B, W, -1)
        return torch.cat([z, x[..., self.state_dim:]], dim=-1)

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        outs_d, outs_r = [], []
        for i in range(0, len(x), 4096):
            xb = torch.from_numpy(np.ascontiguousarray(x[i:i + 4096], dtype=np.float32))
            zx = self.encode_window(xb)
            dz, r = self.dyn(zx)
            s_next = self.ae.decoder(zx[:, -1, :self.ae.latent_dim] + dz)
            outs_d.append((s_next - xb[:, -1, :self.state_dim]).numpy())
            outs_r.append(r.numpy())
        return np.concatenate(outs_d), np.concatenate(outs_r)
