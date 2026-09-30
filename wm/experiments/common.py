"""Utilidades compartidas por los scripts de las Etapas 4-10: datos, checkpoints y resultados."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn

from wm.data.base import FlatWindows, WindowDataset, load_base
from wm.eval.rollout import HorizonArrays, horizon_arrays
from wm.models.ae import LatentPredictor
from wm.models.baselines import WindowMLP
from wm.models.predictor import TorchPredictor
from wm.models.temporal import TEMPORAL_MODELS, from_config
from wm.train.trainer import TrainConfig
from wm.utils import RESULTS_DIR, RUNS_DIR, load_config

BASE = load_config("base")
VERSION = BASE["dataset_version"]
W, H = BASE["window"]["W"], BASE["window"]["H"]
QUEUE_IDX = 2


def results_path(name: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    return RESULTS_DIR / name


def run_dir(*parts: str) -> Path:
    p = RUNS_DIR.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float), encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------ datos
@lru_cache(maxsize=None)
def dataset(split: str, h: int = 1) -> WindowDataset:
    return load_base(VERSION, split, W=W, H=h)


@lru_cache(maxsize=None)
def flat(split: str, k: int = 1) -> FlatWindows:
    """Vista de entrenamiento; con k > 1 solo ventanas con k pasos futuros dentro del episodio."""
    return dataset(split, k).flat()


@lru_cache(maxsize=None)
def horizons(split: str, stride: int = 1) -> HorizonArrays:
    return horizon_arrays(dataset(split, H), stride=stride)


@lru_cache(maxsize=None)
def congestion_threshold() -> float:
    """Percentil 75 de la cola (m) en train: define el 'régimen congestionado'."""
    ds = dataset("train")
    q = np.concatenate([ds.norm.denorm_state(ep.states)[..., QUEUE_IDX].ravel() for ep in ds.episodes])
    return float(np.percentile(q, load_config("train")["eval"]["congestion_percentile"]))


def train_config(**overrides) -> TrainConfig:
    t = load_config("train")
    cfg = TrainConfig(batch_size=t["batch_size"], max_epochs=t["max_epochs"], patience=t["patience"],
                      reward_weight=t["loss"]["reward_weight"], samples_per_epoch=t["samples_per_epoch"],
                      val_max_samples=t["val_max_samples"], rollout_k=t["loss"]["rollout_k"])
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ------------------------------------------------------------ checkpoints
def save_checkpoint(model: nn.Module, kind: str, config: dict, path: Path, extra: dict | None = None) -> None:
    torch.save({"kind": kind, "config": config, "state_dict": model.state_dict(), "dataset": VERSION,
                **(extra or {})}, path)


def load_checkpoint(path: Path) -> nn.Module:
    ck = torch.load(path, map_location="cpu")
    if ck["kind"] in TEMPORAL_MODELS:
        model = from_config(ck["kind"], ck["config"])
    elif ck["kind"] == "mlp":
        model = WindowMLP(**ck["config"])
    else:
        raise ValueError(f"Tipo de checkpoint desconocido: {ck['kind']}")
    model.load_state_dict(ck["state_dict"])
    return model.eval()


# ------------------------------------------------------------ World Models de la Etapa 6
def representation() -> str | None:
    """Checkpoint de AE/VAE si el Experimento 0 decidió usar z; None para el estado crudo."""
    path = results_path("stage5_decision.json")
    if not path.exists():
        return None
    d = load_json(path)
    return None if d["representacion"] == "crudo" else d["mejor_ae"][d["representacion"]]


def make_predictor(ckpt: Path, ae_path: str | None, name: str):
    ds = dataset("train")
    model = load_checkpoint(ckpt)
    if ae_path:
        return LatentPredictor(torch.load(ae_path, map_location="cpu")["model"], model, name, ds.state_dim, ds.n_tls)
    return TorchPredictor(model, name, ds.state_dim)


def world_model(kind: str, seed: int):
    """Predictor del World Model `kind` (lstm | tsmixer | transformer) entrenado con la semilla final `seed`."""
    return make_predictor(run_dir(kind, "final", f"seed{seed}") / "model.pt", representation(),
                          TEMPORAL_MODELS[kind].label)


# ------------------------------------------------------------ evaluación
EVAL_SPLITS = ("val", "test", "test_ood")


def evaluate_predictor(predictor, splits=EVAL_SPLITS) -> dict:
    """Métricas de la sección 4.1 del plan en cada conjunto (ventanas con continuación de H pasos)."""
    from wm.eval.predictive import evaluate

    ds = dataset("train")
    return {s: evaluate(predictor, horizons(s), ds.norm, ds.feature_names, congestion_threshold()) for s in splits}


def summary_row(name: str, metrics: dict, split: str = "test", **extra) -> dict:
    m = metrics[split]
    rmse_h = m["rmse_h"]
    return {"modelo": name, "conjunto": split, **extra,
            "rmse_1": m["rmse_1"], "rmse_3": rmse_h[2], "rmse_5": rmse_h[4], "rmse_10": rmse_h[9],
            "rmse_20": rmse_h[19], "auc_rmse": m["auc_rmse"], "crecimiento": m["growth_ratio"],
            "mae_1": m["mae_1"], "reward_rmse_1": m["reward_rmse_1"], "phase_acc_1": m["phase_acc_1"],
            "phase_acc_20": m["phase_acc_20"], "rmse_1_congestion": m["rmse_1_congested"]}
