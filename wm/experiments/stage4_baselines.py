"""Etapa 4 — baselines predictivos (persistencia, media móvil, ridge y MLP) sobre val / test / test-OOD.

    python -m wm.experiments.stage4_baselines

Salidas: results/stage4/<modelo>.json (métricas completas), results/stage4_baselines.csv (resumen),
runs/baselines/mlp.pt. Criterio de salida: tabla de métricas de los baselines en val y test.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from wm.experiments.common import (dataset, evaluate_predictor, flat, results_path, run_dir, save_checkpoint,
                                   save_json, summary_row, train_config)
from wm.models.baselines import MovingAverage, Persistence, Ridge, WindowMLP, ridge_gram
from wm.models.predictor import TorchPredictor
from wm.models.tsmixer import count_params
from wm.train.trainer import evaluate_loss, fit
from wm.utils import load_config, set_seed


def fit_ridge(alphas: list[float]) -> tuple[Ridge, list[dict]]:
    train, val = flat("train"), flat("val")
    gram, cross = ridge_gram(train)
    cfg = train_config()
    trials = []
    for a in alphas:
        model = Ridge(dataset("train").state_dim, a).fit_gram(gram, cross)
        x, d, r = val.batch(np.arange(len(val)))
        pd_, pr = model.predict(x.numpy())
        loss = float(np.mean((pd_ - d.numpy()) ** 2) + cfg.reward_weight * np.mean((pr - r.numpy()) ** 2))
        trials.append({"alpha": a, "val_loss": loss})
        print(f"  ridge alpha={a}: val_loss={loss:.4f}")
    best = min(trials, key=lambda t: t["val_loss"])["alpha"]
    return Ridge(dataset("train").state_dim, best).fit_gram(gram, cross), trials


def fit_mlp(cfg_m: dict) -> tuple[TorchPredictor, dict]:
    ds = dataset("train")
    set_seed(0)
    conf = {"window": ds.W, "input_dim": ds.input_dim, "state_dim": ds.state_dim, "n_tls": ds.n_tls,
            "hidden": cfg_m["hidden"], "dropout": cfg_m["dropout"]}
    model = WindowMLP(**conf)
    tcfg = train_config(lr=cfg_m["lr"], weight_decay=cfg_m["weight_decay"], seed=0, threads=4)
    out = run_dir("baselines")
    (out / "mlp_log.jsonl").unlink(missing_ok=True)
    model, res = fit(model, flat("train"), flat("val"), tcfg, out / "mlp_log.jsonl")
    save_checkpoint(model, "mlp", conf, out / "mlp.pt", {"fit": res.to_dict()})
    print(f"  MLP: {count_params(model)} parámetros, mejor época {res.best_epoch}, {res.train_seconds / 60:.1f} min")
    return TorchPredictor(model, "MLP", ds.state_dim), {"params": count_params(model), **res.to_dict()}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config("models")["baselines"]
    ds = dataset("train")
    t0 = time.time()
    predictors, extra = [Persistence(ds.state_dim, ds.n_tls), MovingAverage(ds.state_dim, ds.n_tls)], {}
    print("Ajustando ridge ...")
    ridge, trials = fit_ridge(cfg["ridge"]["alphas"])
    predictors.append(ridge)
    extra["Ridge"] = {"alpha": ridge.alpha, "trials": trials, "params": int(ridge.coef.size)}
    print("Entrenando MLP ...")
    mlp, info = fit_mlp(cfg["mlp"])
    predictors.append(mlp)
    extra["MLP"] = info

    rows = []
    for p in predictors:
        print(f"Evaluando {p.name} ...")
        metrics = evaluate_predictor(p)
        save_json({"model": p.name, "extra": extra.get(p.name, {}), "metrics": metrics},
                  results_path(f"stage4/{p.name.replace(' ', '_').lower()}.json"))
        rows += [summary_row(p.name, metrics, s) for s in ("val", "test", "test_ood")]
    df = pd.DataFrame(rows)
    pers = df[df.modelo == "Persistencia"].set_index("conjunto")
    df["skill_1"] = [1 - r.rmse_1 / pers.loc[r.conjunto, "rmse_1"] for r in df.itertuples()]
    df["skill_20"] = [1 - r.rmse_20 / pers.loc[r.conjunto, "rmse_20"] for r in df.itertuples()]
    df.to_csv(results_path("stage4_baselines.csv"), index=False)
    print(df[df.conjunto == "test"][["modelo", "rmse_1", "rmse_5", "rmse_20", "skill_1", "skill_20",
                                     "reward_rmse_1", "phase_acc_1"]].to_string(index=False, float_format="%.4f"))
    print(f"\nEtapa 4 completada en {(time.time() - t0) / 60:.1f} min. Criterio de salida: tabla en "
          f"{results_path('stage4_baselines.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
