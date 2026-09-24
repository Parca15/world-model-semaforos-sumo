"""Etapa 5 — Experimento 0: ¿hace falta AE/VAE?

Entrena AE y VAE (91 -> z -> 91, z ∈ {16, 32}); con el mejor de cada tipo (reconstrucción en validación)
entrena el TSMixer de referencia sobre z y lo compara con el TSMixer sobre el estado crudo normalizado.
Decisión (se toma una vez, en VALIDACIÓN, sin mirar test): se usa z solo si mejora de forma significativa el
error a varios pasos (RMSE a h = 20 por episodio, Wilcoxon pareado + Holm, α = 0,05).

    python -m wm.experiments.stage5_representation
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
import torch

from wm.eval.stats import paired_comparisons
from wm.experiments.common import (dataset, evaluate_predictor, flat, load_checkpoint, results_path, run_dir,
                                   save_json, summary_row)
from wm.models.ae import LatentPredictor, StateAE, fit_ae
from wm.models.predictor import TorchPredictor
from wm.train.runner import run_jobs
from wm.utils import load_config, set_seed, threads_per_worker


def train_autoencoders(cfg: dict) -> tuple[pd.DataFrame, dict[str, str]]:
    ds = dataset("train")
    train_s = flat("train").rows[:, :ds.state_dim].copy()
    val_s = flat("val").rows[:, :ds.state_dim].copy()
    torch.set_num_threads(threads_per_worker(1))
    rows, best = [], {}
    for variational in (False, True):
        kind = "VAE" if variational else "AE"
        for z in cfg["latent_dims"]:
            path = run_dir("exp0") / f"{kind.lower()}{z}.pt"
            if path.exists() and torch.load(path, map_location="cpu", weights_only=False).get("config") == cfg:
                info = torch.load(path, map_location="cpu", weights_only=False)["info"]
                print(f"  {kind} z={z}: ya entrenado, se reutiliza")
            else:
                set_seed(0)
                model = StateAE(ds.state_dim, z, cfg["hidden"], variational)
                model, info = fit_ae(model, train_s, val_s, cfg["lr"], cfg["beta"] if variational else 0.0,
                                     cfg["max_epochs"], cfg["patience"], batch_size=cfg["batch_size"])
                torch.save({"model": model, "info": info, "config": cfg}, path)
            rows.append({"tipo": kind, "z": z, "val_recon_mse": info["val_recon_mse"], "épocas": info["epochs"],
                         "path": str(path)})
            print(f"    ({info['epochs']} épocas)", flush=True)
            print(f"  {kind} z={z}: MSE de reconstrucción (val) = {info['val_recon_mse']:.4f}")
        df_kind = pd.DataFrame([r for r in rows if r["tipo"] == kind])
        best[kind] = df_kind.loc[df_kind.val_recon_mse.idxmin(), "path"]
    return pd.DataFrame(rows), best


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    mcfg, tcfg = load_config("models"), load_config("train")
    print("Entrenando AE / VAE ...")
    ae_table, best = train_autoencoders(mcfg["autoencoder"])
    ae_table.to_csv(results_path("stage5_autoencoders.csv"), index=False)

    d = mcfg["tsmixer"]["default"]
    arch = {k: d[k] for k in ("n_blocks", "d_model", "dropout", "norm")}
    common = {"arch": arch, "lr": d["lr"], "weight_decay": d["weight_decay"], "seed": 0,
              "max_epochs": tcfg["search"]["max_epochs"]}
    jobs = [{"name": "crudo", "out_dir": str(run_dir("exp0", "tsmixer_raw")), **common},
            {"name": "AE", "out_dir": str(run_dir("exp0", "tsmixer_ae")), "ae_path": best["AE"], **common},
            {"name": "VAE", "out_dir": str(run_dir("exp0", "tsmixer_vae")), "ae_path": best["VAE"], **common}]
    print("Entrenando TSMixer sobre estado crudo, z(AE) y z(VAE) ...")
    fits = {f["name"]: f for f in run_jobs(jobs)}

    ds = dataset("train")
    predictors = {"crudo": TorchPredictor(load_checkpoint(run_dir("exp0", "tsmixer_raw") / "model.pt"), "crudo",
                                          ds.state_dim)}
    for kind, sub in (("AE", "tsmixer_ae"), ("VAE", "tsmixer_vae")):
        ae = torch.load(best[kind], map_location="cpu")["model"]
        predictors[kind] = LatentPredictor(ae, load_checkpoint(run_dir("exp0", sub) / "model.pt"), kind,
                                           ds.state_dim, ds.n_tls)
    rows, per_ep = [], {}
    for name, p in predictors.items():
        print(f"Evaluando {name} en validación ...")
        m = evaluate_predictor(p, splits=("val",))
        rows.append(summary_row(name, m, "val", params_temporal=fits[name]["params"],
                                val_loss=fits[name]["best_val_loss"]))
        per_ep[name] = np.array(m["val"]["per_episode"]["rmse_H"])
        save_json(m, results_path(f"stage5/{name}.json"))
    table = pd.DataFrame(rows)
    tests = paired_comparisons("crudo", per_ep, lower_is_better=True)
    alpha = load_config("eval")["stats"]["alpha"]
    use_latent = [t["contra"] for t in tests if t["p_holm"] < alpha and not t["referencia_mejor"]]
    decision = {"representacion": use_latent[0] if use_latent else "crudo",
                "criterio": "RMSE a h=20 por episodio de validación; Wilcoxon pareado + Holm, α=0,05",
                "pruebas": tests, "mejor_ae": best}
    table.to_csv(results_path("stage5_representation.csv"), index=False)
    save_json(decision, results_path("stage5_decision.json"))
    print(table[["modelo", "rmse_1", "rmse_5", "rmse_20", "auc_rmse"]].to_string(index=False, float_format="%.4f"))
    for t in tests:
        print(f"  crudo vs {t['contra']}: p_holm={t['p_holm']:.4g}, Cliff δ={t['cliffs_delta']:+.3f} ({t['efecto']})")
    print(f"\nDecisión: usar representación '{decision['representacion']}'. "
          f"Etapa 5 completada en {(time.time() - t0) / 60:.1f} min.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
