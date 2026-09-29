"""Etapa 6 — TSMixer: búsqueda aleatoria (12 configuraciones), 5 semillas finales, evaluación y costo.

    python -m wm.experiments.stage6_tsmixer [--skip-search]

Criterio de salida: el TSMixer supera a la persistencia y a ridge en test (1 paso y multi-paso), con las
métricas completas de las secciones 4.1 y 4.2 del plan.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from wm.eval.cost import hardware, inference_cost
from wm.eval.stats import bootstrap_ci, paired_comparisons
from wm.experiments.common import (dataset, evaluate_predictor, load_checkpoint, load_json, results_path, run_dir,
                                   save_json, summary_row)
from wm.models.ae import LatentPredictor
from wm.models.predictor import TorchPredictor
from wm.models.tsmixer import count_params
from wm.train.runner import run_jobs
from wm.utils import load_config

ARCH_KEYS = ("n_blocks", "d_model", "dropout", "norm")


def sample_configs(space: dict, n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    seen, configs = set(), []
    while len(configs) < n:
        c = {k: v[rng.integers(len(v))] for k, v in space.items()}
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            configs.append({k: (v.item() if hasattr(v, "item") else v) for k, v in c.items()})
    return configs


def representation() -> str | None:
    """Checkpoint de AE/VAE si el Experimento 0 decidió usar z; None para el estado crudo."""
    path = results_path("stage5_decision.json")
    if not path.exists():
        return None
    d = load_json(path)
    return None if d["representacion"] == "crudo" else d["mejor_ae"][d["representacion"]]


def job(name: str, cfg: dict, seed: int, max_epochs: int, out: Path, ae_path: str | None) -> dict:
    return {"name": name, "arch": {k: cfg[k] for k in ARCH_KEYS}, "lr": cfg["lr"],
            "weight_decay": cfg["weight_decay"], "seed": seed, "max_epochs": max_epochs, "out_dir": str(out),
            **({"ae_path": ae_path} if ae_path else {})}


def make_predictor(ckpt: Path, ae_path: str | None, name: str):
    ds = dataset("train")
    model = load_checkpoint(ckpt)
    if ae_path:
        return LatentPredictor(torch.load(ae_path, map_location="cpu")["model"], model, name, ds.state_dim, ds.n_tls)
    return TorchPredictor(model, name, ds.state_dim)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-search", action="store_true", help="reutilizar runs/tsmixer/search")
    args = ap.parse_args()
    t0 = time.time()
    mcfg, tcfg = load_config("models"), load_config("train")
    ae_path = representation()
    print(f"Representación: {'z (' + ae_path + ')' if ae_path else 'estado crudo normalizado'}")

    # ---------------- búsqueda aleatoria
    configs = sample_configs(mcfg["tsmixer"]["search_space"], tcfg["search"]["n_configs"], tcfg["search"]["seed"])
    if not args.skip_search:
        print(f"Búsqueda: {len(configs)} configuraciones (máx. {tcfg['search']['max_epochs']} épocas) ...")
        jobs = [job(f"cfg{i:02d}", c, 0, tcfg["search"]["max_epochs"], run_dir("tsmixer", "search", f"cfg{i:02d}"),
                    ae_path) for i, c in enumerate(configs)]
        run_jobs(jobs)
    search = pd.DataFrame([{**load_json(run_dir("tsmixer", "search", f"cfg{i:02d}") / "fit.json"), **c}
                           for i, c in enumerate(configs)])
    search = search.drop(columns=["arch"]).sort_values("best_val_loss")
    search.to_csv(results_path("stage6_search.csv"), index=False)
    best = configs[int(search.iloc[0]["name"][3:])]
    print(f"Mejor configuración: {best} (val_loss={search.iloc[0]['best_val_loss']:.4f})")

    # ---------------- 5 semillas finales
    seeds = tcfg["final_seeds"]
    print(f"Entrenamiento final con semillas {seeds} (máx. {tcfg['max_epochs']} épocas) ...")
    run_jobs([job(f"seed{s}", best, s, tcfg["max_epochs"], run_dir("tsmixer", "final", f"seed{s}"), ae_path)
              for s in seeds])

    # ---------------- evaluación
    rows, per_seed = [], {}
    for s in seeds:
        d = run_dir("tsmixer", "final", f"seed{s}")
        fitj = load_json(d / "fit.json")
        print(f"Evaluando semilla {s} ...")
        m = evaluate_predictor(make_predictor(d / "model.pt", ae_path, "TSMixer"))
        save_json({"fit": fitj, "metrics": m}, results_path(f"stage6/seed{s}.json"))
        per_seed[s] = m
        rows += [summary_row("TSMixer", m, sp, semilla=s, params=fitj["params"], mejor_epoca=fitj["best_epoch"],
                             epocas=fitj["epochs_run"], min_entrenamiento=fitj["train_seconds"] / 60,
                             s_por_epoca=fitj["seconds_per_epoch"], rss_pico_mb=fitj["peak_rss_mb"])
                 for sp in ("val", "test", "test_ood")]
    final = pd.DataFrame(rows)
    final.to_csv(results_path("stage6_final_seeds.csv"), index=False)

    # ---------------- costo computacional
    model0 = load_checkpoint(run_dir("tsmixer", "final", f"seed{seeds[0]}") / "model.pt")
    ds = dataset("train")
    cost = {"hardware": hardware(), "params": count_params(model0), "arch": model0.cfg.to_dict(),
            "train_s_per_epoch_mean": float(final[final.conjunto == "test"].s_por_epoca.mean()),
            "train_min_total_mean": float(final[final.conjunto == "test"].min_entrenamiento.mean()),
            "train_peak_rss_mb_max": float(final.rss_pico_mb.max()),
            **inference_cost(model0, ds.W, model0.cfg.input_dim, model0.cfg.state_dim, threads=1)}
    save_json(cost, results_path("stage6_cost.json"))

    # ---------------- comparación con baselines en test (por episodio, RMSE promedio de las 5 semillas)
    base = {n: load_json(results_path(f"stage4/{f}.json"))["metrics"]
            for n, f in (("Persistencia", "persistencia"), ("Media móvil", "media_móvil"), ("Ridge", "ridge"),
                         ("MLP", "mlp"))}
    comparisons = {}
    for key in ("rmse_1", "rmse_H"):
        samples = {"TSMixer": np.mean([per_seed[s]["test"]["per_episode"][key] for s in seeds], axis=0)}
        samples.update({n: np.array(m["test"]["per_episode"][key]) for n, m in base.items()})
        comparisons[key] = paired_comparisons("TSMixer", samples, lower_is_better=True)
    agg = final.groupby("conjunto")[["rmse_1", "rmse_5", "rmse_20", "auc_rmse", "reward_rmse_1",
                                     "phase_acc_1", "rmse_1_congestion"]].agg(["mean", "std"])
    test = final[final.conjunto == "test"]
    ci = {k: bootstrap_ci(test[k]) for k in ("rmse_1", "rmse_20")}
    b4 = pd.read_csv(results_path("stage4_baselines.csv")).set_index(["modelo", "conjunto"])
    beats = {b: bool(test.rmse_1.mean() < b4.loc[(b, "test"), "rmse_1"] and
                     test.rmse_20.mean() < b4.loc[(b, "test"), "rmse_20"]) for b in ("Persistencia", "Ridge")}
    save_json({"best_config": best, "representation": ae_path or "crudo", "aggregate": {f"{m}_{s}": v for (m, s), v in agg.to_dict().items()},
               "ci_test": ci, "comparisons_test": comparisons, "beats": beats}, results_path("stage6_summary.json"))

    print(agg.to_string(float_format="%.4f"))
    print("Costo:", {k: round(v, 3) if isinstance(v, float) else v for k, v in cost.items() if k != "arch"})
    for key, comp in comparisons.items():
        for c in comp:
            print(f"  {key}: TSMixer vs {c['contra']}: p_holm={c['p_holm']:.3g} δ={c['cliffs_delta']:+.3f} "
                  f"({c['efecto']}) TSMixer mejor={c['referencia_mejor']}")
    ok = all(beats.values())
    print(f"\nSupera a persistencia y ridge en test (1 y 20 pasos): {beats} -> Criterio de salida Etapa 6: "
          f"{'CUMPLIDO' if ok else 'NO CUMPLIDO'} ({(time.time() - t0) / 60:.1f} min)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
