"""Etapa 6 — modelos temporales (LSTM, TSMixer, Transformer) con el mismo protocolo de entrenamiento.

    python -m wm.experiments.stage6_temporal [--model tsmixer lstm transformer]

Para cada modelo: búsqueda aleatoria (12 configuraciones, mismo presupuesto), 5 semillas finales con la mejor,
evaluación (secciones 4.1-4.2 del plan) y costo. Al final, comparación entre los tres modelos.

Salidas por modelo en results/stage6/<modelo>/ (search.csv, final_seeds.csv, seed{k}.json, summary.json,
cost.json) y la comparación en results/stage6_comparison.csv y results/stage6_model_tests.json.

Criterio de salida: los tres modelos superan a la persistencia y a ridge en test (1 y 20 pasos).
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from wm.eval.cost import hardware, inference_cost
from wm.eval.stats import bootstrap_ci, paired_comparisons
from wm.experiments.common import (dataset, evaluate_predictor, load_checkpoint, load_json, make_predictor,
                                   representation, results_path, run_dir, save_json, summary_row)
from wm.models.temporal import TEMPORAL_MODELS, build_temporal
from wm.models.tsmixer import count_params
from wm.train.runner import run_jobs
from wm.utils import load_config

ORDER = ("tsmixer", "lstm", "transformer")   # el TSMixer primero: su búsqueda ya estaba hecha
OPTIM_KEYS = ("lr", "weight_decay")
BASELINES = (("Persistencia", "persistencia"), ("Media móvil", "media_móvil"), ("Ridge", "ridge"), ("MLP", "mlp"))
AGG_COLS = ["rmse_1", "rmse_5", "rmse_20", "auc_rmse", "reward_rmse_1", "phase_acc_1", "rmse_1_congestion"]


def within_budget(kind: str, arch: dict) -> bool:
    b = load_config("models")["param_budget"]
    ds = dataset("train")
    model = build_temporal(kind, b["target"], input_dim=ds.state_dim + ds.n_tls, state_dim=ds.state_dim,
                           n_tls=ds.n_tls, window=ds.W, **arch)
    return abs(count_params(model) - b["target"]) <= b["tolerance"] * b["target"]


def sample_configs(kind: str, space: dict, n: int, seed: int) -> list[dict]:
    """n configuraciones distintas del espacio, en orden de muestreo, que cumplen el presupuesto de parámetros."""
    rng = np.random.default_rng(seed)
    seen, configs = set(), []
    n_total = int(np.prod([len(v) for v in space.values()]))
    while len(configs) < n and len(seen) < n_total:
        c = {k: v[rng.integers(len(v))] for k, v in space.items()}
        c = {k: (v.item() if hasattr(v, "item") else v) for k, v in c.items()}
        key = tuple(sorted(c.items()))
        if key in seen:
            continue
        seen.add(key)
        if within_budget(kind, arch_of(c)):
            configs.append(c)
    return configs


def arch_of(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if k not in OPTIM_KEYS}


def job(kind: str, name: str, cfg: dict, seed: int, max_epochs: int, stage: str, ae_path: str | None) -> dict:
    # el TSMixer no lleva la clave "model" (valor por defecto) para conservar la identidad de sus trabajos
    return {**({"model": kind} if kind != "tsmixer" else {}), "name": name, "arch": arch_of(cfg),
            "lr": cfg["lr"], "weight_decay": cfg["weight_decay"], "seed": seed, "max_epochs": max_epochs,
            "out_dir": str(run_dir(kind, stage, name)), **({"ae_path": ae_path} if ae_path else {})}


def out_path(kind: str, name: str):
    path = results_path(f"stage6/{kind}/{name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def run_model(kind: str, ae_path: str | None) -> dict:
    label = TEMPORAL_MODELS[kind].label
    mcfg, tcfg = load_config("models"), load_config("train")
    t0 = time.time()
    print(f"\n===== {label} =====", flush=True)

    # ---------------- búsqueda aleatoria
    configs = sample_configs(kind, mcfg[kind]["search_space"], tcfg["search"]["n_configs"], tcfg["search"]["seed"])
    print(f"Búsqueda: {len(configs)} configuraciones (máx. {tcfg['search']['max_epochs']} épocas) ...", flush=True)
    run_jobs([job(kind, f"cfg{i:02d}", c, 0, tcfg["search"]["max_epochs"], "search", ae_path)
              for i, c in enumerate(configs)])
    search = pd.DataFrame([{**load_json(run_dir(kind, "search", f"cfg{i:02d}") / "fit.json"), **c}
                           for i, c in enumerate(configs)])
    search = search.drop(columns=["arch", "model_config", "job_key"], errors="ignore").sort_values("best_val_loss")
    search.to_csv(out_path(kind, "search.csv"), index=False)
    best = configs[int(search.iloc[0]["name"][3:])]
    print(f"Mejor configuración: {best} (val_loss={search.iloc[0]['best_val_loss']:.4f})", flush=True)

    # ---------------- 5 semillas finales
    seeds = tcfg["final_seeds"]
    print(f"Entrenamiento final con semillas {seeds} (máx. {tcfg['max_epochs']} épocas) ...", flush=True)
    run_jobs([job(kind, f"seed{s}", best, s, tcfg["max_epochs"], "final", ae_path) for s in seeds])

    # ---------------- evaluación
    rows, per_seed = [], {}
    for s in seeds:
        d = run_dir(kind, "final", f"seed{s}")
        fitj = load_json(d / "fit.json")
        print(f"Evaluando semilla {s} ...", flush=True)
        m = evaluate_predictor(make_predictor(d / "model.pt", ae_path, label))
        save_json({"fit": fitj, "metrics": m}, out_path(kind, f"seed{s}.json"))
        per_seed[s] = m
        rows += [summary_row(label, m, sp, semilla=s, params=fitj["params"], mejor_epoca=fitj["best_epoch"],
                             epocas=fitj["epochs_run"], min_entrenamiento=fitj["train_seconds"] / 60,
                             s_por_epoca=fitj["seconds_per_epoch"], rss_pico_mb=fitj["peak_rss_mb"])
                 for sp in ("val", "test", "test_ood")]
    final = pd.DataFrame(rows)
    final.to_csv(out_path(kind, "final_seeds.csv"), index=False)

    # ---------------- costo computacional
    model0 = load_checkpoint(run_dir(kind, "final", f"seed{seeds[0]}") / "model.pt")
    ds = dataset("train")
    test = final[final.conjunto == "test"]
    cost = {"hardware": hardware(), "params": count_params(model0), "arch": model0.cfg.to_dict(),
            "train_s_per_epoch_mean": float(test.s_por_epoca.mean()),
            "train_min_total_mean": float(test.min_entrenamiento.mean()),
            "train_peak_rss_mb_max": float(final.rss_pico_mb.max()),
            **inference_cost(model0, ds.W, model0.cfg.input_dim, model0.cfg.state_dim, threads=1)}
    save_json(cost, out_path(kind, "cost.json"))

    # ---------------- comparación con baselines en test (por episodio, RMSE promedio de las 5 semillas)
    base = {n: load_json(results_path(f"stage4/{f}.json"))["metrics"] for n, f in BASELINES}
    comparisons = {}
    for key in ("rmse_1", "rmse_H"):
        samples = {label: per_episode(per_seed, key)}
        samples.update({n: np.array(m["test"]["per_episode"][key]) for n, m in base.items()})
        comparisons[key] = paired_comparisons(label, samples, lower_is_better=True)
    agg = final.groupby("conjunto")[AGG_COLS].agg(["mean", "std"])
    ci = {k: bootstrap_ci(test[k]) for k in ("rmse_1", "rmse_20")}
    b4 = pd.read_csv(results_path("stage4_baselines.csv")).set_index(["modelo", "conjunto"])
    beats = {b: bool(test.rmse_1.mean() < b4.loc[(b, "test"), "rmse_1"] and
                     test.rmse_20.mean() < b4.loc[(b, "test"), "rmse_20"]) for b in ("Persistencia", "Ridge")}
    summary = {"model": kind, "label": label, "best_config": best, "representation": ae_path or "crudo",
               "aggregate": {f"{m}_{s}": v for (m, s), v in agg.to_dict().items()}, "ci_test": ci,
               "comparisons_test": comparisons, "beats": beats, "minutes": (time.time() - t0) / 60}
    save_json(summary, out_path(kind, "summary.json"))

    print(agg.to_string(float_format="%.4f"))
    print("Costo:", {k: round(v, 3) if isinstance(v, float) else v for k, v in cost.items() if k != "arch"})
    for key, comp in comparisons.items():
        for c in comp:
            print(f"  {key}: {label} vs {c['contra']}: p_holm={c['p_holm']:.3g} δ={c['cliffs_delta']:+.3f} "
                  f"({c['efecto']}) {label} mejor={c['referencia_mejor']}")
    print(f"{label} supera a persistencia y ridge en test (1 y 20 pasos): {beats}", flush=True)
    return summary


def per_episode(per_seed: dict, key: str) -> np.ndarray:
    """RMSE por episodio de test, promediado sobre las semillas finales."""
    return np.mean([m["test"]["per_episode"][key] for m in per_seed.values()], axis=0)


def compare_models(kinds: list[str]) -> dict:
    """Tabla conjunta y pruebas pareadas (Wilcoxon por episodio + Holm) entre los modelos temporales."""
    seeds = load_config("train")["final_seeds"]
    final = pd.concat([pd.read_csv(out_path(k, "final_seeds.csv")) for k in kinds])
    table = final.groupby(["modelo", "conjunto"])[AGG_COLS + ["params", "min_entrenamiento"]].agg(["mean", "std"])
    table.columns = [f"{m}_{s}" for m, s in table.columns]
    table.reset_index().to_csv(results_path("stage6_comparison.csv"), index=False)
    per_seed = {k: {s: load_json(out_path(k, f"seed{s}.json"))["metrics"] for s in seeds} for k in kinds}
    tests = {}
    for key in ("rmse_1", "rmse_H"):
        samples = {TEMPORAL_MODELS[k].label: per_episode(per_seed[k], key) for k in kinds}
        tests[key] = {TEMPORAL_MODELS[k].label: paired_comparisons(TEMPORAL_MODELS[k].label, samples)
                      for k in kinds}
    save_json(tests, results_path("stage6_model_tests.json"))
    print("\n===== Comparación entre modelos temporales (test) =====")
    print(table.xs("test", level="conjunto")[["rmse_1_mean", "rmse_1_std", "rmse_20_mean", "rmse_20_std",
                                              "params_mean"]].to_string(float_format="%.4f"))
    for key, by_ref in tests.items():
        for ref, comp in by_ref.items():
            for c in comp:
                if ref < c["contra"]:
                    print(f"  {key}: {ref} vs {c['contra']}: p_holm={c['p_holm']:.3g} δ={c['cliffs_delta']:+.3f} "
                          f"({c['efecto']}) {ref} mejor={c['referencia_mejor']}")
    return tests


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", choices=ORDER, default=list(ORDER))
    args = ap.parse_args()
    t0 = time.time()
    ae_path = representation()
    print(f"Representación: {'z (' + ae_path + ')' if ae_path else 'estado crudo normalizado'}")
    summaries = {k: run_model(k, ae_path) for k in args.model}
    if len(args.model) > 1:
        compare_models(args.model)
    ok = all(all(s["beats"].values()) for s in summaries.values())
    print(f"\nSupera a persistencia y ridge en test (1 y 20 pasos): "
          f"{ {s['label']: all(s['beats'].values()) for s in summaries.values()} } -> Criterio de salida Etapa 6: "
          f"{'CUMPLIDO' if ok else 'NO CUMPLIDO'} ({(time.time() - t0) / 60:.1f} min)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
