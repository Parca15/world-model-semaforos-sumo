"""Etapa 9 — evaluación final en SUMO y fidelidad del World Model.

    python -m wm.experiments.stage9_evaluation [--skip-control] [--skip-fidelity]

A. Control: todas las condiciones sobre los MISMOS episodios (demandas D1–D5 × semillas nunca vistas).
   Las condiciones aprendidas se evalúan con cada una de sus 5 semillas de entrenamiento.
B. Fidelidad (sección 4.4): retorno imaginado vs real y concordancia del ranking de acciones (Kendall τ).

Salidas: results/stage9_control_episodes.csv, results/stage9_control_summary.csv, results/stage9_fidelity.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

from wm.control.planning import ImaginationPlanner
from wm.control.policies import Actuated, FixedTime, MaxPressure
from wm.control.ppo import PPOPolicy
from wm.eval.control import run_episode
from wm.eval.fidelity import decision_point, imagined_returns, kendall, logged_fidelity, real_return
from wm.eval.stats import bootstrap_ci
from wm.experiments.common import W, dataset, load_json, results_path, run_dir, save_json
from wm.experiments.stage6_tsmixer import make_predictor, representation
from wm.utils import load_config, max_workers

REFERENCE = ("Tiempo fijo", "Actuado", "Max-Pressure")
LEARNED = ("PPO directo", "WM + TSMixer", "WM + TSMixer + planificación")
METRICS = ["waiting_time_s", "travel_time_s", "queue_mean_m", "queue_max_m", "throughput", "stops",
           "co2_g_per_veh", "fuel_g_per_veh", "reward", "teleports", "backlog_end"]


def tsmixer(seed: int):
    return make_predictor(run_dir("tsmixer", "final", f"seed{seed}") / "model.pt", representation(), "TSMixer")


def build_policy(condition: str, train_seed: int):
    norm = dataset("train").norm
    if condition == "Tiempo fijo":
        return FixedTime()
    if condition == "Actuado":
        return Actuated()
    if condition == "Max-Pressure":
        return MaxPressure(epsilon=0.0)
    if condition == "PPO directo":
        return PPOPolicy(run_dir("ppo_sumo", f"seed{train_seed}") / "ppo.zip", norm)
    dream = PPOPolicy(run_dir("ppo_dream", f"seed{train_seed}") / "ppo.zip", norm)
    if condition == "WM + TSMixer":
        return dream
    p = load_config("ppo")["planning"]
    return ImaginationPlanner(tsmixer(train_seed), norm, W, p["n_candidates"], p["horizon"], base_policy=dream)


def control_job(job: tuple[str, int, str, int]) -> dict:
    torch.set_num_threads(1)
    condition, train_seed, demand, seed = job
    t0 = time.time()
    out = run_episode(build_policy(condition, train_seed), demand, seed, policy_seed=train_seed)
    return {"condition": condition, "train_seed": train_seed, **out, "wall_s": time.time() - t0}


def run_control(workers: int) -> pd.DataFrame:
    ev = load_config("eval")
    seeds = load_config("ppo")["seeds"]
    scen = [(d, s) for d in ev["scenarios"]["demands"] for s in ev["scenarios"]["seeds"]]
    jobs = [(c, 0, d, s) for c in REFERENCE for d, s in scen]
    jobs += [(c, k, d, s) for c in LEARNED for k in seeds for d, s in scen]
    # Reanudable: cada episodio terminado se agrega a un .jsonl y no se repite.
    partial = results_path("stage9_control_episodes.partial.jsonl")
    rows = [json.loads(line) for line in partial.read_text(encoding="utf-8").splitlines()] if partial.exists() else []
    done = {(r["condition"], r["train_seed"], r["demand"], r["seed"]) for r in rows}
    todo = [j for j in jobs if j not in done]
    print(f"Evaluando {len(todo)} de {len(jobs)} episodios en SUMO ({len(scen)} escenarios) ...", flush=True)
    with Pool(workers, maxtasksperchild=4) as pool, open(partial, "a", encoding="utf-8") as f:
        for i, r in enumerate(pool.imap_unordered(control_job, todo, chunksize=1), 1):
            rows.append(r)
            f.write(json.dumps(r) + "
")
            f.flush()
            if i % 20 == 0:
                print(f"  {i}/{len(todo)} episodios", flush=True)
    df = pd.DataFrame(rows).sort_values(["condition", "train_seed", "demand", "seed"])
    df.to_csv(results_path("stage9_control_episodes.csv"), index=False)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Media ± desviación e IC 95 % bootstrap. Las condiciones aprendidas se promedian primero sobre sus
    semillas de entrenamiento en cada episodio, así cada condición aporta un valor por escenario."""
    per_scenario = df.groupby(["condition", "demand", "seed"])[METRICS].mean().reset_index()
    rows = []
    for cond, g in per_scenario.groupby("condition"):
        row = {"condition": cond, "episodes": len(g)}
        for m in METRICS:
            mean, lo, hi = bootstrap_ci(g[m])
            row.update({f"{m}_mean": mean, f"{m}_std": float(g[m].std()), f"{m}_ci_lo": lo, f"{m}_ci_hi": hi})
        rows.append(row)
    order = {c: i for i, c in enumerate(REFERENCE + LEARNED)}
    return pd.DataFrame(rows).sort_values("condition", key=lambda s: s.map(order))


def fidelity_job(args):
    return real_return(args)


def run_fidelity(workers: int) -> dict:
    torch.set_num_threads(1)
    ev, seeds = load_config("eval")["fidelity"], load_config("ppo")["seeds"]
    norm = dataset("train").norm
    out = {"logged": {}, "ranking": {}}
    # 1. retorno imaginado vs real con acciones registradas (test)
    ds40 = dataset("test", 40)
    for k in seeds:
        out["logged"][f"seed{k}"] = logged_fidelity(tsmixer(k), ds40, norm, ev["segments_per_episode"])
    # 2. ranking de acciones: estados de decisión en escenarios de evaluación
    demands = load_config("eval")["scenarios"]["demands"]
    specs = [(d, s, t) for d in demands for s in (9, 10) for t in (30, 60, 90)][:ev["ranking_states"]]
    print(f"Fidelidad de ranking: {len(specs)} estados × {ev['ranking_candidates']} candidatas ...", flush=True)
    with Pool(workers) as pool:
        points = pool.starmap(decision_point, [(d, s, t, ev["ranking_candidates"], norm, W) for d, s, t in specs])
        jobs = [(p.demand, p.seed, p.prefix, c, ev["ranking_horizon"]) for p in points for c in p.candidates]
        real = np.array(pool.map(fidelity_job, jobs, chunksize=1)).reshape(len(points), -1)
    taus = {}
    for k in seeds:
        pred = tsmixer(k)
        taus[f"seed{k}"] = [kendall(imagined_returns(pred, norm, p, ev["ranking_horizon"]), real[i])
                            for i, p in enumerate(points)]
    all_tau = np.array(list(taus.values()))
    best_hit = [float(np.mean([np.argmax(imagined_returns(tsmixer(k), norm, p, ev["ranking_horizon"]))
                               == np.argmax(real[i]) for i, p in enumerate(points)])) for k in seeds[:1]]
    out["ranking"] = {"states": len(points), "candidates": ev["ranking_candidates"], "horizon": ev["ranking_horizon"],
                      "tau_by_seed": {k: float(np.mean(v)) for k, v in taus.items()},
                      "tau_mean": float(all_tau.mean()), "tau_std_over_states": float(all_tau.mean(0).std()),
                      "frac_states_tau_positive": float((all_tau.mean(0) > 0).mean()),
                      "top1_agreement_seed0": best_hit[0],
                      "real_returns": real.tolist(), "states_spec": specs}
    lg = pd.DataFrame(out["logged"]).T
    out["logged_mean"] = lg.mean().to_dict()
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-control", action="store_true")
    ap.add_argument("--skip-fidelity", action="store_true")
    args = ap.parse_args()
    workers = max_workers()
    t0 = time.time()
    df = (pd.read_csv(results_path("stage9_control_episodes.csv")) if args.skip_control
          else run_control(workers))
    summary = summarize(df)
    summary.to_csv(results_path("stage9_control_summary.csv"), index=False)
    cols = ["condition", "episodes"] + [f"{m}_mean" for m in ("waiting_time_s", "travel_time_s", "queue_mean_m",
                                                             "throughput", "co2_g_per_veh", "teleports")]
    print(summary[cols].to_string(index=False, float_format="%.1f"))
    if not args.skip_fidelity:
        fid = run_fidelity(workers)
        save_json(fid, results_path("stage9_fidelity.json"))
        print("Fidelidad (acciones registradas):", {k: round(v, 3) for k, v in fid["logged_mean"].items()})
        print("Kendall τ medio:", round(fid["ranking"]["tau_mean"], 3),
              "| estados con τ > 0:", round(fid["ranking"]["frac_states_tau_positive"], 2))
    needed = {f"{m}_{s}" for m in METRICS for s in ("mean", "std", "ci_lo", "ci_hi")}
    ok = set(REFERENCE + LEARNED) <= set(summary.condition) and needed <= set(summary.columns)
    print(f"Criterio de salida Etapa 9 (tabla completa con media ± desv. e IC 95 %): "
          f"{'CUMPLIDO' if ok else 'NO CUMPLIDO'} ({(time.time() - t0) / 60:.1f} min)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
