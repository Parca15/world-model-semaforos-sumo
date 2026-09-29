"""Etapa 8 — PPO directo en SUMO (Condición 2), mismos hiperparámetros que en el sueño.

    python -m wm.experiments.stage8_sumo_ppo

Presupuesto: tantos pasos de SUMO como transiciones se usaron para construir el World Model (train + val).
La curva de aprendizaje (retorno por episodio) se registra frente a los pasos de SUMO consumidos.
Reanudable: una corrida interrumpida continúa desde el último punto de control.
Salidas: runs/ppo_sumo/seed{k}/ppo.zip y curve.csv, results/stage8_sumo_ppo.csv.
"""
from __future__ import annotations

import sys
import time
from multiprocessing import Pool

import pandas as pd
import torch
from stable_baselines3.common.monitor import Monitor

from wm.control.ppo import EpisodeCurve, SumoPPOEnv, learn_resumable, make_ppo, remove_checkpoint
from wm.experiments.common import dataset, load_json, results_path, run_dir, save_json
from wm.utils import load_config, max_workers, set_seed


def train_seed(seed: int) -> dict:
    done = run_dir("ppo_sumo", f"seed{seed}") / "summary.json"
    if done.exists() and (done.parent / "ppo.zip").exists():
        print(f"  PPO SUMO semilla {seed}: ya entrenado, se reutiliza", flush=True)
        return load_json(done)
    torch.set_num_threads(1)
    set_seed(seed)
    cfg = load_config("ppo")
    out = run_dir("ppo_sumo", f"seed{seed}")
    env = Monitor(SumoPPOEnv(dataset("train").norm, cfg["sumo"]["demands"], cfg["sumo"]["seeds"],
                             cfg["reward_scale"]), info_keywords=("demand", "sim_seed"))
    env.reset(seed=seed)
    cb = EpisodeCurve()
    t0 = time.time()
    model = learn_resumable(lambda: make_ppo(env, seed, cfg["sumo"]["n_envs"]), env,
                            cfg["sumo"]["total_timesteps"], out, cb, cfg["rollouts_per_checkpoint"])
    elapsed = time.time() - t0
    model.save(str(out / "ppo.zip"))
    pd.DataFrame(cb.rows).to_csv(out / "curve.csv", index=False)
    remove_checkpoint(out)
    env.close()
    summary = {"seed": seed, "sumo_steps": cfg["sumo"]["total_timesteps"], "train_minutes": elapsed / 60}
    save_json(summary, out / "summary.json")
    print(f"  PPO SUMO semilla {seed}: {elapsed / 60:.1f} min", flush=True)
    return summary


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config("ppo")
    t0 = time.time()
    with Pool(max_workers(), maxtasksperchild=1) as pool:
        rows = pool.map(train_seed, cfg["sumo"]["train_seeds"], chunksize=1)
    pd.DataFrame(rows).to_csv(results_path("stage8_sumo_ppo.csv"), index=False)
    ok = all((run_dir("ppo_sumo", f"seed{s}") / "ppo.zip").exists() for s in cfg["sumo"]["train_seeds"])
    print(f"Criterio de salida Etapa 8 (política × {len(cfg['sumo']['train_seeds'])} semillas): "
          f"{'CUMPLIDO' if ok else 'NO CUMPLIDO'} ({(time.time() - t0) / 60:.1f} min)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
