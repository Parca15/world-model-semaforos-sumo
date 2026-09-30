"""Etapa 7 — PPO entrenado dentro del Dream Environment de cada World Model (LSTM, TSMixer, Transformer).

    python -m wm.experiments.stage7_dream_ppo [--model tsmixer lstm transformer]

La política (modelo m, semilla k) se entrena en el sueño del modelo m de la semilla k, con los mismos
hiperparámetros de PPO para los tres. Criterio de salida: una política por modelo temporal × 5 semillas.
Salidas: runs/ppo_dream/<modelo>/seed{k}/ppo.zip y curve.csv, results/stage7_dream_ppo.csv.
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool

import pandas as pd
import torch
from stable_baselines3.common.callbacks import BaseCallback

from wm.control.ppo import learn_resumable, make_ppo, remove_checkpoint
from wm.dream.dream_env import DreamVecEnv
from wm.experiments.common import dataset, flat, load_json, results_path, run_dir, save_json, world_model
from wm.models.temporal import TEMPORAL_MODELS
from wm.utils import load_config, max_workers, set_seed


class CurveCallback(BaseCallback):
    """Registra el retorno medio de los episodios imaginados en cada actualización de PPO."""

    def __init__(self):
        super().__init__()
        self.rows = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        buf = self.model.ep_info_buffer
        if buf:
            self.rows.append({"timesteps": self.num_timesteps,
                              "dream_return_mean": sum(e["r"] for e in buf) / len(buf)})


def train_seed(job: tuple[str, int]) -> dict:
    kind, seed = job
    label = TEMPORAL_MODELS[kind].label
    out = run_dir("ppo_dream", kind, f"seed{seed}")
    if (out / "summary.json").exists() and (out / "ppo.zip").exists():
        print(f"  PPO sueño {label} semilla {seed}: ya entrenado, se reutiliza", flush=True)
        return load_json(out / "summary.json")
    torch.set_num_threads(1)
    set_seed(seed)
    cfg = load_config("ppo")
    predictor = world_model(kind, seed)
    env = DreamVecEnv(predictor, dataset("train").norm, flat(cfg["dream"]["reset_split"]), cfg["dream"]["n_envs"],
                      cfg["dream"]["episode_steps"], cfg["reward_scale"], seed=seed)
    cb = CurveCallback()
    t0 = time.time()
    model = learn_resumable(lambda: make_ppo(env, seed, cfg["dream"]["n_envs"]), env,
                            cfg["dream"]["total_timesteps"], out, cb, cfg["rollouts_per_checkpoint"])
    elapsed = time.time() - t0
    model.save(str(out / "ppo.zip"))
    pd.DataFrame(cb.rows).to_csv(out / "curve.csv", index=False)
    remove_checkpoint(out)
    summary = {"model": kind, "seed": seed, "timesteps": cfg["dream"]["total_timesteps"],
               "train_minutes": elapsed / 60, "sumo_steps": 0, "final_dream_return": cb.rows[-1]["dream_return_mean"] if cb.rows else None}
    save_json(summary, out / "summary.json")
    print(f"  PPO sueño {label} semilla {seed}: {elapsed / 60:.1f} min, retorno imaginado final "
          f"{summary['final_dream_return']:.2f}", flush=True)
    return summary


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", choices=list(TEMPORAL_MODELS), default=list(TEMPORAL_MODELS))
    args = ap.parse_args()
    seeds = load_config("ppo")["seeds"]
    jobs = [(k, s) for k in args.model for s in seeds]
    t0 = time.time()
    with Pool(min(max_workers(), len(jobs)), maxtasksperchild=1) as pool:
        rows = pool.map(train_seed, jobs, chunksize=1)
    pd.DataFrame(rows).to_csv(results_path("stage7_dream_ppo.csv"), index=False)
    ok = all((run_dir("ppo_dream", k, f"seed{s}") / "ppo.zip").exists() for k, s in jobs)
    print(f"Criterio de salida Etapa 7 (política por modelo temporal × {len(seeds)} semillas): "
          f"{'CUMPLIDO' if ok else 'NO CUMPLIDO'} "
          f"({(time.time() - t0) / 60:.1f} min)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
