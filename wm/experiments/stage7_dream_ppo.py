"""Etapa 7 — PPO entrenado dentro del Dream Environment del World Model TSMixer (una política por semilla).

    python -m wm.experiments.stage7_dream_ppo

La política de la semilla k se entrena en el sueño del TSMixer de la semilla k. Criterio de salida: una
política entrenada × 5 semillas. Salidas: runs/ppo_dream/seed{k}/ppo.zip y curve.csv, results/stage7_dream_ppo.csv.
"""
from __future__ import annotations

import sys
import time
from multiprocessing import Pool

import pandas as pd
import torch
from stable_baselines3.common.callbacks import BaseCallback

from wm.control.ppo import make_ppo
from wm.dream.dream_env import DreamVecEnv
from wm.experiments.common import dataset, load_json, flat, results_path, run_dir, save_json
from wm.experiments.stage6_tsmixer import make_predictor, representation
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


def train_seed(seed: int) -> dict:
    done = run_dir("ppo_dream", f"seed{seed}") / "summary.json"
    if done.exists() and (done.parent / "ppo.zip").exists():
        print(f"  PPO sueño semilla {seed}: ya entrenado, se reutiliza", flush=True)
        return load_json(done)
    torch.set_num_threads(1)
    set_seed(seed)
    cfg = load_config("ppo")
    predictor = make_predictor(run_dir("tsmixer", "final", f"seed{seed}") / "model.pt", representation(), "TSMixer")
    env = DreamVecEnv(predictor, dataset("train").norm, flat(cfg["dream"]["reset_split"]), cfg["dream"]["n_envs"],
                      cfg["dream"]["episode_steps"], cfg["reward_scale"], seed=seed)
    model = make_ppo(env, seed, cfg["dream"]["n_envs"])
    cb = CurveCallback()
    t0 = time.time()
    model.learn(cfg["dream"]["total_timesteps"], callback=cb)
    elapsed = time.time() - t0
    out = run_dir("ppo_dream", f"seed{seed}")
    model.save(str(out / "ppo.zip"))
    pd.DataFrame(cb.rows).to_csv(out / "curve.csv", index=False)
    summary = {"seed": seed, "timesteps": cfg["dream"]["total_timesteps"], "train_minutes": elapsed / 60,
               "sumo_steps": 0, "final_dream_return": cb.rows[-1]["dream_return_mean"] if cb.rows else None}
    save_json(summary, out / "summary.json")
    print(f"  PPO sueño semilla {seed}: {elapsed / 60:.1f} min, retorno imaginado final "
          f"{summary['final_dream_return']:.2f}", flush=True)
    return summary


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    seeds = load_config("ppo")["seeds"]
    t0 = time.time()
    with Pool(min(max_workers(), len(seeds)), maxtasksperchild=1) as pool:
        rows = pool.map(train_seed, seeds, chunksize=1)
    pd.DataFrame(rows).to_csv(results_path("stage7_dream_ppo.csv"), index=False)
    ok = all((run_dir("ppo_dream", f"seed{s}") / "ppo.zip").exists() for s in seeds)
    print(f"Criterio de salida Etapa 7 (política × {len(seeds)} semillas): {'CUMPLIDO' if ok else 'NO CUMPLIDO'} "
          f"({(time.time() - t0) / 60:.1f} min)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
