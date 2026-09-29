"""Ejecuta entrenamientos de TSMixer en paralelo (1 proceso y 1 hilo de torch por entrenamiento).

Un trabajo es un dict:
    name         identificador (carpeta en runs/)
    arch         hiperparámetros de arquitectura (n_blocks, d_model, dropout, norm)
    lr, weight_decay, seed, max_epochs
    out_dir      carpeta donde se guardan model.pt, log.jsonl y fit.json
    ae_path      (opcional) checkpoint de AE/VAE: el modelo trabaja en el espacio latente
Los trabajos son reanudables: si `out_dir/fit.json` ya existe para el MISMO trabajo, se reutiliza; si un
entrenamiento quedó a medias, continúa desde la última época terminada (`out_dir/train_state.pt`).
"""
from __future__ import annotations

import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from wm.data.base import FlatWindows
from wm.experiments.common import flat, load_json, save_checkpoint, save_json, train_config
from wm.models.tsmixer import build_tsmixer, count_params
from wm.train.trainer import fit
from wm.utils import load_config, max_workers, set_seed, threads_per_worker


@torch.no_grad()
def latent_flat(fw: FlatWindows, ae, state_dim: int) -> FlatWindows:
    """Transforma una vista de ventanas al espacio latente: filas (z_t, a_t), objetivo Δz."""
    s = torch.from_numpy(fw.rows[:, :state_dim])
    z = ae.encode(s)
    z_next = ae.encode(s + torch.from_numpy(fw.delta))
    rows = torch.cat([z, torch.from_numpy(fw.rows[:, state_dim:])], 1).numpy()
    return FlatWindows(rows, (z_next - z).numpy(), fw.reward, fw.ends, fw.W)


def _job_key(job: dict) -> str:
    """Identidad del trabajo: sus parámetros más el protocolo de entrenamiento común (p. ej. la pérdida)."""
    train = {k: v for k, v in load_config("train").items() if k not in ("search", "final_seeds")}
    return json.dumps({"job": {k: v for k, v in job.items() if k not in ("out_dir", "threads")},
                       "train": train}, sort_keys=True)


def run_job(job: dict) -> dict:
    out = Path(job["out_dir"])
    if (out / "fit.json").exists() and (out / "model.pt").exists():
        done = load_json(out / "fit.json")
        if done.get("job_key") == _job_key(job):
            print(f"  [{job['name']}] ya entrenado: se reutiliza", flush=True)
            return done
    threads = job.get("threads", 1)
    torch.set_num_threads(threads)
    set_seed(job["seed"])
    out.mkdir(parents=True, exist_ok=True)
    k = load_config("train")["loss"]["rollout_k"]
    train, val = flat("train", k), flat("val", k)
    state_dim, n_tls = train.delta.shape[1], train.reward.shape[1]
    if job.get("ae_path"):
        ae = torch.load(job["ae_path"], map_location="cpu")["model"].eval()
        train, val = latent_flat(train, ae, state_dim), latent_flat(val, ae, state_dim)
    budget = load_config("models")["param_budget"]["target"]
    model = build_tsmixer(budget, input_dim=train.rows.shape[1], state_dim=train.delta.shape[1], n_tls=n_tls,
                          window=train.W, **job["arch"])
    tcfg = train_config(lr=job["lr"], weight_decay=job["weight_decay"], seed=job["seed"],
                        max_epochs=job["max_epochs"], threads=threads)
    t0 = time.time()
    key = _job_key(job)
    model, res = fit(model, train, val, tcfg, out / "log.jsonl", state_path=out / "train_state.pt", state_tag=key)
    summary = {"name": job["name"], "arch": job["arch"], "ff_dim": model.cfg.ff_dim, "lr": job["lr"],
               "weight_decay": job["weight_decay"], "seed": job["seed"], "params": count_params(model),
               "ae_path": job.get("ae_path"), **{k: v for k, v in res.to_dict().items() if k != "history"},
               "wall_seconds": time.time() - t0, "threads": threads, "job_key": key}
    save_checkpoint(model, "tsmixer", model.cfg.to_dict(), out / "model.pt", {"fit": summary})
    save_json(summary, out / "fit.json")
    (out / "train_state.pt").unlink(missing_ok=True)
    print(f"  [{job['name']}] params={summary['params']} val={res.best_val_loss:.4f} "
          f"época={res.best_epoch}/{res.epochs_run} {res.train_seconds / 60:.1f} min", flush=True)
    return summary


def run_jobs(jobs: list[dict], workers: int | None = None) -> list[dict]:
    workers = min(workers or max_workers(), len(jobs))
    jobs = [{**j, "threads": threads_per_worker(workers)} for j in jobs]
    with Pool(workers, maxtasksperchild=1) as pool:
        return pool.map(run_job, jobs, chunksize=1)
