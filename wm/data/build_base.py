"""Etapa 3 — construye el dataset base congelado (p. ej. data/base_v1).

    python -m wm.data.build_base                      # versión de configs/dataset.yaml
    python -m wm.data.build_base --version smoke --episode-seconds 300   # prueba rápida

Pasos: plan de episodios -> recolección en paralelo -> índice y partición -> normalización (solo train)
-> ficha técnica -> MANIFEST.sha256 -> archivos de solo lectura.
Se construye en un directorio temporal y se renombra al final. Un dataset existente nunca se sobrescribe:
cualquier cambio exige una versión nueva (base_v2, ...).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from wm.data.collect import EpisodeSpec, run_episode
from wm.data.manifest import write_manifest
from wm.data.normalization import NormStats
from wm.env.state_builder import FEATURE_NAMES, FEATURES, RAW_METRICS
from wm.utils import DATA_DIR, ROOT, load_config, max_workers

SPLIT_ORDER = ("train", "val", "test", "test_ood")


# ---------------------------------------------------------------- plan
def plan_episodes(ds_cfg: dict, base_cfg: dict) -> list[EpisodeSpec]:
    splits = base_cfg["splits"]
    groups = [(split, splits[split], ds_cfg["demands"]) for split in ("train", "val", "test")]
    groups.append(("test_ood", splits["test_ood"], [ds_cfg["ood_demand"]]))
    specs = []
    for split, seeds, demands in groups:
        for seed in seeds:
            for demand in demands:
                for pname, pcfg in ds_cfg["policies"].items():
                    specs.append(EpisodeSpec(
                        episode=f"ep_{len(specs):03d}", demand=demand, policy=pname, policy_kind=pcfg["kind"],
                        seed=seed, split=split, policy_params=_policy_params(pcfg, seed)))
    check_no_leakage(specs)
    return specs


def _policy_params(pcfg: dict, seed: int) -> dict:
    if pcfg["kind"] == "random_restricted":
        return {"p": pcfg["p_by_seed_parity"]["odd" if seed % 2 else "even"]}
    if pcfg["kind"] == "max_pressure":
        return {"epsilon": pcfg["epsilon"]}
    return {}


def check_no_leakage(specs: list[EpisodeSpec]) -> None:
    """Cada semilla pertenece a un único conjunto (la partición es por episodio/semilla, nunca por ventana)."""
    owner: dict[int, str] = {}
    for s in specs:
        if owner.setdefault(s.seed, s.split) != s.split:
            raise ValueError(f"La semilla {s.seed} aparece en {owner[s.seed]} y en {s.split}")


# ---------------------------------------------------------------- recolección
def _collect(job: tuple[EpisodeSpec, str, int | None, bool]) -> dict:
    spec, out_dir, episode_seconds, compress = job
    rec = run_episode(spec, episode_seconds)
    rec.save(Path(out_dir) / f"{spec.episode}.npz", compress=compress)
    return rec.summary()


def collect_all(specs: list[EpisodeSpec], ep_dir: Path, workers: int, episode_seconds: int | None,
                compress: bool) -> pd.DataFrame:
    jobs = [(s, str(ep_dir), episode_seconds, compress) for s in specs]
    rows = []
    with Pool(workers) as pool:
        for row in tqdm(pool.imap_unordered(_collect, jobs), total=len(jobs), desc="episodios", unit="ep"):
            rows.append(row)
    df = pd.DataFrame(rows).sort_values("episode").reset_index(drop=True)
    df["file"] = "episodes/" + df["episode"] + ".npz"
    return df


# ---------------------------------------------------------------- metadatos
def feature_schema() -> dict:
    return {
        "state_shape": ["n_tls", len(FEATURES)],
        "features": [{"index": i, "name": n, "unit": u, "normalized": norm} for i, (n, u, norm) in enumerate(FEATURES)],
        "raw_metrics": RAW_METRICS,
        "arrays": {
            "states": "[T+1, n_tls, n_features] s_0..s_T (unidades originales)",
            "actions": "[T, n_tls] acción efectiva a_t (1 = empezó un cambio de fase)",
            "forced": "[T, n_tls] el cambio lo forzó el verde máximo",
            "rewards": "[T, n_tls] r_t^i = -(W_{t+1}^i - W_t^i)/100",
            "raw_metrics": "[T+1, n_tls, k] métricas en bruto (ver raw_metrics)",
        },
    }


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _git_commit() -> str:
    return _git("rev-parse", "HEAD") or "desconocido"


def git_is_dirty() -> bool:
    """True si hay cambios sin commitear en código o configuración (los datos generados no cuentan)."""
    return bool(_git("status", "--porcelain", "--", "wm", "configs", "sumo"))


def config_snapshot(version: str, episode_seconds: int | None) -> dict:
    from importlib.metadata import version as pkg_version

    packages = ("eclipse-sumo", "libsumo", "numpy", "pandas", "torch", "gymnasium", "stable-baselines3")
    return {
        "version": version,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": _git_commit(),
        "git_dirty": git_is_dirty(),
        "episode_seconds_override": episode_seconds,
        "software": {"python": platform.python_version(), "platform": platform.platform(),
                     **{p: pkg_version(p) for p in packages}},
        "configs": {name: load_config(name) for name in ("base", "scenario", "dataset")},
    }


def fit_normalization(root: Path, index: pd.DataFrame) -> NormStats:
    train = index[index["split"] == "train"]
    states, rewards = [], []
    for f in train["file"]:
        with np.load(root / f) as z:
            states.append(z["states"])
            rewards.append(z["rewards"])
    mask = np.array([norm for _, _, norm in FEATURES])
    return NormStats.fit(states, rewards, mask)


def dataset_card(root: Path, index: pd.DataFrame, norm: NormStats, cfg: dict) -> str:
    """Ficha técnica en Markdown con estadísticas descriptivas."""
    split_tbl = index.groupby("split").agg(episodios=("episode", "count"), transiciones=("n_steps", "sum"),
                                           semillas=("seed", lambda s: sorted(set(s))))
    split_tbl = split_tbl.reindex([s for s in SPLIT_ORDER if s in split_tbl.index])
    pol_tbl = index.groupby("policy").agg(tipo=("policy_kind", "first"), tasa_cambio=("switch_rate", "mean"),
                                          retorno=("return", "mean"), cola_m=("mean_queue_m", "mean"),
                                          espera_s=("mean_waiting_s", "mean"))
    dem_tbl = index.groupby("demand").agg(episodios=("episode", "count"), retorno=("return", "mean"),
                                          cola_m=("mean_queue_m", "mean"), espera_s=("mean_waiting_s", "mean"))
    states = []
    for f in index.loc[index["split"] == "train", "file"]:
        with np.load(root / f) as z:
            states.append(z["states"].reshape(-1, len(FEATURES)))
    s = np.concatenate(states)
    feat_rows = [f"| {n} | {u} | {'sí' if nz else 'no'} | {s[:, i].mean():.2f} | {s[:, i].std():.2f} | "
                 f"{s[:, i].min():.2f} | {np.percentile(s[:, i], 50):.2f} | {s[:, i].max():.2f} |"
                 for i, (n, u, nz) in enumerate(FEATURES)]
    base = cfg["configs"]["base"]
    return f"""# Dataset `{cfg['version']}` — ficha técnica

Generado el {cfg['created']} desde el commit `{cfg['git_commit'][:10]}`. **Congelado y de solo lectura:**
cualquier cambio exige una versión nueva y reentrenar los tres modelos temporales.

- Red: `{cfg['configs']['scenario']['network']['name']}` ({len(base['network']['tls_ids'])} intersecciones), Δt = {base['env']['delta_t']} s,
  {base['env']['warmup']} s de calentamiento no registrados.
- Estado `[{len(base['network']['tls_ids'])}, {len(FEATURES)}]` por paso; acción efectiva por intersección; recompensa por intersección.
- Partición **por semilla** (episodios completos, sin ventanas que crucen episodios ni conjuntos).
- Normalización z-score por (intersección, variable) ajustada **solo con train** (`norm_stats.json`).
- Acceso único: `wm.data.base.load_base("{cfg['version']}", split, W, H)`, que verifica `MANIFEST.sha256`.

## Partición

{split_tbl.to_markdown()}

## Políticas de recolección

{pol_tbl.to_markdown(floatfmt=".3f")}

## Demandas

{dem_tbl.to_markdown(floatfmt=".2f")}

## Variables del estado (train, unidades originales, todas las intersecciones)

| variable | unidad | normalizada | media | desv. | mín. | mediana | máx. |
|---|---|---|---|---|---|---|---|
{chr(10).join(feat_rows)}

Recompensa por paso (train): media {norm.reward_mean.mean():.3f}, desviación media {norm.reward_std.mean():.3f}.
"""


def make_read_only(root: Path) -> None:
    for p in root.rglob("*"):
        if p.is_file():
            os.chmod(p, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)


# ---------------------------------------------------------------- main
def build(version: str, workers: int, episode_seconds: int | None = None, limit: int | None = None,
          data_dir: Path = DATA_DIR) -> Path:
    ds_cfg, base_cfg = load_config("dataset"), load_config("base")
    out = data_dir / version
    if out.exists():
        raise FileExistsError(f"{out} ya existe y está congelado. Para cambiar el dataset cree una versión nueva.")
    tmp = data_dir / f".{version}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "episodes").mkdir(parents=True)

    specs = plan_episodes(ds_cfg, base_cfg)[:limit]
    t0 = time.time()
    index = collect_all(specs, tmp / "episodes", workers, episode_seconds, ds_cfg["collection"]["compress"])
    print(f"{len(index)} episodios recolectados en {(time.time() - t0) / 60:.1f} min")

    index.to_parquet(tmp / "index.parquet", index=False)
    splits = {s: index.loc[index["split"] == s, "episode"].tolist() for s in SPLIT_ORDER}
    (tmp / "splits.json").write_text(json.dumps(splits, indent=1), encoding="utf-8")
    (tmp / "feature_schema.json").write_text(json.dumps(feature_schema(), indent=1, ensure_ascii=False),
                                             encoding="utf-8")
    norm = fit_normalization(tmp, index)
    norm.save(tmp / "norm_stats.json", FEATURE_NAMES, base_cfg["network"]["tls_ids"])
    cfg = config_snapshot(version, episode_seconds)
    (tmp / "config_used.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (tmp / "README.md").write_text(dataset_card(tmp, index, norm, cfg), encoding="utf-8")
    write_manifest(tmp)
    tmp.rename(out)
    make_read_only(out)
    print(f"Dataset congelado en {out}")
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ds_cfg = load_config("dataset")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default=ds_cfg["version"])
    ap.add_argument("--workers", type=int, default=max_workers())
    ap.add_argument("--episode-seconds", type=int, default=None, help="solo para pruebas rápidas")
    ap.add_argument("--limit", type=int, default=None, help="solo los primeros N episodios (pruebas)")
    ap.add_argument("--allow-dirty", action="store_true", help="permitir cambios sin commitear en wm/, configs/, sumo/")
    args = ap.parse_args()
    if git_is_dirty() and not args.allow_dirty:
        print("Hay cambios sin commitear en wm/, configs/ o sumo/. Haga commit antes de construir el dataset "
              "(para que config_used.yaml apunte a un código exacto) o use --allow-dirty.")
        return 1
    build(args.version, args.workers, args.episode_seconds, args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
