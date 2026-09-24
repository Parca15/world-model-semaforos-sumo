"""Etapa 3 — criterio de salida del dataset congelado.

  * los checksums de MANIFEST.sha256 verifican;
  * las estadísticas descriptivas están documentadas (README.md del dataset);
  * no hay fuga entre conjuntos (semillas disjuntas, normalización solo con train, ventanas dentro del episodio);
  * el dataset es de solo lectura.

Uso:  python scripts/validate_dataset.py [--version base_v1]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wm.data.base import dataset_root, load_base  # noqa: E402
from wm.data.manifest import verify_manifest  # noqa: E402
from wm.data.normalization import NormStats  # noqa: E402
from wm.env.state_builder import FEATURES  # noqa: E402
from wm.utils import load_config  # noqa: E402


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=load_config("dataset")["version"])
    args = ap.parse_args()
    root = dataset_root(args.version)
    base = load_config("base")
    W, H = base["window"]["W"], base["window"]["H"]
    checks: dict[str, bool] = {}

    n = verify_manifest(root)
    checks[f"MANIFEST.sha256 verifica ({n} archivos)"] = True

    index = pd.read_parquet(root / "index.parquet")
    splits = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    seeds = {s: set(index.set_index("episode").loc[eps, "seed"]) for s, eps in splits.items()}
    names = list(seeds)
    checks["semillas disjuntas entre conjuntos"] = all(
        not (seeds[a] & seeds[b]) for i, a in enumerate(names) for b in names[i + 1:])
    checks["cada episodio en un solo conjunto"] = sorted(sum(splits.values(), [])) == sorted(index.episode)
    checks["partición según configs/base.yaml"] = all(
        seeds[s] == set(base["splits"][s]) for s in ("train", "val", "test", "test_ood"))

    mask = np.array([nz for _, _, nz in FEATURES])
    states, rewards = [], []
    for f in index.loc[index.split == "train", "file"]:
        with np.load(root / f) as z:
            states.append(z["states"]), rewards.append(z["rewards"])
    refit = NormStats.fit(states, rewards, mask)
    norm = NormStats.load(root / "norm_stats.json")
    checks["normalización = estadísticas de train"] = bool(
        np.allclose(norm.state_mean, refit.state_mean) and np.allclose(norm.state_std, refit.state_std))

    sizes = {}
    for split in splits:
        ds = load_base(args.version, split, W=W, H=H)
        inside = all(t - W + 1 >= 0 and t + H <= ds.episodes[e].T for e, t in ds.windows)
        checks[f"ventanas de {split} dentro de su episodio"] = inside
        sizes[split] = (len(ds.episodes), sum(ep.T for ep in ds.episodes), len(ds))

    checks["README.md con estadísticas descriptivas"] = "Variables del estado" in (root / "README.md").read_text(
        encoding="utf-8")
    checks["archivos de solo lectura"] = not any(os.access(p, os.W_OK) for p in root.rglob("*") if p.is_file())

    print(f"Dataset {args.version} ({root})\n")
    print(f"{'conjunto':9} {'episodios':>9} {'transiciones':>12} {'ventanas W=' + str(W) + ',H=' + str(H):>18}")
    for s, (e, t, w) in sizes.items():
        print(f"{s:9} {e:9d} {t:12d} {w:18d}")
    print()
    for k, v in checks.items():
        print(f"[{'OK' if v else 'FALLA'}] {k}")
    ok = all(checks.values())
    print("\nCriterio de salida Etapa 3:", "CUMPLIDO" if ok else "NO CUMPLIDO")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
