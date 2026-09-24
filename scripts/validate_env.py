"""Etapa 2 — criterio de salida del entorno.

Corre un episodio completo (720 pasos) con acciones aleatorias en cada demanda D1..D5, verifica que no haya
errores y que las variables queden en rangos físicos válidos, y guarda las estadísticas por variable en
results/stage2_env_check.csv.

Uso:  python scripts/validate_env.py [--seed 1]
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wm.env.state_builder import FEATURE_NAMES  # noqa: E402
from wm.utils import RESULTS_DIR  # noqa: E402

F = {n: i for i, n in enumerate(FEATURE_NAMES)}


def run(args: tuple[str, int]) -> dict:
    from wm.env.traffic_env import make_env

    demand, seed = args
    env = make_env(demand=demand)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    obs, info = env.reset(seed=seed)
    O, A, R = [obs], [], []
    trunc = False
    while not trunc:
        obs, r, _, trunc, info = env.step(rng.integers(0, 2, 7))
        O.append(obs), A.append(info["action"]), R.append(r)
    env.close()
    O = np.stack(O)
    checks = {
        "720 pasos": len(A) == 720,
        "finitos y >= 0": bool(np.isfinite(O).all() and (O >= 0).all()),
        "detenidos <= vehículos": bool((O[..., F["halting"]] <= O[..., F["vehicles"]]).all()),
        "velocidad <= 30 m/s": bool((O[..., F["mean_speed"]] <= 30).all()),
        "ocupación <= 100 %": bool((O[..., F["occupancy"]] <= 100).all()),
        "one-hot de fase válido": bool(np.allclose(O[..., 7:11].sum(-1), 1)),
        "verde <= 60 s": bool((O[..., F["time_in_phase"]] <= 60).all()),
    }
    return {"demand": demand, "O": O, "A": np.stack(A), "R": np.array(R), "checks": checks, "secs": time.time() - t0}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    demands = ["D1", "D2", "D3", "D4", "D5"]
    with Pool(min(4, len(demands))) as pool:
        results = pool.map(run, [(d, args.seed) for d in demands])

    ok = True
    rows = []
    for res in results:
        good = all(res["checks"].values())
        ok &= good
        failed = [k for k, v in res["checks"].items() if not v]
        print(f"{res['demand']}: {len(res['A'])} pasos en {res['secs']:.0f} s | retorno {res['R'].sum():8.1f} | "
              f"cambios efectivos {int(res['A'].sum())} | {'OK' if good else 'FALLA ' + str(failed)}")
        O = res["O"].reshape(-1, len(FEATURE_NAMES))
        for j, name in enumerate(FEATURE_NAMES):
            rows.append((res["demand"], name, O[:, j].min(), O[:, j].mean(), O[:, j].max()))

    out = RESULTS_DIR / "stage2_env_check.csv"
    with open(out, "w", encoding="utf-8") as f:
        f.write("demand,feature,min,mean,max\n")
        for d, n, lo, mu, hi in rows:
            f.write(f"{d},{n},{lo:.3f},{mu:.3f},{hi:.3f}\n")
    print(f"\n{'variable':18} " + " ".join(f"{d:>22}" for d in demands))
    for j, name in enumerate(FEATURE_NAMES):
        cells = [f"{r[2]:6.1f}/{r[3]:7.1f}/{r[4]:7.1f}" for r in rows if r[1] == name]
        print(f"{name:18} " + " ".join(f"{c:>22}" for c in cells))
    print(f"\n(min/media/max por demanda) -> {out}")
    print("Criterio de salida Etapa 2:", "CUMPLIDO" if ok else "NO CUMPLIDO")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
