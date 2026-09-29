"""Corre las Etapas 5-10 en secuencia (una a la vez), con un log por etapa en runs/logs/.

    python scripts/run_pipeline.py [--from 7] [--to 9]

Todas las etapas son reanudables: si el proceso se interrumpe, volver a lanzarlo reutiliza lo terminado.
El paralelismo dentro de cada etapa se controla con configs/compute.yaml (se lee al empezar cada etapa).
Un criterio de salida no cumplido se reporta pero no detiene el pipeline; un error (Traceback) sí.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGES = {
    5: "wm.experiments.stage5_representation",
    6: "wm.experiments.stage6_tsmixer",
    7: "wm.experiments.stage7_dream_ppo",
    8: "wm.experiments.stage8_sumo_ppo",
    9: "wm.experiments.stage9_evaluation",
    10: "wm.experiments.stage10_analysis",
}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=5)
    ap.add_argument("--to", dest="end", type=int, default=10)
    args = ap.parse_args()
    logs = Path(os.environ.get("WM_RUNS_DIR", ROOT / "runs")) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for n, module in STAGES.items():
        if not args.start <= n <= args.end:
            continue
        log = logs / f"stage{n}.log"
        print(f"[{time.strftime('%H:%M:%S')}] Etapa {n}: inicio ({module})", flush=True)
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as f:
            code = subprocess.run([sys.executable, "-u", "-m", module], cwd=ROOT, stdout=f,
                                  stderr=subprocess.STDOUT).returncode
        text = log.read_text(encoding="utf-8", errors="replace")
        crit = [line for line in text.splitlines() if "Criterio de salida" in line or "Decisión" in line]
        print(f"[{time.strftime('%H:%M:%S')}] Etapa {n}: fin (código {code}, {(time.time() - t0) / 60:.1f} min) "
              f"{crit[-1].strip() if crit else ''}", flush=True)
        if "Traceback" in text:
            print(f"[ERROR] Etapa {n} falló; ver {log}", flush=True)
            return 1
    print("PIPELINE COMPLETO", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
