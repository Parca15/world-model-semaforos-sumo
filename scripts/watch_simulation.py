"""Muestra en sumo-gui, en tiempo real, un episodio de evaluación de cualquier condición de la Etapa 9.

    python scripts/watch_simulation.py                                   # WM + LSTM + planificación, D3, semilla 9
    python scripts/watch_simulation.py --condicion "Tiempo fijo" --demanda D1 --delay 50
    python scripts/watch_simulation.py --listar

La simulación arranca sola; en la ventana se puede pausar, cambiar la velocidad (Delay) o hacer zoom.
Al terminar se imprimen las mismas métricas que en la evaluación.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wm.eval.control import run_episode  # noqa: E402
from wm.experiments.stage9_evaluation import LEARNED, REFERENCE, build_policy  # noqa: E402

CONDITIONS = REFERENCE + LEARNED


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condicion", default="WM + LSTM + planificación", choices=CONDITIONS, metavar="CONDICIÓN")
    ap.add_argument("--demanda", default="D3", choices=["D1", "D2", "D3", "D4", "D5"])
    ap.add_argument("--semilla", type=int, default=9, help="semilla de la simulación (9-15: nunca vistas)")
    ap.add_argument("--semilla-politica", type=int, default=0, help="semilla de entrenamiento de la política")
    ap.add_argument("--delay", type=int, default=100, help="ms de pausa por paso de simulación (0 = máxima velocidad)")
    ap.add_argument("--listar", action="store_true", help="listar las condiciones disponibles")
    args = ap.parse_args()
    if args.listar:
        print("\n".join(CONDITIONS))
        return 0
    print(f"{args.condicion} | demanda {args.demanda} | semilla {args.semilla} | política semilla "
          f"{args.semilla_politica}: abriendo sumo-gui ...", flush=True)
    policy = build_policy(args.condicion, args.semilla_politica)
    out = run_episode(policy, args.demanda, args.semilla, policy_seed=args.semilla_politica, gui=True,
                      gui_args=("--start", "--delay", str(args.delay)))
    for k in ("waiting_time_s", "travel_time_s", "queue_mean_m", "throughput", "co2_g_per_veh", "teleports"):
        print(f"  {k}: {out[k]:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
