"""Etapa 1 — criterio de salida del escenario.

Para cada demanda D1..D5 corre SUMO N veces con la misma semilla y el programa de tiempo fijo:
  * las salidas (tripinfo + statistics, sin comentarios de cabecera) deben ser idénticas;
  * teletransportes < 1 % de los vehículos insertados.
También dibuja la red en results/figures/<red>_network.png.

Uso:  python scripts/validate_scenario.py [--runs 5] [--seed 1]
"""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wm.utils import RESULTS_DIR, SUMO_DIR, load_config, sumo_binary  # noqa: E402

NAME = load_config("scenario")["network"]["name"]


def run_once(demand: str, seed: int, outdir: Path, tag: str) -> dict:
    trip, stats = outdir / f"{demand}_{tag}_trip.xml", outdir / f"{demand}_{tag}_stats.xml"
    subprocess.run(
        [sumo_binary("sumo"), "-c", str(SUMO_DIR / f"{NAME}.sumocfg"),
         "-r", str(SUMO_DIR / f"routes_{demand}.rou.xml"), "--seed", str(seed),
         "--tripinfo-output", str(trip), "--statistic-output", str(stats),
         "--duration-log.disable", "true"],
        check=True, capture_output=True,
    )
    digest = hashlib.sha256()
    for f in (trip, stats):
        text = re.sub(r"<!--.*?-->", "", f.read_text(encoding="utf-8"), flags=re.S)
        # la línea <performance> registra tiempos de reloj del computador: se excluye del hash
        text = re.sub(r"<performance [^>]*/>", "", text)
        digest.update(text.encode())
    root = ET.parse(stats).getroot()
    veh, tele, trips = root.find("vehicles"), root.find("teleports"), root.find("vehicleTripStatistics")
    safety = root.find("safety")
    return {
        "hash": digest.hexdigest()[:16],
        "inserted": int(veh.get("inserted")),
        "loaded": int(veh.get("loaded")),
        "running_end": int(veh.get("running")),
        "waiting_end": int(veh.get("waiting")),
        "teleports": int(tele.get("total")),
        "collisions": int(safety.get("collisions")),
        "mean_wait": float(trips.get("waitingTime")),
        "mean_travel": float(trips.get("duration")),
    }


def plot_network(path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sumolib
    from matplotlib.lines import Line2D

    cfg = load_config("scenario")["network"]
    classes = cfg["road_classes"]
    # rampa ordinal de un solo tono (azul 250 -> 700): más oscuro y grueso = mayor jerarquía
    style = {"local": ("#86b6ef", 1.0), "connector": ("#3987e5", 1.4), "collector": ("#1c5cab", 1.8),
             "avenue": ("#0d366b", 2.2)}
    net = sumolib.net.readNet(str(SUMO_DIR / f"{NAME}.net.xml"))
    fig, ax = plt.subplots(figsize=(11, 7))
    for e in net.getEdges():
        color, lw = style[e.getType()]
        for lane in e.getLanes():
            xs, ys = zip(*lane.getShape())
            ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round", zorder=1)
    for tls in net.getTrafficLights():
        x, y = net.getNode(tls.getID()).getCoord()
        ax.plot(x, y, "o", ms=13, mfc="white", mec="#1f1f1d", mew=2, zorder=3)
        ax.annotate(tls.getID(), (x, y), xytext=(11, 9), textcoords="offset points", fontsize=11,
                    weight="bold", color="#1f1f1d", zorder=4)
    for nid in cfg["fringe_nodes"]:
        x, y = net.getNode(nid).getCoord()
        ax.plot(x, y, "s", ms=5, color="#6b6a66", zorder=2)
        ax.annotate(nid, (x, y), xytext=(5, -12), textcoords="offset points", fontsize=9, color="#6b6a66")
    handles = [Line2D([], [], color=style[c][0], lw=style[c][1] * 2.5,
                      label=f"{classes[c]['label']} ({classes[c]['lanes']}+{classes[c]['lanes']} carriles, "
                            f"{classes[c]['speed'] * 3.6:.0f} km/h)")
               for c in ("avenue", "collector", "connector", "local")]
    handles += [Line2D([], [], marker="o", ls="", ms=10, mfc="white", mec="#1f1f1d", mew=2,
                       label="Intersección semaforizada"),
                Line2D([], [], marker="s", ls="", ms=5, color="#6b6a66", label="Entrada / salida")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9)
    ax.set_aspect("equal")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors="#6b6a66", labelsize=9)
    ax.set_xlabel("x (m)", color="#6b6a66")
    ax.set_ylabel("y (m)", color="#6b6a66")
    ax.set_title(f"Malla urbana irregular de 7 intersecciones semaforizadas ({NAME})", loc="left", fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    demands = list(load_config("scenario")["demands"])
    ok = True
    with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(args.jobs) as pool:
        futs = {(d, r): pool.submit(run_once, d, args.seed, Path(tmp), str(r)) for d in demands for r in range(args.runs)}
        print(f"{'Dem':4} {'idénticas':9} {'cargados':>8} {'insert.':>8} {'teleports':>9} {'%tel':>6} {'colis.':>6} "
              f"{'espera(s)':>9} {'viaje(s)':>8} {'en red al final':>15}")
        for d in demands:
            res = [futs[(d, r)].result() for r in range(args.runs)]
            same = len({r["hash"] for r in res}) == 1
            r0 = res[0]
            pct = 100 * r0["teleports"] / max(r0["inserted"], 1)
            good = same and pct < 1.0 and r0['collisions'] == 0
            ok &= good
            print(f"{d:4} {str(same):9} {r0['loaded']:8d} {r0['inserted']:8d} {r0['teleports']:9d} {pct:6.2f} {r0['collisions']:6d} "
                  f"{r0['mean_wait']:9.1f} {r0['mean_travel']:8.1f} {r0['running_end'] + r0['waiting_end']:15d}"
                  f"  {'OK' if good else 'FALLA'}")

    fig = RESULTS_DIR / "figures" / f"{NAME}_network.png"
    plot_network(fig)
    print(f"\nFigura de la red: {fig}")
    print("Criterio de salida Etapa 1:", "CUMPLIDO" if ok else "NO CUMPLIDO")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
