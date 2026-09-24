"""Etapa 1 — criterio de salida del escenario.

Para cada demanda D1..D5 corre SUMO N veces con la misma semilla y el programa de tiempo fijo:
  * las salidas (tripinfo + statistics, sin comentarios de cabecera) deben ser idénticas;
  * teletransportes < 1 % de los vehículos insertados.
También dibuja la red en results/figures/corridor7_network.png.

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


def run_once(demand: str, seed: int, outdir: Path, tag: str) -> dict:
    trip, stats = outdir / f"{demand}_{tag}_trip.xml", outdir / f"{demand}_{tag}_stats.xml"
    subprocess.run(
        [sumo_binary("sumo"), "-c", str(SUMO_DIR / "corridor7.sumocfg"),
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
    return {
        "hash": digest.hexdigest()[:16],
        "inserted": int(veh.get("inserted")),
        "loaded": int(veh.get("loaded")),
        "running_end": int(veh.get("running")),
        "waiting_end": int(veh.get("waiting")),
        "teleports": int(tele.get("total")),
        "mean_wait": float(trips.get("waitingTime")),
        "mean_travel": float(trips.get("duration")),
    }


def plot_network(path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sumolib

    net = sumolib.net.readNet(str(SUMO_DIR / "corridor7.net.xml"))
    fig, ax = plt.subplots(figsize=(12, 3.2))
    for e in net.getEdges():
        for lane in e.getLanes():
            xs, ys = zip(*lane.getShape())
            ax.plot(xs, ys, color="0.35", lw=1.0)
    for tls in net.getTrafficLights():
        x, y = net.getNode(tls.getID()).getCoord()
        ax.plot(x, y, "o", color="#d62728", ms=9, zorder=3)
        ax.annotate(tls.getID(), (x, y), xytext=(8, 8), textcoords="offset points", fontsize=10, weight="bold")
    for nid in ["W", "E"]:
        x, y = net.getNode(nid).getCoord()
        ax.annotate(nid, (x, y), xytext=(-4, 8), textcoords="offset points", fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Corredor arterial de 7 intersecciones semaforizadas (corridor7)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)


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
        print(f"{'Dem':4} {'idénticas':9} {'cargados':>8} {'insert.':>8} {'teleports':>9} {'%tel':>6} "
              f"{'espera(s)':>9} {'viaje(s)':>8} {'en red al final':>15}")
        for d in demands:
            res = [futs[(d, r)].result() for r in range(args.runs)]
            same = len({r["hash"] for r in res}) == 1
            r0 = res[0]
            pct = 100 * r0["teleports"] / max(r0["inserted"], 1)
            good = same and pct < 1.0
            ok &= good
            print(f"{d:4} {str(same):9} {r0['loaded']:8d} {r0['inserted']:8d} {r0['teleports']:9d} {pct:6.2f} "
                  f"{r0['mean_wait']:9.1f} {r0['mean_travel']:8.1f} {r0['running_end'] + r0['waiting_end']:15d}"
                  f"  {'OK' if good else 'FALLA'}")

    fig = RESULTS_DIR / "figures" / "corridor7_network.png"
    plot_network(fig)
    print(f"\nFigura de la red: {fig}")
    print("Criterio de salida Etapa 1:", "CUMPLIDO" if ok else "NO CUMPLIDO")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
