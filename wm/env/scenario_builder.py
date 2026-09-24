"""Etapa 1 — genera el escenario SUMO del corredor de 7 intersecciones.

Produce en sumo/:
  corridor7.nod.xml / .edg.xml   fuentes de la red
  corridor7.net.xml              red compilada con netconvert
  corridor7.tls.add.xml          programas semafóricos 'fixed' (90 s) y 'actuated'
  routes_D1..D5.rou.xml          demanda por escenario
  corridor7.sumocfg              configuración base (la ruta se elige al lanzar SUMO)

Uso:  python -m wm.env.scenario_builder
"""
from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import sumolib

from wm.utils import SUMO_DIR, load_config, sumo_binary

NAME = "corridor7"
AXIS_NS, AXIS_EW = "NS", "EW"
# Índices de las fases verdes dentro del programa (cada verde va seguido de amarillo y todo rojo).
GREEN_PHASES = ("NS_straight", "NS_left", "EW_straight", "EW_left")


def _write_xml(root: ET.Element, path: Path) -> None:
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


# ---------------------------------------------------------------- red
def build_network(cfg: dict) -> Path:
    net = cfg["network"]
    n, dx, att = net["n_intersections"], net["spacing"], net["attach_length"]

    nodes = ET.Element("nodes")
    xs = [i * dx for i in range(n)]
    for i, x in enumerate(xs):
        ET.SubElement(nodes, "node", id=f"J{i}", x=str(x), y="0", type="traffic_light", tl=f"J{i}")
        ET.SubElement(nodes, "node", id=f"N{i}", x=str(x), y=str(att), type="priority")
        ET.SubElement(nodes, "node", id=f"S{i}", x=str(x), y=str(-att), type="priority")
    ET.SubElement(nodes, "node", id="W", x=str(-att), y="0", type="priority")
    ET.SubElement(nodes, "node", id="E", x=str(xs[-1] + att), y="0", type="priority")

    edges = ET.Element("edges")
    cons = ET.Element("connections")
    pocket = net["left_turn_pocket"]
    lanes = {}  # id de edge -> nº de carriles

    def edge(a: str, b: str, kind: str) -> None:
        spec = net[kind]
        eid = f"{a}_{b}"
        e = ET.SubElement(edges, "edge", id=eid, attrib={"from": a, "to": b}, numLanes=str(spec["lanes"]),
                          speed=str(spec["speed"]), priority="2" if kind == "arterial" else "1")
        lanes[eid] = spec["lanes"]
        if b.startswith("J"):  # acceso a un semáforo: los últimos metros tienen carril(es) extra de giro izq.
            k = spec["lanes"] + pocket["lanes"]
            ET.SubElement(e, "split", pos=str(-pocket["length"]), lanes=" ".join(map(str, range(k))),
                          idBefore=eid, idAfter=f"{eid}_in")
            lanes[f"{eid}_in"] = k

    chain = ["W"] + [f"J{i}" for i in range(n)] + ["E"]
    for a, b in zip(chain, chain[1:]):
        edge(a, b, "arterial")
        edge(b, a, "arterial")
    for i in range(n):
        for side in ("N", "S"):
            edge(f"{side}{i}", f"J{i}", "cross")
            edge(f"J{i}", f"{side}{i}", "cross")

    # Conexiones explícitas en cada cruce: carriles de paso -> recto (el 0 también a la derecha),
    # carril(es) de bolsillo -> solo izquierda.
    pos = {f"J{i}": (xs[i], 0.0) for i in range(n)}
    pos.update({f"N{i}": (xs[i], att) for i in range(n)})
    pos.update({f"S{i}": (xs[i], -att) for i in range(n)})
    pos.update({"W": (-att, 0.0), "E": (xs[-1] + att, 0.0)})
    for eid in [e for e in lanes if e.endswith("_in")]:
        a, j = eid[:-3].split("_")
        ax, ay = pos[a]
        jx, jy = pos[j]
        heading = (jx - ax, jy - ay)
        n_through = lanes[eid[:-3]]
        for out in [e for e in lanes if e.startswith(f"{j}_") and not e.endswith("_in")]:
            b = out.split("_")[1]
            if b == a:
                continue
            bx, by = pos[b]
            cross = heading[0] * (by - jy) - heading[1] * (bx - jx)  # >0: giro a la izquierda
            n_out = lanes[out]
            if abs(cross) < 1e-6:  # recto
                for ln in range(n_through):
                    ET.SubElement(cons, "connection", attrib={"from": eid, "to": out, "fromLane": str(ln),
                                                              "toLane": str(min(ln, n_out - 1))})
            elif cross > 0:  # izquierda
                for p in range(pocket["lanes"]):
                    ET.SubElement(cons, "connection", attrib={"from": eid, "to": out, "fromLane": str(n_through + p),
                                                              "toLane": str(n_out - 1)})
            else:  # derecha
                ET.SubElement(cons, "connection", attrib={"from": eid, "to": out, "fromLane": "0", "toLane": "0"})

    nod, edg, con, out = (SUMO_DIR / f"{NAME}.{s}" for s in ("nod.xml", "edg.xml", "con.xml", "net.xml"))
    _write_xml(nodes, nod)
    _write_xml(edges, edg)
    _write_xml(cons, con)
    subprocess.run(
        [sumo_binary("netconvert"), "-n", str(nod), "-e", str(edg), "-x", str(con), "-o", str(out),
         "--no-turnarounds", "true", "--tls.default-type", "static", "--no-warnings", "true"],
        check=True,
    )
    return out


# ---------------------------------------------------------------- semáforos
def _axis(net: sumolib.net.Net, edge) -> str:
    """Eje del acceso según la geometría: NS si el nodo de origen está arriba/abajo del cruce."""
    a, b = edge.getFromNode().getCoord(), edge.getToNode().getCoord()
    return AXIS_NS if abs(a[1] - b[1]) > abs(a[0] - b[0]) else AXIS_EW


def green_states(net: sumolib.net.Net, tls_id: str) -> list[str]:
    """Cadenas de estado de las 4 fases verdes para un semáforo, a partir de sus conexiones."""
    tls = net.getTLS(tls_id)
    n_links = max(tls.getLinks()) + 1
    states = [["r"] * n_links for _ in GREEN_PHASES]
    for idx, conns in tls.getLinks().items():
        for in_lane, out_lane, _ in conns:
            axis = _axis(net, in_lane.getEdge())
            conn = [c for c in in_lane.getOutgoing() if c.getToLane() == out_lane][0]
            d = conn.getDirection().lower()  # s, l, r (t no existe: --no-turnarounds)
            base = 0 if axis == AXIS_NS else 2
            if d in ("s", "r"):
                states[base][idx] = "G"
            elif d == "l":
                states[base + 1][idx] = "G"
    return ["".join(s) for s in states]


def build_tls(cfg: dict, net_file: Path) -> Path:
    t = cfg["tls"]
    net = sumolib.net.readNet(str(net_file), withPrograms=False)
    add = ET.Element("additional")
    for i in range(cfg["network"]["n_intersections"]):
        tls_id = f"J{i}"
        greens = green_states(net, tls_id)
        # SUMO activa el último programa cargado: 'fixed' queda activo por defecto.
        for prog, ptype in (("actuated", "actuated"), ("fixed", "static")):
            logic = ET.SubElement(add, "tlLogic", id=tls_id, type=ptype, programID=prog, offset="0")
            if ptype == "actuated":
                ET.SubElement(logic, "param", key="detector-gap", value="2.0")
                ET.SubElement(logic, "param", key="max-gap", value="3.0")
            for k, g in enumerate(greens):
                attrs = {"duration": str(t["fixed_time_greens"][k]), "state": g, "name": GREEN_PHASES[k]}
                if ptype == "actuated":
                    attrs.update(minDur=str(t["min_green"]), maxDur=str(t["max_green"]))
                ET.SubElement(logic, "phase", attrs)
                ET.SubElement(logic, "phase", duration=str(t["yellow"]), state=g.replace("G", "y"),
                              name=f"{GREEN_PHASES[k]}_yellow")
                ET.SubElement(logic, "phase", duration=str(t["all_red"]), state="r" * len(g),
                              name=f"{GREEN_PHASES[k]}_allred")
    out = SUMO_DIR / f"{NAME}.tls.add.xml"
    _write_xml(add, out)
    return out


# ---------------------------------------------------------------- demanda
def enumerate_routes(net: sumolib.net.Net, entry: str, turning: dict) -> list[tuple[list[str], float]]:
    """Rutas desde una entrada con giros markovianos (recto/izq./der.) en cada intersección."""
    probs = {"s": turning["straight"], "l": turning["left"], "r": turning["right"]}
    routes: list[tuple[list[str], float]] = []

    def walk(path: list[str], p: float) -> None:
        edge = net.getEdge(path[-1])
        options: dict[str, str] = {}
        for out_edge, conns in edge.getOutgoing().items():
            options[conns[0].getDirection().lower()] = out_edge.getID()
        if not options:  # nodo frontera: fin de la ruta
            routes.append((path, p))
            return
        if edge.getToNode().getType() != "traffic_light":  # nodo intermedio del bolsillo de giro
            options = {"s": next(iter(options.values()))}
        total = sum(probs[d] for d in options)
        for d, nxt in options.items():
            walk(path + [nxt], p * probs[d] / total)

    walk([entry], 1.0)
    keep = [(r, p) for r, p in routes if p >= turning["min_route_prob"]]
    z = sum(p for _, p in keep)
    return [(r, p / z) for r, p in keep]


def rate_profile(spec, shape: str, begin: float, warmup: float, end: float, seg: float) -> list[tuple[float, float, float]]:
    """Lista de segmentos (t0, t1, veh/h) de demanda constante a trozos."""
    out = []
    t = begin
    while t < end:
        t1 = min(t + seg, end)
        if shape == "constant":
            rate = float(spec)
        else:  # peak: triangular sobre el periodo registrado, base durante el calentamiento
            base, peak = spec
            mid, half = warmup + (end - warmup) / 2, (end - warmup) / 2
            tm = (t + t1) / 2
            rate = base if tm < warmup else base + (peak - base) * max(0.0, 1 - abs(tm - mid) / half)
        out.append((t, t1, rate))
        t = t1
    return out


def build_routes(cfg: dict, net_file: Path) -> list[Path]:
    net = sumolib.net.readNet(str(net_file))
    ep, v, tr = cfg["episode"], cfg["vehicle"], cfg["turning"]
    begin, warmup = ep["begin"], ep["warmup"]
    end = warmup + ep["recorded"]
    n = cfg["network"]["n_intersections"]

    entries = {"W_J0": ("arterial", "W"), f"E_J{n - 1}": ("arterial", "E")}
    for i in range(n):
        entries[f"N{i}_J{i}"] = ("cross", None)
        entries[f"S{i}_J{i}"] = ("cross", None)
    routes = {e: enumerate_routes(net, e, tr) for e in entries}

    outs = []
    for dname, d in cfg["demands"].items():
        root = ET.Element("routes")
        ET.SubElement(root, "vType", id="car", vClass=v["vclass"], emissionClass=v["emission_class"], sigma=str(v["sigma"]))
        for e, rs in routes.items():
            for k, (edges, _) in enumerate(rs):
                ET.SubElement(root, "route", id=f"{e}__{k}", edges=" ".join(edges))
        flows = []
        for e, (kind, side) in entries.items():
            spec = d["arterial"][side] if kind == "arterial" else d["cross"]
            for t0, t1, rate in rate_profile(spec, d["shape"], begin, warmup, end, cfg["demand_segment"]):
                for k, (_, p) in enumerate(routes[e]):
                    lam = rate * p / 3600.0  # veh/s en esta ruta
                    if lam > 0:
                        flows.append((t0, f"{e}__{k}__{int(t0)}", f"{e}__{k}", t0, t1, lam))
        for _, fid, rid, t0, t1, lam in sorted(flows):  # SUMO exige orden por 'begin'
            ET.SubElement(root, "flow", id=fid, type="car", route=rid, begin=str(t0), end=str(t1),
                          period=f"exp({lam:.8f})", departLane="best", departSpeed="max")
        out = SUMO_DIR / f"routes_{dname}.rou.xml"
        _write_xml(root, out)
        outs.append(out)
    return outs


def build_sumocfg(cfg: dict) -> Path:
    ep = cfg["episode"]
    root = ET.Element("configuration")
    inp = ET.SubElement(root, "input")
    ET.SubElement(inp, "net-file", value=f"{NAME}.net.xml")
    ET.SubElement(inp, "additional-files", value=f"{NAME}.tls.add.xml")
    tm = ET.SubElement(root, "time")
    ET.SubElement(tm, "begin", value=str(ep["begin"]))
    ET.SubElement(tm, "end", value=str(ep["warmup"] + ep["recorded"]))
    proc = ET.SubElement(root, "processing")
    ET.SubElement(proc, "time-to-teleport", value=str(ep["time_to_teleport"]))
    rep = ET.SubElement(root, "report")
    ET.SubElement(rep, "no-step-log", value="true")
    ET.SubElement(rep, "no-warnings", value="true")
    out = SUMO_DIR / f"{NAME}.sumocfg"
    _write_xml(root, out)
    return out


def main() -> None:
    cfg = load_config("scenario")
    SUMO_DIR.mkdir(exist_ok=True)
    net_file = build_network(cfg)
    build_tls(cfg, net_file)
    build_routes(cfg, net_file)
    build_sumocfg(cfg)
    net = sumolib.net.readNet(str(net_file))
    for i in range(cfg["network"]["n_intersections"]):
        print(f"J{i}: {green_states(net, f'J{i}')}")
    print("Escenario generado en", SUMO_DIR)


if __name__ == "__main__":
    main()
