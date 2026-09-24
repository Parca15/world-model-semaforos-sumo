"""Etapa 1 — genera el escenario SUMO: malla urbana irregular de 7 intersecciones (configs/scenario.yaml).

Produce en sumo/ (con NAME = network.name):
  NAME.nod.xml / .edg.xml / .con.xml   fuentes de la red
  NAME.net.xml                         red compilada con netconvert
  NAME.tls.add.xml                     programas semafóricos 'fixed' (90 s) y 'actuated'
  routes_D1..D5.rou.xml                demanda por escenario
  NAME.sumocfg                         configuración base (la ruta se elige al lanzar SUMO)

Los movimientos (recto / izquierda / derecha) se definen con la geometría de cada cruce: los 4 brazos se
emparejan en dos ejes de brazos opuestos; ir al brazo opuesto es "recto" y los otros dos son izquierda o
derecha según el lado. El mismo mapa de movimientos genera las conexiones, los semáforos y las rutas.

Uso:  python -m wm.env.scenario_builder
"""
from __future__ import annotations

import itertools
import math
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import sumolib

from wm.utils import SUMO_DIR, load_config, sumo_binary

AXIS_NS, AXIS_EW = "NS", "EW"
GREEN_PHASES = ("NS_straight", "NS_left", "EW_straight", "EW_left")


def net_name(cfg: dict | None = None) -> str:
    return (cfg or load_config("scenario"))["network"]["name"]


def _write_xml(root: ET.Element, path: Path) -> None:
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


# ---------------------------------------------------------------- geometría de los cruces
class Topology:
    """Nodos, vías y movimientos de la red, calculados solo a partir de la configuración."""

    def __init__(self, cfg: dict):
        net = cfg["network"]
        self.pos = {k: tuple(v) for k, v in {**net["tls_nodes"], **net["fringe_nodes"]}.items()}
        self.tls = list(net["tls_nodes"])
        self.road_class: dict[tuple[str, str], str] = {}
        self.neighbors: dict[str, list[str]] = {n: [] for n in self.pos}
        for a, b, cls in net["links"]:
            self.road_class[(a, b)] = self.road_class[(b, a)] = cls
            self.neighbors[a].append(b)
            self.neighbors[b].append(a)
        self.axes = {j: self._pair_arms(j) for j in self.tls}

    def _angle(self, j: str, arm: str) -> float:
        (x0, y0), (x1, y1) = self.pos[j], self.pos[arm]
        return math.atan2(y1 - y0, x1 - x0)

    def _pair_arms(self, j: str) -> dict[str, str]:
        """Empareja los 4 brazos en dos ejes (los más opuestos entre sí) y los etiqueta NS/EW."""
        arms = self.neighbors[j]
        assert len(arms) == 4, f"{j} debe tener 4 brazos, tiene {arms}"

        def opposition(p, q):  # 0 si son exactamente opuestos
            d = abs(self._angle(j, p) - self._angle(j, q)) % (2 * math.pi)
            return abs(math.pi - d)

        a = arms[0]
        best = min(arms[1:], key=lambda b: opposition(a, b) + opposition(*[x for x in arms if x not in (a, b)]))
        pair1, pair2 = (a, best), tuple(x for x in arms if x not in (a, best))

        def horizontality(pair):
            return sum(abs(math.cos(self._angle(j, x))) for x in pair)

        ew, ns = (pair1, pair2) if horizontality(pair1) >= horizontality(pair2) else (pair2, pair1)
        return {x: AXIS_EW for x in ew} | {x: AXIS_NS for x in ns}

    def opposite(self, j: str, arm: str) -> str:
        return next(x for x in self.neighbors[j] if x != arm and self.axes[j][x] == self.axes[j][arm])

    def turn(self, j: str, came_from: str, go_to: str) -> str:
        """'s', 'l' o 'r' para el movimiento came_from -> j -> go_to."""
        if go_to == self.opposite(j, came_from):
            return "s"
        (ax, ay), (jx, jy), (bx, by) = self.pos[came_from], self.pos[j], self.pos[go_to]
        cross = (jx - ax) * (by - jy) - (jy - ay) * (bx - jx)
        return "l" if cross > 0 else "r"

    def movements(self, j: str) -> dict[tuple[str, str], str]:
        mv = {}
        for a in self.neighbors[j]:
            for b in self.neighbors[j]:
                if a != b:
                    mv[(a, b)] = self.turn(j, a, b)
            assert sorted(mv[(a, b)] for b in self.neighbors[j] if b != a) == ["l", "r", "s"], (j, a)
        return mv

    def is_tls(self, n: str) -> bool:
        return n in self.tls


def in_edge(a: str, j: str, topo: Topology) -> str:
    """Edge que llega a la línea de pare de j desde a (el bolsillo `_in` si j es semáforo)."""
    return f"{a}_{j}_in" if topo.is_tls(j) else f"{a}_{j}"


# ---------------------------------------------------------------- red
def build_network(cfg: dict, topo: Topology) -> Path:
    net = cfg["network"]
    classes, pocket = net["road_classes"], net["left_turn_pocket"]
    name = net["name"]

    nodes = ET.Element("nodes")
    for n, (x, y) in topo.pos.items():
        attrs = {"id": n, "x": str(x), "y": str(y)}
        attrs.update(type="traffic_light", tl=n) if topo.is_tls(n) else attrs.update(type="priority")
        ET.SubElement(nodes, "node", attrs)

    edges = ET.Element("edges")
    lanes: dict[str, int] = {}
    for a, b, cls in net["links"]:
        spec = classes[cls]
        for u, v in ((a, b), (b, a)):
            eid = f"{u}_{v}"
            e = ET.SubElement(edges, "edge", id=eid, attrib={"from": u, "to": v}, numLanes=str(spec["lanes"]),
                              speed=str(spec["speed"]), priority=str(spec["priority"]), type=cls)
            lanes[eid] = spec["lanes"]
            if topo.is_tls(v):  # acceso a semáforo: los últimos metros tienen carril(es) extra de giro izq.
                k = spec["lanes"] + pocket["lanes"]
                ET.SubElement(e, "split", pos=str(-pocket["length"]), lanes=" ".join(map(str, range(k))),
                              idBefore=eid, idAfter=f"{eid}_in")
                lanes[f"{eid}_in"] = k

    # Conexiones explícitas: carriles de paso -> recto (el 0 también a la derecha); bolsillo -> solo izquierda.
    cons = ET.Element("connections")
    for j in topo.tls:
        for (a, b), d in topo.movements(j).items():
            src, dst = f"{a}_{j}_in", f"{j}_{b}"
            n_through, n_out = lanes[f"{a}_{j}"], lanes[dst]
            if d == "s":
                pairs = [(ln, min(ln, n_out - 1)) for ln in range(n_through)]
            elif d == "l":
                pairs = [(n_through + p, n_out - 1) for p in range(pocket["lanes"])]
            else:
                pairs = [(0, 0)]
            for fl, tl in pairs:
                ET.SubElement(cons, "connection", attrib={"from": src, "to": dst,
                                                          "fromLane": str(fl), "toLane": str(tl)})

    nod, edg, con, out = (SUMO_DIR / f"{name}.{s}" for s in ("nod.xml", "edg.xml", "con.xml", "net.xml"))
    _write_xml(nodes, nod)
    _write_xml(edges, edg)
    _write_xml(cons, con)
    subprocess.run(
        [sumo_binary("netconvert"), "-n", str(nod), "-e", str(edg), "-x", str(con), "-o", str(out),
         "--no-turnarounds", "true", "--tls.default-type", "static", "--junctions.corner-detail", "5",
         "--no-warnings", "true"],
        check=True,
    )
    return out


# ---------------------------------------------------------------- semáforos
def _node_of(edge_id: str, end: int) -> str:
    return edge_id.split("_")[end]


def green_states(net: sumolib.net.Net, topo: Topology, tls_id: str) -> list[str]:
    """Cadenas de estado de las 4 fases verdes de un semáforo, a partir de sus conexiones."""
    tls = net.getTLS(tls_id)
    n_links = max(tls.getLinks()) + 1
    states = [["r"] * n_links for _ in GREEN_PHASES]
    for idx, conns in tls.getLinks().items():
        for in_lane, out_lane, _ in conns:
            a = _node_of(in_lane.getEdge().getID(), 0)
            b = _node_of(out_lane.getEdge().getID(), 1)
            d = topo.turn(tls_id, a, b)
            base = 0 if topo.axes[tls_id][a] == AXIS_NS else 2
            states[base + (1 if d == "l" else 0)][idx] = "G"
    return ["".join(s) for s in states]


def build_tls(cfg: dict, topo: Topology, net_file: Path) -> Path:
    t = cfg["tls"]
    net = sumolib.net.readNet(str(net_file), withPrograms=False)
    add = ET.Element("additional")
    for tls_id in topo.tls:
        greens = green_states(net, topo, tls_id)
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
    out = SUMO_DIR / f"{cfg['network']['name']}.tls.add.xml"
    _write_xml(add, out)
    return out


# ---------------------------------------------------------------- demanda
def enumerate_routes(topo: Topology, entry: str, turning: dict) -> list[tuple[list[str], float]]:
    """Rutas desde un nodo frontera con giros markovianos (recto/izq./der.) en cada intersección."""
    probs = {"s": turning["straight"], "l": turning["left"], "r": turning["right"]}
    routes: list[tuple[list[str], float]] = []
    first = topo.neighbors[entry][0]

    def walk(prev: str, j: str, edges: list[str], p: float) -> None:
        if p < turning["min_route_prob"] / 10:  # poda de ramas despreciables (bucles en la malla)
            return
        if not topo.is_tls(j):  # llegó a un nodo frontera: fin de la ruta
            routes.append((edges, p))
            return
        for b in topo.neighbors[j]:
            if b == prev:
                continue
            nxt = [f"{j}_{b}"] + ([f"{j}_{b}_in"] if topo.is_tls(b) else [])
            walk(j, b, edges + nxt, p * probs[topo.turn(j, prev, b)])

    walk(entry, first, [f"{entry}_{first}", f"{entry}_{first}_in"], 1.0)
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


def build_routes(cfg: dict, topo: Topology) -> list[Path]:
    ep, v, tr = cfg["episode"], cfg["vehicle"], cfg["turning"]
    begin, warmup = ep["begin"], ep["warmup"]
    end = warmup + ep["recorded"]
    arterial = set(cfg["network"]["arterial_entries"])
    entries = [n for n in cfg["network"]["fringe_nodes"]]
    routes = {e: enumerate_routes(topo, e, tr) for e in entries}

    outs = []
    for dname, d in cfg["demands"].items():
        root = ET.Element("routes")
        ET.SubElement(root, "vType", id="car", vClass=v["vclass"], emissionClass=v["emission_class"], sigma=str(v["sigma"]))
        for e, rs in routes.items():
            for k, (edges, _) in enumerate(rs):
                ET.SubElement(root, "route", id=f"{e}__{k}", edges=" ".join(edges))
        flows = []
        for e in entries:
            spec = d["arterial"][e] if e in arterial else d["cross"]
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
    ep, name = cfg["episode"], cfg["network"]["name"]
    root = ET.Element("configuration")
    inp = ET.SubElement(root, "input")
    ET.SubElement(inp, "net-file", value=f"{name}.net.xml")
    ET.SubElement(inp, "additional-files", value=f"{name}.tls.add.xml")
    tm = ET.SubElement(root, "time")
    ET.SubElement(tm, "begin", value=str(ep["begin"]))
    ET.SubElement(tm, "end", value=str(ep["warmup"] + ep["recorded"]))
    proc = ET.SubElement(root, "processing")
    ET.SubElement(proc, "time-to-teleport", value=str(ep["time_to_teleport"]))
    rep = ET.SubElement(root, "report")
    ET.SubElement(rep, "no-step-log", value="true")
    ET.SubElement(rep, "no-warnings", value="true")
    out = SUMO_DIR / f"{name}.sumocfg"
    _write_xml(root, out)
    return out


def main() -> None:
    cfg = load_config("scenario")
    SUMO_DIR.mkdir(exist_ok=True)
    for old in itertools.chain(SUMO_DIR.glob("*.xml"), SUMO_DIR.glob("*.sumocfg")):
        old.unlink()
    topo = Topology(cfg)
    net_file = build_network(cfg, topo)
    build_tls(cfg, topo, net_file)
    build_routes(cfg, topo)
    build_sumocfg(cfg)
    net = sumolib.net.readNet(str(net_file))
    for j in topo.tls:
        ew = [a for a, ax in topo.axes[j].items() if ax == AXIS_EW]
        ns = [a for a, ax in topo.axes[j].items() if ax == AXIS_NS]
        print(f"{j}: EO={ew} NS={ns} fases={green_states(net, topo, j)}")
    n_routes = {e: len(enumerate_routes(topo, e, cfg["turning"])) for e in cfg["network"]["fringe_nodes"]}
    print("rutas por entrada:", n_routes)
    print("Escenario generado en", SUMO_DIR)


if __name__ == "__main__":
    main()
