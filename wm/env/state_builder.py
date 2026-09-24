"""CustomStateBuilder — construye el estado s_t ∈ R^{7×13} a partir de SUMO.

Variables por intersección (sobre los carriles de entrada; cada acceso = tramo aguas arriba + bolsillo `_in`):
   0 vehicles        vehículos presentes                          veh
   1 halting         vehículos detenidos (v < 0,1 m/s)            veh
   2 queue_m         longitud de cola (detenidos × 7,5 m)         m
   3 mean_speed      velocidad promedio de los vehículos          m/s   (sin vehículos: velocidad límite)
   4 occupancy       ocupación promedio de los carriles           %
   5 waiting_acc     tiempo de espera acumulado de los vehículos  s
   6 outflow         vehículos que cruzaron la línea de pare en Δt veh
   7-10 phase_*      fase verde activa (one-hot, 4 fases)          —
  11 time_in_phase   tiempo en el verde actual o en la transición s
  12 yellow          transición (amarillo o todo rojo) activa     {0,1}
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import sumolib

FEATURES = [
    # nombre,          unidad,  se normaliza
    ("vehicles", "veh", True),
    ("halting", "veh", True),
    ("queue_m", "m", True),
    ("mean_speed", "m/s", True),
    ("occupancy", "%", True),
    ("waiting_acc", "s", True),
    ("outflow", "veh", True),
    ("phase_NS_straight", "-", False),
    ("phase_NS_left", "-", False),
    ("phase_EW_straight", "-", False),
    ("phase_EW_left", "-", False),
    ("time_in_phase", "s", True),
    ("yellow", "-", False),
]
FEATURE_NAMES = [f[0] for f in FEATURES]
N_FEATURES = len(FEATURES)
RAW_METRICS = ["queue_m", "halting", "outflow", "waiting_acc"]  # guardadas en bruto por intersección
VEH_SPACE = 7.5  # m que ocupa un vehículo detenido (longitud 5 m + minGap 2,5 m)


@dataclass
class Junction:
    tls_id: str
    lanes: list[str]              # todos los carriles de entrada (tramo aguas arriba + bolsillo)
    stopline_lanes: list[str]     # carriles de los bolsillos `_in` (los que cruzan la línea de pare)
    speed_limit: float
    prev_ids: set[str] = field(default_factory=set)
    outflow: int = 0


class CustomStateBuilder:
    def __init__(self, net_file: str, tls_ids: list[str]):
        net = sumolib.net.readNet(net_file)
        self.junctions: list[Junction] = []
        for tid in tls_ids:
            node = net.getNode(tid)
            lanes, stop = [], []
            for e in node.getIncoming():
                stop += [ln.getID() for ln in e.getLanes()]
                lanes += [ln.getID() for ln in e.getLanes()]
                for up in e.getFromNode().getIncoming():  # tramo aguas arriba del bolsillo
                    if up.getFromNode().getID() != tid:
                        lanes += [ln.getID() for ln in up.getLanes()]
            limit = max(net.getLane(ln).getSpeed() for ln in lanes)
            self.junctions.append(Junction(tid, lanes, stop, limit))

    # --- flujo de salida: se actualiza cada segundo de simulación
    def reset_counters(self, sim) -> None:
        for j in self.junctions:
            j.prev_ids = self._ids_at_stopline(sim, j)
            j.outflow = 0

    def track_outflow(self, sim) -> None:
        for j in self.junctions:
            now = self._ids_at_stopline(sim, j)
            j.outflow += len(j.prev_ids - now)
            j.prev_ids = now

    @staticmethod
    def _ids_at_stopline(sim, j: Junction) -> set[str]:
        ids: set[str] = set()
        for ln in j.stopline_lanes:
            ids.update(sim.lane.getLastStepVehicleIDs(ln))
        return ids

    # --- observación en el instante de decisión
    def build(self, sim, phase_info: list[tuple[int, float, bool]]) -> np.ndarray:
        """phase_info[i] = (fase verde k, tiempo en fase, transición activa) de cada semáforo."""
        s = np.zeros((len(self.junctions), N_FEATURES), dtype=np.float32)
        for i, j in enumerate(self.junctions):
            n_veh = halt = 0
            occ = 0.0
            speed_sum = 0.0
            wait = 0.0
            for ln in j.lanes:
                ids = sim.lane.getLastStepVehicleIDs(ln)
                n_veh += len(ids)
                halt += sim.lane.getLastStepHaltingNumber(ln)
                occ += sim.lane.getLastStepOccupancy(ln)
                for v in ids:
                    speed_sum += sim.vehicle.getSpeed(v)
                    wait += sim.vehicle.getAccumulatedWaitingTime(v)
            k, t_phase, yellow = phase_info[i]
            s[i, 0] = n_veh
            s[i, 1] = halt
            s[i, 2] = halt * VEH_SPACE
            s[i, 3] = speed_sum / n_veh if n_veh else j.speed_limit
            s[i, 4] = 100.0 * occ / len(j.lanes)
            s[i, 5] = wait
            s[i, 6] = j.outflow
            s[i, 7 + k] = 1.0
            s[i, 11] = t_phase
            s[i, 12] = float(yellow)
            j.outflow = 0
        return s

    @staticmethod
    def raw_metrics(state: np.ndarray) -> np.ndarray:
        idx = [FEATURE_NAMES.index(m) for m in RAW_METRICS]
        return state[:, idx].copy()
