"""Presión por fase verde (para Max-Pressure).

Para cada movimiento (carril de entrada -> carril de salida) que la fase habilita:
    presión = cola en la entrada − cola en la salida
donde
  * cola en la entrada = vehículos detenidos en el carril de la línea de pare (bolsillo `_in`) más los del
    carril correspondiente del tramo aguas arriba (la cola se extiende más allá de los 60 m del bolsillo);
  * cola en la salida = vehículos detenidos en el carril de salida (lo que realmente bloquea el avance;
    los vehículos que circulan libremente aguas abajo no restan presión).
La presión de la fase es la suma sobre sus movimientos (cada carril se cuenta una vez por movimiento).
"""
from __future__ import annotations

import numpy as np

N_GREENS = 4
POCKET_SUFFIX = "_in"


def upstream_lane(sim, lane: str) -> str | None:
    """Carril del tramo aguas arriba con el mismo índice (el bolsillo de giro no tiene equivalente)."""
    edge, idx = lane.rsplit("_", 1)
    if not edge.endswith(POCKET_SUFFIX):
        return None
    up = f"{edge[:-len(POCKET_SUFFIX)]}_{idx}"
    try:
        sim.lane.getLength(up)
    except Exception:  # noqa: BLE001 — el carril no existe (bolsillo)
        return None
    return up


class PressureSensor:
    def __init__(self, sim, tls_ids: list[str], program: str = "fixed"):
        self.sim = sim
        # phase_links[i][k] = [(carriles de entrada, carril de salida)] con verde en la fase k del semáforo i
        self.phase_links: list[list[list[tuple[tuple[str, ...], str]]]] = []
        for tid in tls_ids:
            logic = next(lg for lg in sim.trafficlight.getAllProgramLogics(tid) if lg.programID == program)
            links = sim.trafficlight.getControlledLinks(tid)
            per_phase = []
            for k in range(N_GREENS):
                state = logic.phases[3 * k].state
                pairs = sorted({(lk[0][0], lk[0][1]) for idx, lk in enumerate(links) if lk and state[idx] in "Gg"})
                per_phase.append([((a,) + ((up,) if (up := upstream_lane(sim, a)) else ()), b) for a, b in pairs])
            self.phase_links.append(per_phase)

    def pressures(self) -> np.ndarray:
        """Matriz [n_tls, 4] con la presión de cada fase verde."""
        cache: dict[str, int] = {}

        def queue(lane: str) -> int:
            if lane not in cache:
                cache[lane] = self.sim.lane.getLastStepHaltingNumber(lane)
            return cache[lane]

        out = np.zeros((len(self.phase_links), N_GREENS), dtype=np.float32)
        for i, per_phase in enumerate(self.phase_links):
            for k, movements in enumerate(per_phase):
                out[i, k] = sum(sum(queue(a) for a in ins) - queue(b) for ins, b in movements)
        return out
