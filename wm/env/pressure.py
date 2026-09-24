"""Presión por fase verde (para Max-Pressure): Σ vehículos en carriles de entrada − Σ en carriles de salida
de los movimientos que la fase habilita."""
from __future__ import annotations

import numpy as np

N_GREENS = 4


class PressureSensor:
    def __init__(self, sim, tls_ids: list[str], program: str = "fixed"):
        self.sim = sim
        # phase_links[i][k] = pares (carril de entrada, carril de salida) con verde en la fase k del semáforo i
        self.phase_links: list[list[list[tuple[str, str]]]] = []
        for tid in tls_ids:
            logic = next(lg for lg in sim.trafficlight.getAllProgramLogics(tid) if lg.programID == program)
            links = sim.trafficlight.getControlledLinks(tid)
            per_phase = []
            for k in range(N_GREENS):
                state = logic.phases[3 * k].state
                pairs = {(lk[0][0], lk[0][1]) for idx, lk in enumerate(links) if lk and state[idx] in "Gg"}
                per_phase.append(sorted(pairs))
            self.phase_links.append(per_phase)

    def pressures(self) -> np.ndarray:
        """Matriz [n_tls, 4] con la presión de cada fase verde."""
        cache: dict[str, int] = {}

        def count(lane: str) -> int:
            if lane not in cache:
                cache[lane] = self.sim.lane.getLastStepVehicleNumber(lane)
            return cache[lane]

        out = np.zeros((len(self.phase_links), N_GREENS), dtype=np.float32)
        for i, per_phase in enumerate(self.phase_links):
            for k, pairs in enumerate(per_phase):
                out[i, k] = sum(count(a) - count(b) for a, b in pairs)
        return out
