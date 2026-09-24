"""Reglas conocidas del sistema que el entorno imaginado aplica sobre las predicciones del modelo.

* `effective_action`: replica la lógica del TrafficEnvironment (una petición de cambio solo es efectiva con
  verde activo y al menos `min_green` s de verde; con `max_green` el cambio se fuerza). El modelo aprendió la
  dinámica con acciones efectivas, así que en la imaginación también se le entregan acciones efectivas.
* `project_state`: lleva el estado imaginado a rangos físicos válidos (no negativos, fase one-hot, indicador
  binario), lo que evita que el error compuesto produzca estados imposibles.
Ambas operan en unidades originales, con estados [..., n_tls, n_features].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wm.env.state_builder import FEATURE_NAMES
from wm.utils import load_config

F = {n: i for i, n in enumerate(FEATURE_NAMES)}
PHASE = slice(F["phase_NS_straight"], F["phase_EW_left"] + 1)


@dataclass(frozen=True)
class SignalRules:
    min_green: float
    max_green: float
    delta_t: int

    @classmethod
    def from_config(cls) -> "SignalRules":
        t, b = load_config("scenario")["tls"], load_config("base")["env"]
        return cls(t["min_green"], t["max_green"], b["delta_t"])

    def effective_action(self, raw_state: np.ndarray, requested: np.ndarray) -> np.ndarray:
        """raw_state [..., n_tls, F], requested [..., n_tls] -> acción efectiva {0,1}."""
        green = raw_state[..., F["yellow"]] < 0.5
        t = raw_state[..., F["time_in_phase"]]
        asked = (requested > 0) & (t >= self.min_green)
        forced = t >= self.max_green - self.delta_t + 1   # el verde máximo se alcanzaría durante el paso
        return (green & (asked | forced)).astype(np.float32)

    def project_state(self, raw_state: np.ndarray) -> np.ndarray:
        s = np.maximum(raw_state, 0.0)
        s[..., F["halting"]] = np.minimum(s[..., F["halting"]], s[..., F["vehicles"]])
        s[..., F["occupancy"]] = np.minimum(s[..., F["occupancy"]], 100.0)
        s[..., F["time_in_phase"]] = np.minimum(s[..., F["time_in_phase"]], self.max_green)
        phase = s[..., PHASE].argmax(-1)
        s[..., PHASE] = np.eye(PHASE.stop - PHASE.start, dtype=s.dtype)[phase]
        s[..., F["yellow"]] = (s[..., F["yellow"]] >= 0.5).astype(s.dtype)
        return s
