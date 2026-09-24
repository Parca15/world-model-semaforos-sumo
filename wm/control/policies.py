"""Políticas heurísticas: se usan para recolectar `base_v1` (P1–P4) y como baselines de control.

Todas comparten la interfaz `Policy`:
  control   modo del TrafficEnvironment que requieren ('agent', 'fixed' o 'actuated');
  reset()   se llama después de env.reset() (con un generador aleatorio propio y reproducible);
  act()     devuelve la acción MultiDiscrete([2]*7) para la observación actual.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from wm.env.pressure import PressureSensor
from wm.env.traffic_env import GREEN, N_GREENS, TrafficEnvironment


class Policy(ABC):
    control: str = "agent"

    def reset(self, env: TrafficEnvironment, rng: np.random.Generator) -> None:
        self.rng = rng
        self.n = len(env.tls_ids)

    @abstractmethod
    def act(self, obs: np.ndarray, env: TrafficEnvironment) -> np.ndarray: ...


class FixedTime(Policy):
    """P1 / Condición 1: SUMO ejecuta el programa de tiempo fijo (ciclo 90 s)."""
    control = "fixed"

    def act(self, obs, env):
        return np.zeros(self.n, dtype=np.int64)


class Actuated(Policy):
    """P3: SUMO ejecuta el control actuado por detectores."""
    control = "actuated"

    def act(self, obs, env):
        return np.zeros(self.n, dtype=np.int64)


class RandomRestricted(Policy):
    """P2: pide cambiar de fase con probabilidad p; el entorno impone los verdes mínimo y máximo."""

    def __init__(self, p: float):
        assert 0.0 <= p <= 1.0
        self.p = p

    def act(self, obs, env):
        return (self.rng.random(self.n) < self.p).astype(np.int64)


class MaxPressure(Policy):
    """P4: Max-Pressure cíclico. Como la acción solo permite avanzar a la fase siguiente, se cambia cuando
    la fase actual no es la de mayor presión. Con probabilidad epsilon la acción es aleatoria."""

    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon

    def reset(self, env, rng):
        super().reset(env, rng)
        self.sensor = PressureSensor(env.sim, env.tls_ids)

    def act(self, obs, env):
        pressure = self.sensor.pressures()
        action = np.zeros(self.n, dtype=np.int64)
        for i, ph in enumerate(env.phase_states()):
            if ph.mode == GREEN:
                others = [pressure[i, k] for k in range(N_GREENS) if k != ph.green]
                action[i] = int(pressure[i, ph.green] < max(others))
        explore = self.rng.random(self.n) < self.epsilon
        action[explore] = self.rng.integers(0, 2, explore.sum())
        return action


def make_policy(kind: str, **params) -> Policy:
    registry = {
        "fixed_time": FixedTime,
        "actuated": Actuated,
        "random_restricted": RandomRestricted,
        "max_pressure": MaxPressure,
    }
    if kind not in registry:
        raise ValueError(f"Política desconocida '{kind}'. Opciones: {sorted(registry)}")
    return registry[kind](**params)
