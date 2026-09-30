"""TrafficEnvironment — entorno gymnasium sobre SUMO para la malla urbana de 7 intersecciones.

* Paso de decisión Δt = 5 s -> 720 pasos por episodio (tras 300 s de calentamiento con tiempo fijo).
* Acción por intersección: 0 = mantener fase, 1 = pasar a la siguiente fase
  (amarillo 3 s + todo rojo 1 s automáticos; verde mínimo 10 s y máximo 60 s).
  Espacio conjunto MultiDiscrete([2]*7).
* Calentamiento: 300 s con el programa de tiempo fijo (con el actuado en modo 'actuated').
* Modos de control:
    'agent'    las acciones vienen de fuera (políticas P2/P4, PPO, ...);
    'fixed'    SUMO ejecuta el programa de tiempo fijo (ciclo 90 s) y la acción se ignora;
    'actuated' SUMO ejecuta el programa actuado y la acción se ignora.
  En todos los modos `info["action"]` es la acción *efectiva*: 1 si en el paso comenzó un cambio de fase.
* Recompensa por intersección r_t^i = −(W_{t+1}^i − W_t^i)/100, con W = espera acumulada en los accesos;
  la recompensa escalar es la suma y el vector va en `info["reward_vec"]`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from wm.env.state_builder import FEATURE_NAMES, N_FEATURES, CustomStateBuilder
from wm.utils import SUMO_DIR, load_config, sumo_binary, sumo_home

GREEN, YELLOW, ALLRED = 0, 1, 2  # posición dentro del bloque de 3 fases de cada verde en el programa
N_GREENS = 4
W_IDX = FEATURE_NAMES.index("waiting_acc")
_libsumo_in_use = False


@dataclass
class PhaseState:
    """Máquina de estados de un semáforo: verde k -> amarillo -> todo rojo -> verde k+1."""
    green: int = 0      # índice de la fase verde (0..3)
    mode: int = GREEN
    elapsed: float = 0  # s en el modo actual (en transición: desde que empezó el amarillo)

    @property
    def sumo_index(self) -> int:
        return 3 * self.green + self.mode

    def feature(self) -> tuple[int, float, bool]:
        return self.green, self.elapsed, self.mode != GREEN


class TrafficEnvironment(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, demand: str = "D2", control: str = "agent", backend: str = "libsumo",
                 episode_seconds: int | None = None, sumo_args: list[str] | None = None, gui: bool = False):
        """`gui=True` abre sumo-gui para ver la simulación en tiempo real (solo con TraCI: libsumo no tiene GUI)."""
        assert control in ("agent", "fixed", "actuated")
        base, scen = load_config("base"), load_config("scenario")
        self.tls_ids: list[str] = base["network"]["tls_ids"]
        self.dt: int = base["env"]["delta_t"]
        self.warmup: int = base["env"]["warmup"]
        self.episode_seconds: int = episode_seconds or base["env"]["episode_seconds"]
        self.max_steps = self.episode_seconds // self.dt
        t = scen["tls"]
        self.yellow, self.all_red = t["yellow"], t["all_red"]
        self.min_green, self.max_green = t["min_green"], t["max_green"]
        self.demand, self.control, self.gui = demand, control, gui
        self.backend = "traci" if gui else backend
        self.net_name: str = scen["network"]["name"]
        self.extra_args = sumo_args or []

        n = len(self.tls_ids)
        self.action_space = gym.spaces.MultiDiscrete([2] * n)
        self.observation_space = gym.spaces.Box(0.0, np.inf, shape=(n, N_FEATURES), dtype=np.float32)
        self.builder = CustomStateBuilder(str(SUMO_DIR / f"{self.net_name}.net.xml"), self.tls_ids)
        self.sim = None
        self.phases: list[PhaseState] = []

    # ------------------------------------------------------------ SUMO
    def _start(self, seed: int) -> None:
        global _libsumo_in_use
        sumo_home()
        cmd = [sumo_binary("sumo-gui" if self.gui else "sumo"), "-c", str(SUMO_DIR / f"{self.net_name}.sumocfg"),
               "-r", str(SUMO_DIR / f"routes_{self.demand}.rou.xml"),
               "--seed", str(seed), "--end", str(self.warmup + self.episode_seconds),
               "--waiting-time-memory", "100000", "--no-step-log", "true", "--no-warnings", "true",
               *self.extra_args]
        if self.backend == "libsumo":
            import libsumo as sim
            if _libsumo_in_use:  # libsumo admite una sola simulación por proceso
                sim.close()
            sim.start(cmd)
            _libsumo_in_use = True
        else:
            import traci as sim
            label = f"env{id(self)}"
            try:
                traci_conn = sim.getConnection(label)
                traci_conn.close()
            except Exception:  # noqa: BLE001
                pass
            sim.start(cmd, label=label)
            sim = sim.getConnection(label)
        self.sim = sim

    def close(self) -> None:
        global _libsumo_in_use
        if self.sim is not None:
            self.sim.close()
            if self.backend == "libsumo":
                _libsumo_in_use = False
            self.sim = None

    # ------------------------------------------------------------ fases
    def _read_phases(self) -> None:
        """Sincroniza la máquina de estados con lo que SUMO está ejecutando."""
        self.phases = []
        for tid in self.tls_ids:
            idx = self.sim.trafficlight.getPhase(tid)
            p = PhaseState(green=idx // 3, mode=idx % 3, elapsed=self.sim.trafficlight.getSpentDuration(tid))
            if p.mode == ALLRED:
                p.elapsed += self.yellow
            self.phases.append(p)

    def _freeze(self, i: int) -> None:
        tid = self.tls_ids[i]
        self.sim.trafficlight.setPhase(tid, self.phases[i].sumo_index)
        self.sim.trafficlight.setPhaseDuration(tid, 1e6)

    def _start_switch(self, i: int) -> None:
        p = self.phases[i]
        p.mode, p.elapsed = YELLOW, 0
        self._freeze(i)

    def _tick_agent(self) -> None:
        """Avanza 1 s las máquinas de estado en modo 'agent' (antes de simulationStep)."""
        for i, p in enumerate(self.phases):
            p.elapsed += 1
            if p.mode == YELLOW and p.elapsed >= self.yellow:
                p.mode = ALLRED
                self._freeze(i)
            elif p.mode == ALLRED and p.elapsed >= self.yellow + self.all_red:
                p.green, p.mode, p.elapsed = (p.green + 1) % N_GREENS, GREEN, 0
                self._freeze(i)

    def _tick_external(self, switched: np.ndarray) -> None:
        """Lee el estado de SUMO después de simulationStep en los modos 'fixed'/'actuated'."""
        for i, (tid, p) in enumerate(zip(self.tls_ids, self.phases)):
            idx = self.sim.trafficlight.getPhase(tid)
            green, mode = idx // 3, idx % 3
            if green == p.green and mode == p.mode:
                p.elapsed += 1
                continue
            if p.mode == GREEN and mode == YELLOW:
                switched[i] = 1
                p.elapsed = 0
            elif mode == GREEN:
                p.elapsed = 0
            else:  # amarillo -> todo rojo: el tiempo de transición sigue corriendo
                p.elapsed += 1
            p.green, p.mode = green, mode

    # ------------------------------------------------------------ gym
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        self.demand = options.get("demand", self.demand)
        self._seed = 0 if seed is None else int(seed)
        self._start(self._seed)
        if self.control == "actuated":  # el control actuado opera desde t=0
            for tid in self.tls_ids:
                self.sim.trafficlight.setProgram(tid, "actuated")
        self.sim.simulationStep(self.warmup)  # calentamiento (tiempo fijo en los modos 'agent' y 'fixed')
        self._read_phases()
        if self.control == "agent":  # desde aquí las fases solo cambian por decisión del entorno
            for tid in self.tls_ids:
                self.sim.trafficlight.setPhaseDuration(tid, 1e6)
        self.builder.reset_counters(self.sim)
        self.t = 0
        self.state = self.builder.build(self.sim, [p.feature() for p in self.phases])
        return self.state.copy(), self._info(np.zeros(len(self.tls_ids), np.int64))

    def step(self, action):
        action = np.asarray(action, dtype=np.int64).reshape(-1)
        applied = np.zeros(len(self.tls_ids), np.int64)
        forced = np.zeros(len(self.tls_ids), np.int64)
        if self.control == "agent":
            for i, p in enumerate(self.phases):
                if action[i] == 1 and p.mode == GREEN and p.elapsed >= self.min_green:
                    self._start_switch(i)
                    applied[i] = 1
        for _ in range(self.dt):
            if self.control == "agent":
                for i, p in enumerate(self.phases):
                    if p.mode == GREEN and p.elapsed >= self.max_green:
                        self._start_switch(i)
                        applied[i] = forced[i] = 1
            self.sim.simulationStep()
            if self.control == "agent":
                self._tick_agent()
            else:
                self._tick_external(applied)
            self.builder.track_outflow(self.sim)

        prev = self.state
        self.state = self.builder.build(self.sim, [p.feature() for p in self.phases])
        reward_vec = -(self.state[:, W_IDX] - prev[:, W_IDX]) / 100.0
        self.t += 1
        truncated = self.t >= self.max_steps
        info = self._info(applied, forced)
        info["reward_vec"] = reward_vec.astype(np.float32)
        info["requested_action"] = action
        return self.state.copy(), float(reward_vec.sum()), False, truncated, info

    def _info(self, applied: np.ndarray, forced: np.ndarray | None = None) -> dict:
        return {
            "action": applied,
            "forced": forced if forced is not None else np.zeros_like(applied),
            "raw_metrics": self.builder.raw_metrics(self.state),
            "sim_time": self.sim.simulation.getTime(),
            "step": self.t,
        }

    # utilidades para políticas heurísticas (Max-Pressure, etc.)
    def phase_states(self) -> list[PhaseState]:
        return self.phases


def make_env(**kwargs) -> TrafficEnvironment:
    os.environ.setdefault("SUMO_HOME", str(sumo_home()))
    return TrafficEnvironment(**kwargs)
