"""Recolección de un episodio de SUMO con una política heurística."""
from __future__ import annotations

import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from wm.control.policies import make_policy
from wm.env.traffic_env import make_env


@dataclass(frozen=True)
class EpisodeSpec:
    episode: str            # identificador, p. ej. "ep_000"
    demand: str             # D1..D5
    policy: str             # P1..P4
    policy_kind: str        # fixed_time | random_restricted | actuated | max_pressure
    seed: int               # semilla de SUMO (define la partición)
    split: str              # train | val | test | test_ood
    policy_params: dict = field(default_factory=dict)

    def policy_rng(self) -> np.random.Generator:
        """Generador de la política, reproducible y distinto para cada (semilla, política, demanda)."""
        key = [self.seed, zlib.crc32(self.policy.encode()), zlib.crc32(self.demand.encode())]
        return np.random.default_rng(key)


@dataclass
class EpisodeRecord:
    spec: EpisodeSpec
    states: np.ndarray       # [T+1, 7, 13]  s_0 .. s_T
    actions: np.ndarray      # [T, 7]        acción efectiva a_t (1 = empezó un cambio de fase)
    forced: np.ndarray       # [T, 7]        el cambio lo forzó el verde máximo
    rewards: np.ndarray      # [T, 7]        r_t por intersección
    raw_metrics: np.ndarray  # [T+1, 7, k]   cola, detenidos, flujo de salida, espera (sin normalizar)

    @property
    def n_steps(self) -> int:
        return len(self.actions)

    def save(self, path: Path, compress: bool = True) -> None:
        saver = np.savez_compressed if compress else np.savez
        saver(path, states=self.states, actions=self.actions, forced=self.forced,
              rewards=self.rewards, raw_metrics=self.raw_metrics)

    def summary(self) -> dict:
        return {
            **{k: v for k, v in asdict(self.spec).items() if k != "policy_params"},
            "policy_params": str(self.spec.policy_params),
            "n_steps": self.n_steps,
            "return": float(self.rewards.sum()),
            "switch_rate": float(self.actions.mean()),
            "mean_queue_m": float(self.raw_metrics[..., 0].mean()),
            "mean_waiting_s": float(self.raw_metrics[..., 3].mean()),
        }


def run_episode(spec: EpisodeSpec, episode_seconds: int | None = None) -> EpisodeRecord:
    policy = make_policy(spec.policy_kind, **spec.policy_params)
    env = make_env(demand=spec.demand, control=policy.control, episode_seconds=episode_seconds)
    try:
        obs, info = env.reset(seed=spec.seed)
        policy.reset(env, spec.policy_rng())
        states, raws = [obs], [info["raw_metrics"]]
        actions, forced, rewards = [], [], []
        truncated = False
        while not truncated:
            obs, _, _, truncated, info = env.step(policy.act(obs, env))
            states.append(obs)
            raws.append(info["raw_metrics"])
            actions.append(info["action"])
            forced.append(info["forced"])
            rewards.append(info["reward_vec"])
    finally:
        env.close()
    return EpisodeRecord(
        spec=spec,
        states=np.stack(states).astype(np.float32),
        actions=np.stack(actions).astype(np.int8),
        forced=np.stack(forced).astype(np.int8),
        rewards=np.stack(rewards).astype(np.float32),
        raw_metrics=np.stack(raws).astype(np.float32),
    )
