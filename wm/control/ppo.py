"""PPO (Stable-Baselines3) con los MISMOS hiperparámetros en el Dream Environment y en SUMO directo.

* `make_ppo`: construye PPO desde configs/ppo.yaml (rollout_size = n_steps × n_envs es igual en ambos casos).
* `learn_resumable`: entrena por tramos con un punto de control atómico tras cada tramo, para reanudar una
  corrida interrumpida desde el último tramo terminado (junto con las filas de su curva de aprendizaje).
* `EpisodeCurve`: retorno de cada episodio terminado frente a los pasos de entorno consumidos.
* `SumoPPOEnv`: TrafficEnvironment con la misma observación (estado normalizado aplanado) y la misma escala de
  recompensa que el DreamEnv; cada reinicio elige demanda y semilla de simulación de train.
* `PPOPolicy`: una política PPO entrenada con la interfaz `Policy` (para evaluarla en SUMO).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from wm.control.policies import Policy
from wm.data.normalization import NormStats
from wm.env.traffic_env import TrafficEnvironment, make_env
from wm.utils import load_config


def make_ppo(env, seed: int, n_envs: int, tensorboard_log: str | None = None) -> PPO:
    hp = load_config("ppo")["hyperparams"]
    return PPO("MlpPolicy", env, seed=seed, device="cpu", verbose=0,
               learning_rate=hp["learning_rate"], n_steps=hp["rollout_size"] // n_envs,
               batch_size=hp["batch_size"], n_epochs=hp["n_epochs"], gamma=hp["gamma"],
               gae_lambda=hp["gae_lambda"], clip_range=hp["clip_range"], ent_coef=hp["ent_coef"],
               vf_coef=hp["vf_coef"], max_grad_norm=hp["max_grad_norm"],
               policy_kwargs={"net_arch": {"pi": hp["net_arch"], "vf": hp["net_arch"]}},
               tensorboard_log=tensorboard_log)


class EpisodeCurve(BaseCallback):
    """Una fila por episodio terminado: pasos consumidos hasta ese momento y la información del Monitor
    (retorno `r`, longitud `l` y las `info_keywords`)."""

    def __init__(self):
        super().__init__()
        self.rows: list[dict] = []

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" in info:
                self.rows.append({"timesteps": self.num_timesteps,
                                  **{k: v for k, v in info["episode"].items() if k != "t"}})
        return True


def learn_resumable(make_model: Callable[[], PPO], env, total_timesteps: int, out: Path,
                    callback: BaseCallback, rollouts_per_checkpoint: int) -> PPO:
    """`model.learn` por tramos de `rollouts_per_checkpoint` actualizaciones con punto de control en `out`.

    El callback debe exponer `rows` (lista de dicts serializables): se guardan con cada punto de control y se
    restauran al reanudar. Al reanudar, los entornos se reinician (el episodio en curso se descarta) y el
    generador aleatorio de PPO no se restaura, así que el resultado no es idéntico bit a bit al de una corrida
    sin interrupciones; sí lo son el número de pasos consumidos y la curva registrada hasta el punto de control.
    """
    ckpt, meta = out / "checkpoint.zip", out / "checkpoint.json"
    model = None
    if ckpt.exists() and meta.exists():
        model = PPO.load(str(ckpt), env=env, device="cpu")
        saved = json.loads(meta.read_text(encoding="utf-8"))
        if saved["timesteps"] == model.num_timesteps:
            callback.rows = saved["rows"]
        else:   # interrupción entre las dos escrituras: el punto de control no es coherente
            model = None
    model = model or make_model()
    chunk = rollouts_per_checkpoint * model.n_steps * model.n_envs
    while model.num_timesteps < total_timesteps:
        model.learn(min(chunk, total_timesteps - model.num_timesteps), callback=callback,
                    reset_num_timesteps=False)
        model.save(str(out / "checkpoint_tmp.zip"))
        (out / "checkpoint_tmp.json").write_text(json.dumps({"timesteps": model.num_timesteps,
                                                             "rows": callback.rows}, default=float),
                                                 encoding="utf-8")
        (out / "checkpoint_tmp.zip").replace(ckpt)
        (out / "checkpoint_tmp.json").replace(meta)
    return model


def remove_checkpoint(out: Path) -> None:
    for name in ("checkpoint.zip", "checkpoint.json"):
        (out / name).unlink(missing_ok=True)


class SumoPPOEnv(gym.Env):
    """Envoltura de SUMO para PPO directo (Condición 2)."""

    def __init__(self, norm: NormStats, demands: list[str], seeds: list[int], reward_scale: float,
                 episode_seconds: int | None = None):
        self.env: TrafficEnvironment = make_env(control="agent", episode_seconds=episode_seconds)
        self.norm, self.demands, self.sim_seeds, self.reward_scale = norm, demands, seeds, reward_scale
        n = norm.state_mean.size
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(n,), dtype=np.float32)
        self.action_space = self.env.action_space

    def _obs(self, s: np.ndarray) -> np.ndarray:
        return self.norm.norm_state(s).reshape(-1)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        demand = self.demands[self.np_random.integers(len(self.demands))]
        sim_seed = int(self.sim_seeds[self.np_random.integers(len(self.sim_seeds))])
        obs, info = self.env.reset(seed=sim_seed, options={"demand": demand})
        self.episode_info = {"demand": demand, "sim_seed": sim_seed}
        return self._obs(obs), {**info, **self.episode_info}

    def step(self, action):
        obs, _, term, trunc, info = self.env.step(action)
        reward = float(info["reward_vec"].sum()) / self.reward_scale
        return self._obs(obs), reward, term, trunc, {**info, **self.episode_info}

    def close(self):
        self.env.close()


class PPOPolicy(Policy):
    """Política PPO entrenada (en el sueño o en SUMO), evaluada de forma determinista."""

    def __init__(self, path: Path, norm: NormStats):
        self.model = PPO.load(str(path), device="cpu")
        self.norm = norm

    def act(self, obs, env):
        action, _ = self.model.predict(self.norm.norm_state(obs).reshape(-1), deterministic=True)
        return np.asarray(action, dtype=np.int64)
