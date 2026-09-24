"""Etapa 0 — verifica que el entorno tiene todo lo necesario para el proyecto.

Uso:  python check_env.py
Sale con código 0 si todas las verificaciones pasan.
"""
from __future__ import annotations

import subprocess
import sys

from wm.utils import sumo_binary, sumo_home

results: list[tuple[str, bool, str]] = []


def check(name: str):
    def deco(fn):
        try:
            info = fn()
            results.append((name, True, info or ""))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"{type(e).__name__}: {e}"))
        return fn

    return deco


@check("Python 3.11")
def _python():
    v = sys.version_info
    assert (v.major, v.minor) == (3, 11), f"se encontró {v.major}.{v.minor}"
    return sys.version.split()[0]


@check("SUMO_HOME")
def _sumo_home():
    return str(sumo_home())


@check("sumo --version")
def _sumo_version():
    out = subprocess.run([sumo_binary("sumo"), "--version"], capture_output=True, text=True, check=True)
    return out.stdout.splitlines()[0]


@check("netgenerate / netconvert")
def _net_tools():
    return ", ".join(sumo_binary(b).rsplit("\\", 1)[-1] for b in ("netgenerate", "netconvert"))


@check("TraCI")
def _traci():
    import traci

    return traci.__version__ if hasattr(traci, "__version__") else "ok"


@check("libsumo")
def _libsumo():
    import libsumo

    return libsumo.__file__


@check("sumolib")
def _sumolib():
    import sumolib  # noqa: F401

    return "ok"


@check("PyTorch")
def _torch():
    import torch

    x = torch.randn(4, 12, 98)
    lstm = torch.nn.LSTM(98, 32, batch_first=True)
    y, _ = lstm(x)
    assert y.shape == (4, 12, 32)
    return f"{torch.__version__} (CUDA={torch.cuda.is_available()}, hilos={torch.get_num_threads()})"


@check("numpy < 2")
def _numpy():
    import numpy as np

    assert int(np.__version__.split(".")[0]) < 2, np.__version__
    return np.__version__


@check("pandas + pyarrow")
def _pandas():
    import pandas as pd
    import pyarrow

    return f"pandas {pd.__version__}, pyarrow {pyarrow.__version__}"


@check("gymnasium")
def _gym():
    import gymnasium

    return gymnasium.__version__


@check("stable-baselines3 (PPO)")
def _sb3():
    import gymnasium as gym
    import stable_baselines3 as sb3
    from stable_baselines3 import PPO

    env = gym.make("CartPole-v1")
    PPO("MlpPolicy", env, n_steps=64, batch_size=32, verbose=0).learn(64)
    return sb3.__version__


@check("pyyaml, tqdm, matplotlib")
def _misc():
    import matplotlib
    import tqdm
    import yaml

    return f"yaml {yaml.__version__}, tqdm {tqdm.__version__}, mpl {matplotlib.__version__}"


if __name__ == "__main__":
    width = max(len(n) for n, _, _ in results)
    for name, ok, info in results:
        print(f"[{'OK' if ok else 'FALLA'}] {name.ljust(width)}  {info}")
    failed = [n for n, ok, _ in results if not ok]
    print("\nEntorno listo." if not failed else f"\n{len(failed)} verificación(es) fallaron: {failed}")
    sys.exit(1 if failed else 0)
