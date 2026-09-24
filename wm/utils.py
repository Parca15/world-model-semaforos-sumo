"""Utilidades comunes: rutas del proyecto, binarios de SUMO, configuración y semillas."""
from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"
SUMO_DIR = ROOT / "sumo"
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
RESULTS_DIR = ROOT / "results"


def sumo_home() -> Path:
    """SUMO_HOME: variable de entorno o, si no existe, el paquete pip `eclipse-sumo`."""
    env = os.environ.get("SUMO_HOME")
    if env and Path(env).exists():
        return Path(env)
    import sumo  # paquete eclipse-sumo

    os.environ["SUMO_HOME"] = sumo.SUMO_HOME
    return Path(sumo.SUMO_HOME)


def sumo_binary(name: str = "sumo") -> str:
    """Ruta absoluta de un binario de SUMO (sumo, netconvert, netgenerate, ...)."""
    exe = name + (".exe" if os.name == "nt" else "")
    candidate = sumo_home() / "bin" / exe
    if candidate.exists():
        return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(f"No se encontró el binario de SUMO '{name}'")


sumo_home()  # libsumo necesita SUMO_HOME definida antes de importarse


def load_config(name: str) -> dict:
    """Carga configs/<name>.yaml."""
    path = CONFIGS / (name if name.endswith(".yaml") else f"{name}.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
