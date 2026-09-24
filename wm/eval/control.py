"""Evaluación del control en SUMO real (PLAN_DE_TRABAJO.md, sección 4.3).

Un episodio de evaluación corre una política sobre (demanda, semilla) y devuelve métricas de tripinfo
(viaje, espera, paradas, CO₂, combustible, throughput), de TraCI (colas) y del entorno (recompensa).
Solo cuentan los vehículos que salen después del calentamiento (viajes registrados completos).
"""
from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import numpy as np

from wm.control.policies import Policy
from wm.env.traffic_env import make_env
from wm.utils import load_config


def _tripinfo_metrics(path: Path, warmup: float) -> dict:
    trips = [t for t in ET.parse(path).getroot().iter("tripinfo")]
    done = [t for t in trips if float(t.get("depart")) >= warmup]
    arrived = [t for t in trips if float(t.get("arrival")) >= warmup]

    def mean(attr):
        return float(np.mean([float(t.get(attr)) for t in done])) if done else float("nan")

    co2 = [float(t.find("emissions").get("CO2_abs")) / 1000 for t in done if t.find("emissions") is not None]
    fuel = [float(t.find("emissions").get("fuel_abs")) / 1000 for t in done if t.find("emissions") is not None]
    return {"travel_time_s": mean("duration"), "waiting_time_s": mean("waitingTime"),
            "stops": mean("waitingCount"), "throughput": len(arrived), "trips_recorded": len(done),
            "co2_g_per_veh": float(np.mean(co2)) if co2 else float("nan"),
            "fuel_g_per_veh": float(np.mean(fuel)) if fuel else float("nan"),
            "co2_kg_total": float(np.sum(co2)) / 1000}


def _statistics(path: Path) -> dict:
    root = ET.parse(path).getroot()
    veh = root.find("vehicles")
    return {"teleports": int(root.find("teleports").get("total")), "loaded": int(veh.get("loaded")),
            "inserted": int(veh.get("inserted")), "backlog_end": int(veh.get("waiting"))}


def run_episode(policy: Policy, demand: str, seed: int, policy_seed: int = 0) -> dict:
    warmup = load_config("base")["env"]["warmup"]
    with tempfile.TemporaryDirectory() as tmp:
        trip, stats = Path(tmp) / "trip.xml", Path(tmp) / "stats.xml"
        env = make_env(demand=demand, control=policy.control,
                       sumo_args=["--tripinfo-output", str(trip), "--statistic-output", str(stats),
                                  "--device.emissions.probability", "1", "--duration-log.disable", "true"])
        try:
            obs, info = env.reset(seed=seed)
            policy.reset(env, np.random.default_rng([seed, policy_seed, zlib.crc32(demand.encode())]))
            queues, reward, switches = [info["raw_metrics"][:, 0]], 0.0, 0
            trunc = False
            while not trunc:
                obs, _, _, trunc, info = env.step(policy.act(obs, env))
                policy.observe(info)
                queues.append(info["raw_metrics"][:, 0])
                reward += float(info["reward_vec"].sum())
                switches += int(info["action"].sum())
        finally:
            env.close()
        q = np.stack(queues)
        return {"demand": demand, "seed": seed, "reward": reward, "queue_mean_m": float(q.mean()),
                "queue_max_m": float(q.max()), "switches": switches,
                **_tripinfo_metrics(trip, warmup), **_statistics(stats)}
