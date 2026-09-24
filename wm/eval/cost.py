"""Costo computacional de inferencia (PLAN_DE_TRABAJO.md, sección 4.2), medido en el mismo hardware."""
from __future__ import annotations

import os
import platform
import time

import numpy as np
import psutil
import torch
from torch import nn


def hardware() -> str:
    return f"{platform.processor() or platform.machine()} | {psutil.cpu_count(logical=False)} núcleos | " \
           f"{psutil.virtual_memory().total / 2**30:.1f} GB RAM | torch {torch.__version__} (CPU)"


@torch.no_grad()
def inference_cost(model: nn.Module, window: int, input_dim: int, state_dim: int, threads: int = 1,
                   rollout_steps: int = 40, repeats: int = 200) -> dict:
    torch.set_num_threads(threads)
    model.eval()
    proc = psutil.Process(os.getpid())
    rss0 = proc.memory_info().rss

    x1 = torch.randn(1, window, input_dim)
    for _ in range(20):
        model(x1)
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        model(x1)
        times.append(time.perf_counter() - t)
    step_ms = 1000 * float(np.median(times))

    def imagine(x, steps):
        for _ in range(steps):
            d, _ = model(x)
            s = x[:, -1, :state_dim] + d
            row = torch.cat([s, x[:, -1, state_dim:]], 1)[:, None]
            x = torch.cat([x[:, 1:], row], 1)

    t = time.perf_counter()
    for _ in range(10):
        imagine(x1, rollout_steps)
    rollout_ms = 1000 * (time.perf_counter() - t) / 10

    x64 = torch.randn(64, window, input_dim)
    imagine(x64, 5)
    t = time.perf_counter()
    imagine(x64, rollout_steps)
    throughput = 64 * rollout_steps / (time.perf_counter() - t)

    return {"latency_ms_step_b1": step_ms, "latency_ms_rollout40_b1": rollout_ms,
            "imagined_steps_per_s_b64": throughput,
            "inference_rss_delta_mb": (proc.memory_info().rss - rss0) / 2**20,
            "inference_rss_mb": proc.memory_info().rss / 2**20, "threads": threads}
