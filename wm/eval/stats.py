"""Protocolo estadístico (PLAN_DE_TRABAJO.md, sección 4.5): bootstrap, Wilcoxon pareado + Holm y Cliff's delta."""
from __future__ import annotations

import numpy as np
from scipy import stats


def bootstrap_ci(values, n: int = 10000, alpha: float = 0.05, seed: int = 7) -> tuple[float, float, float]:
    """Media e intervalo de confianza bootstrap percentil."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def cliffs_delta(a, b) -> float:
    """P(a > b) − P(a < b). |δ| < 0,147 despreciable, < 0,33 pequeño, < 0,474 mediano, si no grande."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a[:, None] - b[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)


def cliffs_label(d: float) -> str:
    d = abs(d)
    return "despreciable" if d < 0.147 else "pequeño" if d < 0.33 else "mediano" if d < 0.474 else "grande"


def holm(pvalues: list[float]) -> list[float]:
    """Corrección de Holm-Bonferroni (p ajustados)."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def paired_comparisons(reference: str, samples: dict[str, np.ndarray], lower_is_better: bool = True) -> list[dict]:
    """Wilcoxon pareado de `reference` contra cada otra condición, con corrección de Holm y Cliff's delta."""
    ref = np.asarray(samples[reference], float)
    rows = []
    for name, vals in samples.items():
        if name == reference:
            continue
        vals = np.asarray(vals, float)
        diff = ref - vals
        p = 1.0 if np.allclose(diff, 0) else float(stats.wilcoxon(ref, vals, zero_method="wilcox").pvalue)
        d = cliffs_delta(ref, vals)
        better = (np.median(diff) < 0) if lower_is_better else (np.median(diff) > 0)
        rows.append({"referencia": reference, "contra": name, "n": len(ref), "mediana_dif": float(np.median(diff)),
                     "p": p, "cliffs_delta": d, "efecto": cliffs_label(d), "referencia_mejor": bool(better)})
    for row, p_adj in zip(rows, holm([r["p"] for r in rows])):
        row["p_holm"] = p_adj
    return rows
