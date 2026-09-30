"""Diagrama de las tres arquitecturas tal como se implementaron (configuración final de la búsqueda).

    python articulo/diagrama_arquitecturas.py results/figures/arquitecturas_implementadas.png
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = sys.argv[1]
INK, INK2, EDGE = "#0b0b0b", "#52514e", "#d8d6d0"
COLOR = {"LSTM": "#008300", "TSMixer": "#2a78d6", "Transformer": "#4a3aa7"}

MODELS = {
    "LSTM": ["Ventana x ∈ ℝ¹²ˣ⁹⁸", "LSTM capa 1 (h = 93)", "LSTM capa 2 (h = 93)", "último estado oculto h₁₂",
             "Densa 93 → 98"],
    "TSMixer": ["Ventana x ∈ ℝ¹²ˣ⁹⁸", "Proyección 98 → 64", "6 × bloque mixer\nmezcla temporal (12×12)\n"
                "mezcla de variables (64→174→64)\nBatchNorm + residuales", "Proyección temporal 12 → 1",
                "Densa 64 → 98"],
    "Transformer": ["Ventana x ∈ ℝ¹²ˣ⁹⁸", "Embedding 98 → 64\n+ posición sinusoidal",
                    "4 × capa encoder (pre-norm)\natención causal, 4 cabezas\nMLP 64→134→64 + residuales",
                    "último token", "Densa 64 → 98"],
}
PARAMS = {"LSTM": "150 944 parámetros", "TSMixer": "150 251 parámetros", "Transformer": "149 818 parámetros"}

fig, axes = plt.subplots(1, 3, figsize=(12, 4.9))
for ax, (name, boxes) in zip(axes, MODELS.items()):
    ax.set_xlim(0, 1)
    ax.set_ylim(0.22, 1)
    ax.axis("off")
    ax.text(0.5, 0.985, name, ha="center", va="top", fontsize=14, weight="bold", color=COLOR[name])
    ax.text(0.5, 0.925, PARAMS[name], ha="center", va="top", fontsize=9, color=INK2)
    heights = [0.07 if "\n" not in b else 0.05 + 0.035 * b.count("\n") for b in boxes]
    gap = 0.045
    y = 0.86
    centers = []
    for i, (b, h) in enumerate(zip(boxes, heights)):
        core = 0 < i < len(boxes) - 1
        face = COLOR[name] if core else "white"
        patch = FancyBboxPatch((0.08, y - h), 0.84, h, boxstyle="round,pad=0.008,rounding_size=0.02",
                               fc=face, ec=COLOR[name] if core else EDGE, lw=1.2, alpha=0.14 if core else 1)
        ax.add_patch(patch)
        ax.add_patch(FancyBboxPatch((0.08, y - h), 0.84, h, boxstyle="round,pad=0.008,rounding_size=0.02",
                                    fc="none", ec=COLOR[name] if core else EDGE, lw=1.2))
        ax.text(0.5, y - h / 2, b, ha="center", va="center", fontsize=9, color=INK, linespacing=1.35)
        centers.append((y, y - h))
        y -= h + gap
    for (top, bottom), (ntop, _) in zip(centers[:-1], centers[1:]):
        ax.annotate("", xy=(0.5, ntop + 0.004), xytext=(0.5, bottom - 0.004),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.1))
    last = centers[-1][1]
    ax.text(0.5, last - 0.03, "Δŝₜ₊₁ ∈ ℝ⁹¹   ·   r̂ₜ ∈ ℝ⁷", ha="center", va="top", fontsize=9.5,
            color=INK)
fig.suptitle("Entrada y salida idénticas; solo cambia el bloque central (≈150 k parámetros, ±10 %)",
             fontsize=10.5, color=INK2, y=0.03)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
