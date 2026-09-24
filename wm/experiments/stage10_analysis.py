"""Etapa 10 — análisis estadístico, figuras y tablas para el reporte.

    python -m wm.experiments.stage10_analysis

Lee los resultados de las Etapas 4-9 y produce:
  results/stage10_control_tests.csv     Wilcoxon pareado por escenario + Holm + Cliff's delta
  results/stage10_sample_efficiency.json
  results/tables/*.md                   tablas en Markdown para RESULTADOS_TSMIXER.md
  results/figures/*.png                 figuras
"""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from wm.eval.stats import paired_comparisons  # noqa: E402
from wm.experiments.common import load_json, results_path, run_dir, save_json  # noqa: E402
from wm.experiments.stage9_evaluation import LEARNED, REFERENCE  # noqa: E402
from wm.utils import RESULTS_DIR, load_config  # noqa: E402

# Paleta categórica validada (orden fijo; el color sigue a la entidad, nunca a su posición)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CONDITIONS = list(REFERENCE + LEARNED)
COND_COLOR = dict(zip(CONDITIONS, PALETTE))
MODEL_COLOR = {"TSMixer": PALETTE[0], "Ridge": PALETTE[1], "MLP": PALETTE[2], "Media móvil": PALETTE[3],
               "Persistencia": PALETTE[4]}
FIG = RESULTS_DIR / "figures"
TABLES = RESULTS_DIR / "tables"


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def md(df: pd.DataFrame, name: str, floatfmt: str = ".3f") -> str:
    TABLES.mkdir(parents=True, exist_ok=True)
    text = df.to_markdown(index=False, floatfmt=floatfmt)
    (TABLES / f"{name}.md").write_text(text + "\n", encoding="utf-8")
    return text


# ------------------------------------------------------------ predicción
def prediction_tables() -> None:
    b4 = pd.read_csv(results_path("stage4_baselines.csv"))
    f6 = pd.read_csv(results_path("stage6_final_seeds.csv"))
    cols = ["rmse_1", "rmse_5", "rmse_20", "auc_rmse", "crecimiento", "reward_rmse_1", "phase_acc_1",
            "phase_acc_20", "rmse_1_congestion"]
    for split in ("test", "test_ood"):
        ts = f6[f6.conjunto == split][cols]
        row = {"modelo": "TSMixer (media ± desv., 5 semillas)"}
        row.update({c: f"{ts[c].mean():.3f} ± {ts[c].std():.3f}" for c in cols})
        base = b4[b4.conjunto == split]
        rows = [row] + [{"modelo": r.modelo, **{c: f"{getattr(r, c):.3f}" for c in cols}} for r in base.itertuples()]
        pers = base.set_index("modelo").loc["Persistencia"]
        for r in rows:
            name = "TSMixer" if r["modelo"].startswith("TSMixer") else r["modelo"]
            v1 = ts.rmse_1.mean() if name == "TSMixer" else base.set_index("modelo").loc[name, "rmse_1"]
            v20 = ts.rmse_20.mean() if name == "TSMixer" else base.set_index("modelo").loc[name, "rmse_20"]
            r["skill_1"] = f"{1 - v1 / pers.rmse_1:+.3f}"
            r["skill_20"] = f"{1 - v20 / pers.rmse_20:+.3f}"
        md(pd.DataFrame(rows), f"prediccion_{split}")

    # por variable (1 paso, test, unidades originales) — TSMixer semilla 0 vs Ridge
    ts0 = load_json(results_path("stage6/seed0.json"))["metrics"]["test"]["per_variable"]
    rd = load_json(results_path("stage4/ridge.json"))["metrics"]["test"]["per_variable"]
    per_var = pd.DataFrame([{"variable": a["variable"], "MAE TSMixer": a["mae"], "MAE Ridge": b["mae"],
                             "RMSE TSMixer": a["rmse"], "RMSE Ridge": b["rmse"], "R² TSMixer": a["r2"],
                             "R² Ridge": b["r2"], "sMAPE % TSMixer": a["smape_pct"]} for a, b in zip(ts0, rd)])
    md(per_var, "prediccion_por_variable")

    # pruebas TSMixer vs baselines
    s6 = load_json(results_path("stage6_summary.json"))
    rows = []
    for key, label in (("rmse_1", "1 paso"), ("rmse_H", "20 pasos")):
        for c in s6["comparisons_test"][key]:
            rows.append({"horizonte": label, "TSMixer vs": c["contra"], "mediana dif. RMSE": c["mediana_dif"],
                         "p (Holm)": c["p_holm"], "Cliff δ": c["cliffs_delta"], "efecto": c["efecto"],
                         "TSMixer mejor": "sí" if c["referencia_mejor"] else "no"})
    md(pd.DataFrame(rows), "prediccion_pruebas", floatfmt=".4g")

    # figura RMSE(h)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    H = np.arange(1, 21)
    for name, fname in (("Persistencia", "persistencia"), ("Media móvil", "media_móvil"), ("MLP", "mlp"),
                        ("Ridge", "ridge")):
        y = load_json(results_path(f"stage4/{fname}.json"))["metrics"]["test"]["rmse_h"]
        ax.plot(H, y, color=MODEL_COLOR[name], lw=2, label=name)
        ax.annotate(name, (20, y[-1]), xytext=(4, 0), textcoords="offset points", color=INK2, fontsize=8,
                    va="center")
    curves = np.array([load_json(results_path(f"stage6/seed{s}.json"))["metrics"]["test"]["rmse_h"]
                       for s in load_config("train")["final_seeds"]])
    m, sd = curves.mean(0), curves.std(0)
    ax.fill_between(H, m - sd, m + sd, color=MODEL_COLOR["TSMixer"], alpha=0.18, lw=0)
    ax.plot(H, m, color=MODEL_COLOR["TSMixer"], lw=2, label="TSMixer (5 semillas)")
    ax.annotate("TSMixer", (20, m[-1]), xytext=(4, 0), textcoords="offset points", color=INK, fontsize=8,
                va="center", weight="bold")
    style(ax, "Error autorregresivo en test según el horizonte", "horizonte h (pasos de 5 s)",
          "RMSE (estado normalizado)")
    ax.set_xticks([1, 3, 5, 10, 15, 20])
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "rmse_vs_horizonte.png")

    # curvas de entrenamiento
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for s in load_config("train")["final_seeds"]:
        log = pd.read_json(run_dir("tsmixer", "final", f"seed{s}") / "log.jsonl", lines=True)
        ax.plot(log.epoch, log.val_loss, color=PALETTE[0], alpha=0.35 + 0.13 * s, lw=1.5,
                label=f"semilla {s}")
    style(ax, "TSMixer: pérdida de validación por época (5 semillas)", "época (11 040 ventanas)",
          "MSE(Δs) + MSE(r) en validación")
    ax.legend(frameon=False, fontsize=8, ncol=5)
    save(fig, "tsmixer_entrenamiento.png")


# ------------------------------------------------------------ control
def control_analysis() -> None:
    ep = pd.read_csv(results_path("stage9_control_episodes.csv"))
    summ = pd.read_csv(results_path("stage9_control_summary.csv"))

    def fmt(r, m, d=1):
        return f"{r[m + '_mean']:.{d}f} ± {r[m + '_std']:.{d}f} [{r[m + '_ci_lo']:.{d}f}, {r[m + '_ci_hi']:.{d}f}]"

    rows = []
    for r in summ.to_dict("records"):
        rows.append({"condición": r["condition"], "espera (s)": fmt(r, "waiting_time_s"),
                     "viaje (s)": fmt(r, "travel_time_s"), "cola media (m)": fmt(r, "queue_mean_m"),
                     "cola máx. (m)": fmt(r, "queue_max_m", 0), "throughput": fmt(r, "throughput", 0),
                     "paradas": fmt(r, "stops", 2), "CO₂ (g/veh)": fmt(r, "co2_g_per_veh", 0),
                     "recompensa": fmt(r, "reward", 0), "teletransp.": f"{r['teleports_mean']:.2f}"})
    md(pd.DataFrame(rows), "control_resumen")

    per_scen = ep.groupby(["condition", "demand", "seed"]).mean(numeric_only=True).reset_index()
    by_dem = per_scen.pivot_table(index="demand", columns="condition", values="waiting_time_s", aggfunc="mean")
    md(by_dem[[c for c in CONDITIONS if c in by_dem]].reset_index(), "control_espera_por_demanda", floatfmt=".1f")
    by_dem_t = per_scen.pivot_table(index="demand", columns="condition", values="travel_time_s", aggfunc="mean")
    md(by_dem_t[[c for c in CONDITIONS if c in by_dem_t]].reset_index(), "control_viaje_por_demanda", floatfmt=".1f")

    # pruebas pareadas por escenario
    tests = []
    for metric, lower in (("waiting_time_s", True), ("travel_time_s", True), ("queue_mean_m", True),
                          ("throughput", False), ("co2_g_per_veh", True)):
        piv = per_scen.pivot_table(index=["demand", "seed"], columns="condition", values=metric)
        for ref in ("WM + TSMixer", "WM + TSMixer + planificación"):
            for row in paired_comparisons(ref, {c: piv[c].values for c in CONDITIONS}, lower_is_better=lower):
                tests.append({"métrica": metric, **row})
        for row in paired_comparisons("PPO directo", {c: piv[c].values for c in ("PPO directo", "Tiempo fijo")},
                                      lower_is_better=lower):
            tests.append({"métrica": metric, **row})
    tdf = pd.DataFrame(tests)
    tdf.to_csv(results_path("stage10_control_tests.csv"), index=False)
    show = tdf[tdf.métrica.isin(["waiting_time_s", "travel_time_s"])]
    md(show[["métrica", "referencia", "contra", "n", "mediana_dif", "p_holm", "cliffs_delta", "efecto",
             "referencia_mejor"]], "control_pruebas", floatfmt=".4g")

    # figura: espera por demanda
    fig, ax = plt.subplots(figsize=(9, 4.2))
    demands = list(by_dem.index)
    width = 0.8 / len(CONDITIONS)
    for i, c in enumerate(CONDITIONS):
        if c not in by_dem:
            continue
        x = np.arange(len(demands)) + (i - (len(CONDITIONS) - 1) / 2) * width
        ax.bar(x, by_dem[c].values, width=width * 0.92, color=COND_COLOR[c], label=c)
    ax.set_xticks(np.arange(len(demands)), [f"{d}{' (OOD)' if d == 'D5' else ''}" for d in demands])
    style(ax, "Tiempo de espera medio por vehículo en SUMO (escenarios de evaluación)", "demanda", "espera (s)")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    save(fig, "control_espera_por_demanda.png")


# ------------------------------------------------------------ eficiencia muestral y aprendizaje
def sample_efficiency() -> None:
    ppo = load_config("ppo")
    seeds = ppo["seeds"]
    curves = []
    for s in ppo["sumo"]["train_seeds"]:
        mon = pd.read_csv(run_dir("ppo_sumo", f"seed{s}") / "monitor.monitor.csv", skiprows=1)
        mon["steps"] = mon.l.cumsum()
        mon["seed"] = s
        curves.append(mon)
    mon = pd.concat(curves)
    # progreso normalizado: fracción de la mejora entre el primer episodio y el final (media móvil de 5)
    reach = []
    for s, g in mon.groupby("seed"):
        r = g.r.rolling(5, min_periods=1).mean().values
        first, final = r[0], r[-1]
        prog = (r - first) / (final - first) if final != first else np.ones_like(r)
        idx = np.flatnonzero(prog >= 0.9)
        reach.append(int(g.steps.values[idx[0]]) if len(idx) else None)
    splits = load_json(RESULTS_DIR.parent / "data" / load_config("base")["dataset_version"] / "splits.json")
    wm_steps = (len(splits["train"]) + len(splits["val"])) * 720
    out = {"sumo_steps_world_model": wm_steps, "sumo_steps_ppo_direct": ppo["sumo"]["total_timesteps"],
           "dream_steps_ppo": ppo["dream"]["total_timesteps"],
           "ppo_direct_steps_to_90pct": reach,
           "ppo_direct_steps_to_90pct_mean": float(np.mean([r for r in reach if r is not None]))}
    save_json(out, results_path("stage10_sample_efficiency.json"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for s, g in mon.groupby("seed"):
        axes[0].plot(g.steps, g.r.rolling(5, min_periods=1).mean(), color=COND_COLOR["PPO directo"],
                     alpha=0.35 + 0.13 * s, lw=1.5, label=f"semilla {s}")
    style(axes[0], "PPO directo: aprendizaje frente a pasos de SUMO", "pasos de SUMO consumidos",
          "retorno por episodio (media móvil 5)")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    for s in seeds:
        c = pd.read_csv(run_dir("ppo_dream", f"seed{s}") / "curve.csv")
        axes[1].plot(c.timesteps, c.dream_return_mean, color=COND_COLOR["WM + TSMixer"], alpha=0.35 + 0.13 * s,
                     lw=1.5, label=f"semilla {s}")
    style(axes[1], "PPO en el sueño (TSMixer): 0 pasos de SUMO", "pasos imaginados",
          "retorno imaginado (episodios de 40 pasos)")
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    save(fig, "curvas_ppo.png")


def fidelity_figure() -> None:
    fid = load_json(results_path("stage9_fidelity.json"))
    rows = [{"semilla": k, **v} for k, v in fid["logged"].items()]
    md(pd.DataFrame(rows)[["semilla", "pearson", "spearman", "gap_mean", "gap_mae", "real_mean",
                           "imagined_mean"]], "fidelidad_registrada")
    r = fid["ranking"]
    md(pd.DataFrame([{"estados": r["states"], "candidatas": r["candidates"], "horizonte": r["horizon"],
                      "Kendall τ medio": r["tau_mean"], "fracción de estados con τ > 0": r["frac_states_tau_positive"],
                      "acierto top-1 (semilla 0)": r["top1_agreement_seed0"]}]), "fidelidad_ranking")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    prediction_tables()
    control_analysis()
    sample_efficiency()
    fidelity_figure()
    print("Etapa 10: tablas en", TABLES, "y figuras en", FIG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
