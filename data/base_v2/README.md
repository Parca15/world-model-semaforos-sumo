# Dataset `base_v2` — ficha técnica

Generado el 2026-09-24 01:38:44 desde el commit `8ea982111d`. **Congelado y de solo lectura:**
cualquier cambio exige una versión nueva y reentrenar los tres modelos temporales.

- Red: `urban7` (7 intersecciones), Δt = 5 s,
  300 s de calentamiento no registrados.
- Estado `[7, 13]` por paso; acción efectiva por intersección; recompensa por intersección.
- Partición **por semilla** (episodios completos, sin ventanas que crucen episodios ni conjuntos).
- Normalización z-score por (intersección, variable) ajustada **solo con train** (`norm_stats.json`).
- Acceso único: `wm.data.base.load_base("base_v2", split, W, H)`, que verifica `MANIFEST.sha256`.

## Partición

| split    |   episodios |   transiciones | semillas     |
|:---------|------------:|---------------:|:-------------|
| train    |          64 |          46080 | [1, 2, 3, 4] |
| val      |          16 |          11520 | [5]          |
| test     |          16 |          11520 | [6]          |
| test_ood |           8 |           5760 | [7, 8]       |

## Políticas de recolección

| policy   | tipo              |   tasa_cambio |   retorno |   cola_m |   espera_s |
|:---------|:------------------|--------------:|----------:|---------:|-----------:|
| P1       | fixed_time        |         0.222 |  -180.892 |  148.372 |   2893.234 |
| P2       | random_restricted |         0.201 |  -541.289 |  236.168 |   6701.717 |
| P3       | actuated          |         0.281 |    -3.665 |   68.321 |    705.063 |
| P4       | max_pressure      |         0.257 |   -42.250 |  115.972 |   1705.956 |

## Demandas

| demand   |   episodios |   retorno |   cola_m |   espera_s |
|:---------|------------:|----------:|---------:|-----------:|
| D1       |       24.00 |     -6.68 |    43.79 |     503.44 |
| D2       |       24.00 |    -84.47 |   111.73 |    1910.07 |
| D3       |       24.00 |   -462.26 |   257.35 |    6467.59 |
| D4       |       24.00 |   -215.66 |   153.75 |    3217.40 |
| D5       |        8.00 |   -189.10 |   148.87 |    2723.90 |

## Variables del estado (train, unidades originales, todas las intersecciones)

| variable | unidad | normalizada | media | desv. | mín. | mediana | máx. |
|---|---|---|---|---|---|---|---|
| vehicles | veh | sí | 32.78 | 22.94 | 0.00 | 26.00 | 157.00 |
| halting | veh | sí | 18.74 | 18.29 | 0.00 | 12.00 | 157.00 |
| queue_m | m | sí | 140.58 | 137.18 | 0.00 | 90.00 | 1177.50 |
| mean_speed | m/s | sí | 4.19 | 2.16 | 0.00 | 3.98 | 16.78 |
| occupancy | % | sí | 13.93 | 7.91 | 0.00 | 12.64 | 46.04 |
| waiting_acc | s | sí | 2986.31 | 4666.53 | 0.00 | 1121.00 | 43085.00 |
| outflow | veh | sí | 2.07 | 2.13 | 0.00 | 2.00 | 12.00 |
| phase_NS_straight | - | no | 0.29 | 0.45 | 0.00 | 0.00 | 1.00 |
| phase_NS_left | - | no | 0.21 | 0.40 | 0.00 | 0.00 | 1.00 |
| phase_EW_straight | - | no | 0.28 | 0.45 | 0.00 | 0.00 | 1.00 |
| phase_EW_left | - | no | 0.22 | 0.41 | 0.00 | 0.00 | 1.00 |
| time_in_phase | s | sí | 10.12 | 10.23 | 0.00 | 7.00 | 60.00 |
| yellow | - | no | 0.10 | 0.30 | 0.00 | 0.00 | 1.00 |

Recompensa por paso (train): media -0.037, desviación media 3.161.
