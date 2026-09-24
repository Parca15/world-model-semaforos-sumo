# Plan de trabajo — World Models (LSTM · TSMixer · Transformer) para control de semáforos en SUMO

Documento de referencia para ejecutar el proyecto descrito en
`Articulo_WorldModels_LSTM_TSMixer_Transformer_SUMO_1.docx.md`.
Cada etapa tiene **entregables** y un **criterio de salida**. No se pasa a la siguiente etapa sin cumplirlo.

---

## 0. Principios que no se negocian

1. **Una sola variable experimental:** el bloque de modelo temporal (LSTM / TSMixer / Transformer).
   Datos, representación, ventana, partición, pérdida, presupuesto de ajuste, controlador y métricas son idénticos.
2. **Red de 7 intersecciones semaforizadas** en SUMO para todo el proyecto.
3. **Dataset base único y congelado (`base_v1`)**: los tres modelos se entrenan, validan y prueban
   exclusivamente sobre él. Si algo del dataset cambia, se crea `base_v2` y **se reentrenan los tres modelos**.
4. **Partición por episodios completos**: ninguna ventana cruza episodios ni conjuntos.
5. **Normalización ajustada solo con train**.
6. **Reproducibilidad**: semillas fijas, configuración en YAML y un checksum de cada archivo del dataset.
7. **No se decide de antemano cuál modelo gana**: la conclusión sale de las métricas (sección 4).

---

## 1. Requisitos

### 1.1 Software

| Componente | Versión propuesta | Uso |
|---|---|---|
| Python | **3.11** (PyTorch no publica paquetes para Python 3.13 en Mac Intel) | Todo el pipeline |
| SUMO (`eclipse-sumo`, `traci`, `sumolib`, `libsumo`) | 1.20.x | Simulación y control por TraCI/libsumo |
| PyTorch | 2.2.2 (última versión disponible para Mac Intel x86_64) | Modelos temporales y AE/VAE |
| numpy | < 2 (lo exige torch 2.2.2) | Arreglos |
| pandas + pyarrow | 2.x / ≥15 | Índices y tablas del dataset (parquet) |
| gymnasium | 0.29.1 | Interfaz de entorno (SUMO y Dream) |
| stable-baselines3 | 2.3.2 (compatible con torch 2.2) | PPO |
| pyyaml, tqdm, matplotlib | recientes | Configuración, progreso, figuras |

Nota para macOS: los binarios de `eclipse-sumo` necesitan la librería `gettext` (`libintl.8.dylib`),
que se instala con Homebrew.

### 1.2 Hardware (equipo actual: i7-7567U, 4 núcleos, 16 GB, sin GPU CUDA)
- SUMO con 7 intersecciones corre bien en CPU; se usa `libsumo` para acelerar la recolección.
- Los modelos deben ser **pequeños** (~100–250 k parámetros) para que entrenar en CPU sea viable.
  Si hay acceso a una GPU (Colab o un servidor de la UQ), las etapas 6–8 se pueden llevar allá sin cambiar el código.

### 1.3 Estructura del repositorio

```
world model/
├── configs/                 # YAML: escenario, dataset, modelos, PPO, evaluación
├── sumo/                    # red (.net.xml), rutas (.rou.xml), .sumocfg, tls
├── wm/
│   ├── env/                 # TrafficEnvironment (TraCI), CustomStateBuilder, recompensas
│   ├── data/                # recolección, construcción de base_v1, loader común
│   ├── models/              # lstm.py, tsmixer.py, transformer.py, ae.py, baselines.py
│   ├── train/               # bucle de entrenamiento común para los tres modelos
│   ├── dream/               # DreamEnv (gymnasium) que usa el modelo temporal
│   ├── control/             # fixed_time, ppo_sumo, ppo_dream
│   └── eval/                # métricas predictivas, de costo y de control
├── data/base_v1/            # DATASET BASE CONGELADO (ver Etapa 3)
├── runs/                    # checkpoints y logs por experimento/semilla
├── results/                 # tablas CSV y figuras del artículo
└── PLAN_DE_TRABAJO.md
```

---

## 2. Etapas

### Etapa 0 — Entorno y esqueleto del proyecto
- Crear el entorno Python 3.11 e instalar las dependencias de 1.1.
- Crear la estructura de carpetas y el archivo `configs/` base.
- **Entregables:** `requirements.txt`, estructura del repositorio, script `check_env.py`
  (verifica `sumo --version`, TraCI, libsumo, torch y SB3).
- **Criterio de salida:** `check_env.py` pasa en limpio.

### Etapa 1 — Escenario SUMO de 7 intersecciones
- **Topología:** corredor arterial de 7 intersecciones (`J0 … J6`) en línea (1×7), cada una con 4 brazos.
  - Arteria E–O: 2 carriles por sentido, 300 m entre intersecciones, 50 km/h.
  - Calles transversales N–S: 1–2 carriles por sentido, brazos de 200 m.
  - Generada con `netgenerate --grid` (7×1 con `attach-length`) y ajustada con `netconvert`.
  - *Alternativa, si se prefiere una malla:* 7 nodos en una rejilla irregular (2×4 menos uno).
    Se fija **una sola** topología y no se cambia después.
- **Programa semafórico por intersección:** 4 fases verdes
  (NS recto, NS giro, EO recto, EO giro) + amarillo 3 s + todo rojo 1 s.
  Verde mínimo 10 s, verde máximo 60 s.
- **Escenarios de demanda** (flujos con semilla fija):

  | Escenario | Arteria (veh/h por entrada) | Transversales (veh/h por entrada) | Forma temporal |
  |---|---|---|---|
  | D1 Baja | 400 | 150 | constante |
  | D2 Media | 700 | 300 | constante |
  | D3 Alta | 1000 | 450 | constante |
  | D4 Pico | 400 → 1100 → 400 | 150 → 500 → 150 | pico en el centro del episodio |
  | D5 OOD *(solo prueba)* | 850 con desbalance E>O | 350 | no se usa en train |

  Porcentaje de giros: 70 % recto, 15 % izquierda y 15 % derecha (configurable).
- **Duración del episodio:** 300 s de calentamiento (no se registran) + 3600 s registrados.
- **Entregables:** `sumo/corridor7.net.xml`, `sumo/routes_D*.rou.xml`, `corridor7.sumocfg`, y una figura de la red.
- **Criterio de salida:** 5 corridas con la misma semilla dan salidas idénticas, sin teletransportes masivos
  (menos del 1 % de los vehículos).

### Etapa 2 — `TrafficEnvironment` y `CustomStateBuilder`
- **Paso de decisión:** Δt = 5 s, es decir, **720 pasos por episodio**.
- **Acción** (discreta, por intersección): `0 = mantener fase`, `1 = pasar a la siguiente fase`
  (con amarillo automático y respeto a verde mínimo/máximo). Espacio conjunto `MultiDiscrete([2]*7)`.
- **Estado por intersección (F = 13 variables)**, calculado sobre los carriles de entrada:

  | # | Variable | Unidad |
  |---|---|---|
  | 1 | vehículos presentes | veh |
  | 2 | vehículos detenidos (v < 0,1 m/s) | veh |
  | 3 | longitud de cola | m |
  | 4 | velocidad promedio | m/s |
  | 5 | ocupación promedio | % |
  | 6 | tiempo de espera acumulado | s |
  | 7 | flujo de salida durante Δt | veh |
  | 8–11 | fase activa (one-hot, 4 fases) | — |
  | 12 | tiempo en la fase actual | s |
  | 13 | indicador de amarillo activo | {0,1} |

  Estado global: `s_t ∈ ℝ^{7×13}` (91 variables).
- **Recompensa:** por intersección `r_t^i = −(W_{t+1}^i − W_t^i)/100`, donde `W` es el tiempo de espera
  acumulado; la global es `r_t = Σ_i r_t^i`. Además se guardan cola, detenidos y throughput en bruto
  para poder recalcular otras recompensas sin volver a simular.
- **Entregables:** `wm/env/traffic_env.py` (interfaz gymnasium), `wm/env/state_builder.py`, pruebas unitarias.
- **Criterio de salida:** un episodio completo corre con acciones aleatorias, sin errores, y las variables
  están en rangos físicos válidos.

### Etapa 3 — Dataset base `base_v1` (congelado)
Esta es la base sobre la que se entrenan **todos** los modelos (AE/VAE, LSTM, TSMixer, Transformer y el Dream Environment).

**3.1 Políticas de recolección.** Hacen falta acciones variadas para que el modelo aprenda el efecto de cada acción.

| Política | Descripción |
|---|---|
| P1 Tiempo fijo | ciclo fijo de 90 s |
| P2 Aleatoria restringida | cambia con probabilidad p ∈ {0,2; 0,5}, respetando verdes mín/máx |
| P3 Actuada | control actuado de SUMO (detectores) |
| P4 Max-Pressure | heurística de presión, con ε = 0,1 de acciones aleatorias |

**3.2 Diseño de episodios:** 4 demandas (D1–D4) × 4 políticas × 6 semillas = **96 episodios**,
lo que da unas **69 000 transiciones** (720 × 96).

**3.3 Partición por semilla (nunca por ventana):**

| Conjunto | Semillas | Episodios |
|---|---|---|
| Train | 1, 2, 3, 4 | 64 |
| Validación | 5 | 16 |
| Test | 6 | 16 |
| Test-OOD | 7, 8 con demanda D5 | 8 |

**3.4 Ventanas:** entrada de W = 12 pasos (60 s) con `[s_{t−W+1..t}, a_{t−W+1..t}]`.
El objetivo es `Δs_{t+1} = s_{t+1} − s_t` junto con `r_t`. Para evaluar a varios pasos se guardan
continuaciones de hasta H = 20 pasos (100 s). Ninguna ventana cruza el límite de un episodio.

**3.5 Normalización:** z-score por variable, con media y desviación calculadas **solo con train**.
La fase one-hot y el indicador de amarillo no se normalizan. Se guarda en `norm_stats.json`.

**3.6 Formato en disco:**
```
data/base_v1/
├── episodes/ep_XXX.npz     # states[T,7,13], actions[T,7], rewards[T,7], raw_metrics[T,7,k]
├── index.parquet           # episodio, demanda, política, semilla, split, n_pasos
├── splits.json             # lista de episodios por conjunto
├── feature_schema.json     # nombre, unidad y si se normaliza cada variable
├── norm_stats.json         # media/std de train
├── config_used.yaml        # copia exacta de la configuración de generación
├── MANIFEST.sha256         # checksum de cada archivo
└── README.md               # ficha técnica del dataset (estadísticas descriptivas)
```

**3.7 Loader común:** `wm/data/base.py → load_base("base_v1", split, W, H)`. Es el **único** punto de acceso a los
datos para los tres modelos y verifica `MANIFEST.sha256` antes de cargar.

- **Criterio de salida:** los checksums verifican, las estadísticas descriptivas quedan documentadas,
  no hay fuga entre conjuntos (prueba automática) y el dataset se marca como solo lectura.

### Etapa 4 — Baselines
- **Predictivos** (piso que los tres modelos deben superar):
  1. Persistencia: `ŝ_{t+1} = s_t`.
  2. Media móvil de la ventana.
  3. Regresión lineal/ridge sobre la ventana aplanada.
  4. MLP simple, sin estructura temporal.
- **De control:** tiempo fijo (Condición 1), evaluado en los mismos escenarios de prueba.
- **Criterio de salida:** tabla de métricas de los baselines sobre val/test de `base_v1`.

### Etapa 5 — Experimento 0: ¿hace falta AE/VAE?
- Entrenar un AE y un VAE sobre `s_t` (91 → z de 16 o 32 dimensiones).
- Comparar el modelo LSTM de referencia con **estado crudo normalizado** frente a **z**.
- Decisión: se usa z solo si mejora de forma significativa el error a varios pasos (sección 4.4).
  **La decisión se toma una vez y aplica a los tres modelos.**
- **Criterio de salida:** decisión documentada con cifras.

### Etapa 6 — Modelos temporales (núcleo del artículo)
**Interfaz idéntica:** entrada `[B, W, 91+7]` y salida `Δŝ_{t+1} ∈ ℝ^{91}`, `r̂_t ∈ ℝ^{7}`.

| Modelo | Arquitectura |
|---|---|
| LSTM | 2 capas LSTM → capa densa de salida (determinista, sin MDN) |
| TSMixer | N bloques [mezcla temporal MLP + mezcla de variables MLP] con normalización y residuales → proyección |
| Transformer | embedding lineal + codificación posicional → N capas encoder (atención multi-cabeza) → último token → cabeza densa |

**Protocolo de entrenamiento común:**
- Parámetros comparables: objetivo de ~150 k, con **±10 %** entre modelos (se reporta el número exacto).
- Pérdida: `MSE(Δs) + λ·MSE(r)`, con λ = 1 (fijo para todos).
- Optimizador AdamW, *batch* 256, *early stopping* sobre la pérdida de validación (paciencia 10), máximo 200 épocas.
- **Mismo presupuesto de ajuste de hiperparámetros:** 12 configuraciones por modelo (búsqueda aleatoria con semilla).
- **5 semillas** por configuración final.
- Opcional, igual para los tres: entrenamiento con *scheduled sampling* o pérdida multi-paso (k = 5).
- **Criterio de salida:** los tres modelos superan a la persistencia y a ridge en el test,
  con métricas completas de la sección 4.1–4.2.

### Etapa 7 — Dream Environment y controlador PPO
- `DreamEnv` (gymnasium): se reinicia con una ventana **real** muestreada de train de `base_v1` y avanza
  con el modelo temporal. El episodio imaginado dura 40 pasos (200 s), para limitar el error compuesto.
- PPO (SB3, `MultiDiscrete([2]*7)`) con **los mismos hiperparámetros** para los tres World Models.
- Planificación por imaginación (opcional, complementaria): en cada decisión se evalúan K acciones candidatas
  con *rollouts* de horizonte h = 5 y se ejecuta la mejor.
- **Criterio de salida:** una política entrenada por modelo temporal × 5 semillas.

### Etapa 8 — RL directo en SUMO (Condición 2)
- PPO con los mismos hiperparámetros, entrenado interactuando directamente con SUMO.
- Se registra la curva de aprendizaje frente al **número de pasos de SUMO consumidos**.
- **Criterio de salida:** política entrenada × 5 semillas.

### Etapa 9 — Evaluación final en SUMO (las 5 condiciones)
- Condiciones: (1) tiempo fijo, (2) PPO directo, (3) WM + LSTM, (4) WM + TSMixer, (5) WM + Transformer.
- Escenarios de prueba D1–D4 con semillas **nunca vistas** (9–18) + D5 OOD. Son los mismos episodios
  para todas las condiciones.
- **Criterio de salida:** tabla completa de la sección 4.3 con media ± desviación estándar e IC del 95 %.

### Etapa 10 — Análisis estadístico y redacción
- Pruebas estadísticas (sección 4.5), figuras y tablas para las secciones IV y V del artículo.
- Actualizar el artículo con resultados, limitaciones y amenazas a la validez.

---

## 3. Qué es fijo y qué cambia entre modelos

| Elemento | ¿Cambia entre LSTM / TSMixer / Transformer? |
|---|---|
| Red SUMO de 7 intersecciones, demandas y semillas | No |
| Dataset `base_v1`, partición y normalización | No |
| Ventana W = 12 y horizonte H = 20 | No |
| Entrada/salida (Δs, r) y pérdida | No |
| Presupuesto de hiperparámetros y semillas | No |
| DreamEnv, PPO y sus hiperparámetros | No |
| Escenarios y métricas de evaluación | No |
| **Bloque de modelo temporal** | **Sí (única variable)** |

---

## 4. Métricas para comparar los modelos

### 4.1 Precisión predictiva (sobre test y test-OOD de `base_v1`, en unidades originales)

| Métrica | Definición / nota |
|---|---|
| MAE, RMSE del estado a 1 paso | global, por variable y por intersección |
| MAE, RMSE de la recompensa a 1 paso | global y por intersección |
| sMAPE / MAE normalizado | para variables con escalas distintas (cola, velocidad, etc.) |
| R² por variable | qué tanto de la varianza explica cada modelo |
| **Error multi-paso autorregresivo** | RMSE a h = 1, 3, 5, 10, 20 pasos (5 s a 100 s) |
| **Curva de error compuesto** | RMSE(h) frente a h y área bajo la curva (AUC) |
| Razón de crecimiento del error | RMSE(h=20) / RMSE(h=1) |
| **Skill score frente a persistencia** | `1 − RMSE_modelo / RMSE_persistencia` (>0 = mejor que el baseline) |
| Exactitud de la fase predicha | *accuracy* de la fase one-hot (argmax) |
| Error en régimen congestionado | RMSE solo en pasos con cola > percentil 75 |

### 4.2 Costo computacional (mismo hardware, reportado explícitamente)

| Métrica | Nota |
|---|---|
| Número de parámetros entrenables | debe quedar dentro de ±10 % |
| Tiempo por época y tiempo total hasta *early stopping* | segundos (CPU) |
| Latencia de inferencia | ms por paso (batch = 1) y por *rollout* imaginado de 40 pasos |
| Throughput de imaginación | pasos imaginados por segundo (batch = 64) |
| Memoria pico | MB durante el entrenamiento y la inferencia |
| Tiempo de entrenamiento de PPO en DreamEnv | depende de la velocidad del modelo |

### 4.3 Desempeño del control (evaluado en SUMO real)

| Métrica | Fuente SUMO |
|---|---|
| Tiempo promedio de viaje | `tripinfo` |
| Tiempo promedio de espera | `tripinfo` (`waitingTime`) |
| Longitud de cola promedio y máxima | TraCI por intersección |
| Throughput (vehículos que completan el viaje) | `tripinfo` / estadísticas |
| Número promedio de paradas | `tripinfo` (`waitingCount`) |
| Consumo de combustible y CO₂ | `emissions` (modelo HBEFA) |
| Recompensa acumulada | entorno |
| Vehículos teletransportados | estadística de SUMO (control de calidad) |

### 4.4 Eficiencia muestral y fidelidad del World Model

| Métrica | Propósito |
|---|---|
| **Interacciones con SUMO** hasta el desempeño final | recolección de `base_v1` + ajuste fino, frente a PPO directo |
| Pasos de SUMO para alcanzar el 90 % del desempeño de PPO directo | eficiencia muestral |
| Correlación entre retorno imaginado y retorno real en SUMO | ¿el sueño es fiel? |
| Brecha sueño-realidad | `retorno_dream − retorno_SUMO` de la misma política |
| Concordancia en el ranking de acciones | Kendall τ entre el ranking imaginado y el real, sobre K acciones candidatas |

### 4.5 Protocolo estadístico
- 5 semillas de entrenamiento × 10 episodios de prueba por escenario.
- Reportar **media ± desviación estándar** e **IC del 95 %** (bootstrap).
- Modelos predictivos: prueba de **Friedman** entre los tres modelos + post-hoc **Wilcoxon**
  pareado con corrección de **Holm**.
- Control: **Wilcoxon** pareado por episodio (mismos episodios de prueba) entre condiciones, con corrección de Holm.
- Tamaño del efecto: *Cliff's delta*.
- Significancia: α = 0,05.

### 4.6 Tabla final de comparación (plantilla para el artículo)

| Modelo | Parámetros | RMSE h=1 | RMSE h=20 | Skill vs persistencia | ms/paso | T. espera (s) | T. viaje (s) | Cola (m) | Throughput | CO₂ (g) | Pasos SUMO |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Tiempo fijo | — | — | — | — | — | | | | | | 0 |
| PPO directo | — | — | — | — | — | | | | | | |
| WM + LSTM | | | | | | | | | | | |
| WM + TSMixer | | | | | | | | | | | |
| WM + Transformer | | | | | | | | | | | |

**Criterio de "mejor arquitectura":** no hay un ganador único por defecto. Se reporta el frente de Pareto
entre **precisión multi-paso**, **costo computacional** y **desempeño del control**, y la conclusión se
justifica con las pruebas de la sección 4.5.

---

## 5. Riesgos y mitigación

| Riesgo | Mitigación |
|---|---|
| El error compuesto hace inútil el DreamEnv | *rollouts* cortos (40 pasos), reinicio desde estados reales, pérdida multi-paso |
| Poca diversidad de acciones en el dataset | 4 políticas de recolección, incluida la aleatoria restringida |
| La comparación sale injusta por el tamaño de los modelos | parámetros ±10 % y mismo presupuesto de ajuste |
| Entrenar en CPU es lento | modelos pequeños, `libsumo`, GPU externa opcional para las etapas 6–8 |
| Cambios tardíos en el dataset | `base_v1` inmutable + checksums; cualquier cambio → `base_v2` y reentrenar todo |
