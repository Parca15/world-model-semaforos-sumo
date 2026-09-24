# World Models (LSTM · TSMixer · Transformer) para control de semáforos en SUMO

Implementación del proyecto descrito en `Articulo_WorldModels_LSTM_TSMixer_Transformer_SUMO_1.docx.md`.
El plan por etapas y los criterios de salida están en [`PLAN_DE_TRABAJO.md`](PLAN_DE_TRABAJO.md).

## Instalación (Python 3.11)

```bash
# Windows
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
# macOS / Linux
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt

python check_env.py      # Etapa 0: debe terminar con "Entorno listo."
```

SUMO se instala desde pip (`eclipse-sumo`); `wm/utils.py` encuentra los binarios automáticamente
aunque `SUMO_HOME` no esté definida. En macOS hace falta `brew install gettext`.

## Estado de las etapas

| Etapa | Descripción | Estado |
|---|---|---|
| 0 | Entorno y esqueleto | ✅ |
| 1 | Escenario SUMO de 7 intersecciones | ⏳ |
| 2 | `TrafficEnvironment` y `CustomStateBuilder` | — |
| 3 | Dataset `base_v1` | — |
| 4 | Baselines | — |
| 5 | Experimento 0 (AE/VAE) | — |
| 6 | Modelos temporales | — |
| 7 | Dream Environment + PPO | — |
| 8 | PPO directo en SUMO | — |
| 9 | Evaluación final | — |
| 10 | Análisis y redacción | — |
