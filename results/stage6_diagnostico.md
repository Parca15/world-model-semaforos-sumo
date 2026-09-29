# Etapa 6 — diagnóstico: TSMixer frente a Ridge (2026-09-29)

## Síntoma
Con `max_epochs: 60`, el TSMixer (mejor configuración de la búsqueda: 6 bloques, d_model 64, dropout 0,1,
BatchNorm, lr 3e-3) en la época 18–21 quedaba por detrás de Ridge en test:

| Modelo | RMSE 1 paso | RMSE 20 pasos |
|---|---|---|
| Ridge (Etapa 4) | 0,245 | 0,442 |
| TSMixer semilla 0, época 21 | 0,294 | 0,481 |

El error era 10–30 % mayor en casi todas las variables (1,9× en `yellow`), sin ninguna variable problemática
en particular.

## Hipótesis descartadas (15 épocas, protocolo de la búsqueda, mismo presupuesto de parámetros)
Pérdida Δs de validación (k = 5) en la época 15:

| Variante | val Δs |
|---|---|
| Actual (proyección de entrada a d_model) | **0,144** |
| TSMixer "nativo" (mezcla sobre las 98 variables, sin proyección), BatchNorm | 0,231 (época 6; peor) |
| TSMixer nativo, LayerNorm, 4 bloques | 0,178 (época 8; igual o peor) |
| lr con decaimiento coseno | 0,154 |
| Conexión lineal directa desde el último paso | 0,156 |
| Coseno + conexión lineal | 0,179 |

Ninguna variante mejora la arquitectura actual; no se adopta ninguna.

## Diagnóstico: falta de entrenamiento
Con `samples_per_epoch: 11040` y batch 256, una época tiene solo 43 pasos del optimizador: 60 épocas son unos
2 600 pasos. Las curvas seguían bajando de forma sostenida al llegar al tope. Con la MISMA pérdida (k = 5) sobre
la misma submuestra de validación:

| Modelo | val Δs | val r | total |
|---|---|---|---|
| Ridge (alpha = 1) | 0,123 | 0,933 | 1,056 |
| TSMixer, época 21 | 0,131 | 0,651 | 0,781 |
| TSMixer, época 40 (corrida interrumpida) | 0,115 | 0,625 | 0,740 |

El TSMixer ya supera a Ridge en la pérdida total y en la recompensa, y en el estado lo supera a partir de la
época ~40. Ridge gana a 1 paso porque se ajusta exactamente a 1 paso, mientras que el TSMixer optimiza el error
medio de 5 pasos.

## Decisión
Se vuelve al valor del plan: `max_epochs: 200` con early stopping (paciencia 10). No cambia el protocolo común;
el baseline MLP de la Etapa 4 se entrenó sin tope efectivo (paró por early stopping en la época 71), así que la
comparación queda en igualdad de condiciones.
