# Registro de versiones del dataset base

Los datasets son inmutables: una versión nunca se modifica. Cualquier cambio en el escenario, el entorno,
las políticas de recolección o el procesamiento crea una versión nueva y obliga a reentrenar los tres modelos.
Los episodios (`episodes/*.npz`) no se versionan en git; los metadatos sí, y `MANIFEST.sha256` garantiza la integridad.

| Versión | Estado | Commit de construcción | Notas |
|---|---|---|---|
| `base_v1` | **descartada antes de usarse** | `8247d70` | El sensor de presión de P4 (Max-Pressure) contaba solo los 60 m del bolsillo de giro en la entrada y todos los vehículos de la salida, así que P4 cambiaba de fase casi siempre y rendía peor que el tiempo fijo. Ningún modelo se entrenó con esta versión. |
| `base_v2` | vigente | ver `base_v2/config_used.yaml` | P4 corregido: cola de entrada = bolsillo + tramo aguas arriba, cola de salida = vehículos detenidos, y regla cíclica (cambiar si la fase siguiente tiene más presión). |
