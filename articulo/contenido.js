// Texto del artículo. Todas las cifras provienen de results/ (commits 2928661 … 4b6d99c).
module.exports = ({ P, H1, H2, EQ, BULLET, figure, table, runs, Paragraph, TextRun, AlignmentType }) => {
  const center = (children, spacing = { after: 120 }) => new Paragraph({ children, alignment: AlignmentType.CENTER, spacing });
  const out = [];
  const add = (...xs) => xs.flat().forEach((x) => out.push(x));

  // ============================================================ portada
  add(center([new TextRun({ text: "Modelos Temporales para World Models Aplicados al Control Inteligente de Semáforos en Entornos de Simulación SUMO: una Comparación entre LSTM, TSMixer y Transformer", bold: true, size: 30 })], { after: 240 }));
  add(center([new TextRun({ text: "Julián Andrés Ladino Moreno" }), new TextRun({ text: "1", superScript: true }),
              new TextRun({ text: ", Hulbert Alejandro Arango Fajardo" }), new TextRun({ text: "1", superScript: true }),
              new TextRun({ text: ", Jeyson Styven Caceres Mosquera" }), new TextRun({ text: "1", superScript: true })]));
  add(center([new TextRun({ text: "1", superScript: true, size: 20 }),
              new TextRun({ text: "Programa de Ingeniería de Sistemas y Computación, Universidad del Quindío, UQ. Armenia, Colombia", size: 20 })], { after: 300 }));

  add(P("**Resumen**", { alignment: AlignmentType.LEFT }));
  add(P("La congestión vehicular es uno de los principales problemas de la movilidad urbana. El aprendizaje por refuerzo permite obtener controladores semafóricos adaptativos, pero suele requerir muchas interacciones con el simulador; los World Models proponen aprender primero un modelo de la dinámica del tráfico y entrenar o planificar dentro de él. Este trabajo compara tres arquitecturas para el bloque de dinámica de un mismo World Model —LSTM, TSMixer y Transformer— sobre una red urbana irregular de siete intersecciones simulada en SUMO, con el estado representado por 13 variables numéricas por intersección. Los tres modelos se entrenan con el mismo conjunto de datos (104 episodios, partición por semilla), la misma ventana de 60 s, una pérdida multi-paso común, un presupuesto de ≈150 000 parámetros (±10 %), 12 configuraciones de búsqueda y 5 semillas cada uno. En predicción, los tres superan con significancia estadística a la persistencia y a la regresión ridge; LSTM (RMSE 0,204 a un paso y 0,335 a 20 pasos) y TSMixer (0,205 y 0,336) no difieren entre sí y ambos superan al Transformer a 20 pasos (0,375; p_Holm < 0,001). En control, se evaluaron diez condiciones sobre los mismos escenarios de SUMO: las políticas PPO entrenadas solo dentro del modelo aprendido, sin pasos adicionales de simulador, quedan a la par del control de tiempo fijo (96–104 s de espera media frente a 101 s), mientras que añadir planificación por imaginación reduce la espera de WM + LSTM de 104 s a 85 s (p_Holm = 0,018). El PPO entrenado directamente en SUMO (78 s) y, sobre todo, el control actuado (48 s) siguen siendo superiores. Los resultados indican que la precisión a un paso no basta para seleccionar el modelo de un World Model de control: importan el error compuesto y la fidelidad del retorno imaginado (correlación 0,48–0,58 con el retorno real)."));
  add(P("**Palabras clave:** World Models, LSTM, TSMixer, Transformer, SUMO, control de semáforos, aprendizaje por refuerzo, PPO."));
  add(P("**Abstract**", { alignment: AlignmentType.LEFT }));
  add(P("Vehicular congestion is a major urban-mobility problem. Reinforcement learning yields adaptive traffic-signal controllers but usually needs many simulator interactions; World Models first learn a model of traffic dynamics and then train or plan inside it. This work compares three architectures for the dynamics block of the same World Model —LSTM, TSMixer and Transformer— on an irregular seven-intersection urban network simulated in SUMO, with the state described by 13 numerical variables per intersection. All models share the dataset (104 episodes, seed-based split), a 60 s window, a common multi-step loss, a budget of ≈150,000 parameters (±10 %), 12 search configurations and 5 seeds. In prediction, the three models significantly outperform persistence and ridge regression; LSTM (RMSE 0.204 one step, 0.335 at 20 steps) and TSMixer (0.205, 0.336) are statistically tied and both beat the Transformer at 20 steps (0.375; Holm-adjusted p < 0.001). In control, ten conditions were evaluated on the same SUMO scenarios: PPO policies trained only inside the learned model, with no additional simulator steps, perform on par with fixed-time control (96–104 s mean waiting time vs. 101 s), while adding imagination-based planning lowers the waiting time of WM + LSTM from 104 s to 85 s (p = 0.018). PPO trained directly in SUMO (78 s) and, above all, actuated control (48 s) remain better. One-step accuracy is not enough to choose the model of a control-oriented World Model: compounding error and the fidelity of imagined returns (correlation 0.48–0.58 with real returns) matter."));
  add(P("**Keywords:** World Models, LSTM, TSMixer, Transformer, SUMO, traffic signal control, reinforcement learning, PPO."));

  // ============================================================ I. Introducción
  add(H1("I. Introducción"));
  add(P("El control de semáforos es un problema de decisión secuencial en el que las acciones tomadas en una intersección afectan directamente la formación de colas, los tiempos de espera, la velocidad promedio y el flujo vehicular. El incremento sostenido del parque automotor y la interacción entre distintos flujos de tráfico agravan estos efectos, especialmente en redes compuestas por varias intersecciones interconectadas [1]."));
  add(P("Los esquemas de control de tiempo fijo no se adaptan a las variaciones que presenta el tráfico durante el día, por lo que el aprendizaje por refuerzo (Reinforcement Learning, RL) se ha consolidado como alternativa para obtener controladores adaptativos [2]. Sin embargo, cuando el agente aprende únicamente mediante interacción directa con el simulador, el proceso puede requerir una cantidad considerable de episodios para explorar situaciones de tráfico y aprender las consecuencias de sus acciones."));
  add(P("Los World Models constituyen una aproximación alternativa: el agente aprende un modelo interno de la dinámica del entorno y lo utiliza para anticipar estados futuros y evaluar acciones hipotéticas antes de ejecutarlas en el entorno real o simulado [3][4]. A diferencia de la formulación original de Ha y Schmidhuber, que opera sobre observaciones visuales comprimidas mediante un autoencoder variacional, este trabajo adapta la idea a un dominio numérico: el estado del tráfico se describe mediante variables como número de vehículos, longitud de cola, velocidad, ocupación, tiempo de espera y fase semafórica, obtenidas de SUMO (Simulation of Urban MObility) [19]."));
  add(P("El componente central del World Model —el modelo temporal que predice la evolución del estado— puede implementarse con arquitecturas de distinta naturaleza. En este trabajo se comparan tres alternativas bajo condiciones experimentales equivalentes: LSTM (Long Short-Term Memory) [23], red recurrente adoptada como referencia; TSMixer, una arquitectura basada exclusivamente en capas densas que alterna mezcla a lo largo del tiempo y entre variables, sin recurrencia ni atención [5]; y un Transformer con atención temporal [6]. Las tres comparten el pipeline de datos, la representación del estado, la ventana temporal, el presupuesto de parámetros, el protocolo de entrenamiento y la evaluación; el único componente que cambia es el bloque de modelo temporal."));
  add(P("Las contribuciones del trabajo son: (i) un escenario urbano realista de siete intersecciones y un conjunto de datos congelado y reproducible con partición por semilla; (ii) una comparación controlada de LSTM, TSMixer y Transformer como modelo de dinámica, con igual número de parámetros, igual presupuesto de búsqueda y cinco semillas por modelo; (iii) la evaluación del efecto de cada modelo sobre el control derivado —PPO entrenado en el entorno imaginado y planificación por imaginación— frente a tiempo fijo, control actuado, Max-Pressure y PPO directo sobre los mismos episodios de SUMO; y (iv) un análisis de la fidelidad del modelo (retorno imaginado frente a real y ranking de acciones), con pruebas de Wilcoxon pareadas y corrección de Holm."));
  add(P("El artículo se organiza así: la Sección II presenta los trabajos relacionados; la Sección III describe el entorno, los datos, la representación y los métodos; la Sección IV el diseño experimental; la Sección V los resultados; la Sección VI la discusión y la Sección VII las conclusiones."));

  // ============================================================ II. Trabajos relacionados
  add(H1("II. Trabajos Relacionados"));
  add(P("El control inteligente de semáforos ha evolucionado desde estrategias de tiempo fijo hacia métodos de aprendizaje por refuerzo, aprendizaje por refuerzo basado en modelos (Model-Based RL) y, más recientemente, World Models. En [3], Ha y Schmidhuber introducen la formulación original de los World Models, compuesta por un modelo de visión (autoencoder variacional), un modelo de memoria recurrente que predice la dinámica en el espacio latente y un controlador entrenado dentro del entorno aprendido. Hafner et al. [2] extienden esta idea con Dreamer, que aprende comportamientos mediante imaginación latente, mostrando que un agente puede entrenarse casi por completo dentro de un modelo del mundo aprendido."));
  add(P("En el dominio del control semafórico, Li et al. [7] aplican DreamerV3 al control de semáforos y estudian el ajuste de hiperparámetros. ModelLight [8] propone meta-aprendizaje por refuerzo basado en modelos para generalizar entre configuraciones de intersección, y Jaggi et al. [9] muestran que los enfoques microscópicos de Model-Based RL generalizan mejor que los métodos libres de modelo. TrafficWise [10] utiliza World Models para obtener un control interpretable y generalizable, y un enfoque relacionado combina A2C con predicción de estados mediante LSTM [11]. Otros trabajos abordan la coordinación entre intersecciones: Sun et al. [12] integran predicción de flujo con redes de atención sobre grafos, Peng et al. [13] proponen un esquema jerárquico híbrido, Huang et al. [14] estudian la planificación bajo discrepancias de observación mediante World Models modulares y SocialLight [15] plantea cooperación distribuida para toda la red. Como referencias clásicas sin aprendizaje destacan el control actuado por detectores y Max-Pressure [20]."));
  add(P("Respecto al modelo temporal, la LSTM ha sido el componente recurrente estándar de los World Models, incluida la MDN-RNN de la propuesta original [3]. TSMixer [5] plantea bloques de perceptrones multicapa que mezclan información a lo largo del tiempo y de las variables, con resultados competitivos frente a modelos recurrentes y de atención en pronóstico de series temporales. Los Transformers [6] modelan dependencias de largo alcance mediante atención. Los trabajos anteriores adoptan en su mayoría una única arquitectura temporal, lo que impide aislar su aporte; aquí se comparan explícitamente las tres como bloque de dinámica de un mismo World Model, con los mismos datos, presupuesto y métricas."));

  // ============================================================ III. Métodos
  add(H1("III. Entorno, Datos y Métodos"));
  add(H2("A. Escenario de simulación"));
  add(P("El escenario *urban7* es una malla urbana irregular —no un corredor recto— con siete intersecciones semaforizadas de cuatro brazos (Figura 1): una avenida principal que zigzaguea (J0–J3; 2+2 carriles, 50 km/h), una calle colectora no paralela al norte (J4–J6; 2+2 carriles, 40 km/h), conectores diagonales entre ambas y calles locales de distinto largo y ángulo (1+1 carril, 30 km/h), con 12 entradas/salidas y bolsillos de giro a la izquierda de 60 m. Cada intersección tiene cuatro fases verdes (norte–sur recto y derecha, norte–sur izquierda, este–oeste recto y derecha, este–oeste izquierda), seguidas de 3 s de amarillo y 1 s de todo rojo. La demanda es de tipo Poisson por ruta, constante a trozos cada 300 s, con cinco niveles (Tabla 1); D5 se reserva como escenario fuera de distribución (OOD)."));
  add(figure("urban7_network.png", "Red urbana irregular urban7 en SUMO: siete intersecciones semaforizadas (J0–J6), avenida principal, calle colectora, conectores diagonales y calles locales."));
  add(table(["Demanda", "Vehículos insertados", "Espera media (s)", "Viaje medio (s)", "Uso"],
    [["D1 Baja", "2 458", "67,8", "171,8", "train / val / test"],
     ["D2 Media", "4 719", "77,2", "186,3", "train / val / test"],
     ["D3 Alta", "6 782", "162,2", "290,1", "train / val / test"],
     ["D4 Pico", "4 960", "118,5", "235,9", "train / val / test"],
     ["D5 OOD", "5 577", "91,8", "204,8", "solo test-OOD y evaluación"]],
    [1.3, 1.5, 1.3, 1.3, 2.2],
    "Niveles de demanda del escenario (validación con tiempo fijo, 1 h simulada, sin teletransportes ni colisiones). D3 queda cerca de la saturación: 347 vehículos no alcanzan a entrar."));

  add(H2("B. Estado, acciones y recompensa"));
  add(P("El entorno sigue la interfaz de Gymnasium con un paso de decisión de Δt = 5 s, 300 s de calentamiento y episodios de 3 600 s (720 pasos). El estado de cada intersección tiene 13 variables: vehículos, vehículos detenidos, longitud de cola (m), velocidad media, ocupación, tiempo de espera acumulado, flujo de salida, fase activa codificada en one-hot (4), tiempo en la fase e indicador de amarillo; el estado conjunto es sₜ ∈ ℝ⁷ˣ¹³ = ℝ⁹¹. La acción es binaria por intersección (mantener la fase o pasar a la siguiente), aₜ ∈ {0,1}⁷, y el entorno aplica verdes mínimos y máximos (10 y 60 s), por lo que se registra la acción efectiva. La recompensa de cada intersección penaliza el aumento de la espera acumulada:"));
  add(EQ("rₜⁱ = −(Wⁱₜ₊₁ − Wⁱₜ) / 100"));

  add(H2("C. Conjunto de datos"));
  add(P("El conjunto *base_v2* se construyó una sola vez y se congeló (checksums, archivos de solo lectura y registro de la versión del código). Contiene 104 episodios generados con cuatro políticas de recolección para cubrir comportamientos diversos: tiempo fijo (ciclo de 90 s), aleatoria restringida (probabilidad de cambio 0,2 o 0,5), actuada de SUMO y Max-Pressure con exploración ε = 0,1. La partición se hace por semilla de simulación, nunca por ventana, para evitar fugas entre conjuntos (Tabla 2). Las variables se normalizan con z-score por intersección y variable usando solo el conjunto de entrenamiento. Cada ejemplo es una ventana de W = 12 pasos (60 s) de estados y acciones, x ∈ ℝ¹²ˣ⁹⁸, y el objetivo es el incremento del estado Δsₜ₊₁ = sₜ₊₁ − sₜ (91) y la recompensa rₜ (7)."));
  add(table(["Partición", "Semillas", "Demandas", "Episodios", "Transiciones"],
    [["Entrenamiento", "1–4", "D1–D4", "64", "46 080"],
     ["Validación", "5", "D1–D4", "16", "11 520"],
     ["Prueba", "6", "D1–D4", "16", "11 520"],
     ["Prueba OOD", "7–8", "D5", "8", "5 760"],
     ["Evaluación de control", "9–10", "D1–D5", "(no forman parte del dataset)", "—"]],
    [1.6, 1.0, 1.0, 1.8, 1.2],
    "Partición del conjunto de datos base_v2 (4 políticas por demanda y semilla; 720 pasos por episodio)."));

  add(H2("D. Representación del estado (Experimento 0)"));
  add(P("Antes de entrenar los modelos temporales se evaluó si comprimir el estado aporta valor. Se entrenaron un autoencoder (AE) y un autoencoder variacional (VAE) con espacios latentes de 16 y 32 dimensiones, y un modelo temporal de referencia sobre z(AE), z(VAE) y sobre el estado crudo normalizado. Con el RMSE a 20 pasos por episodio de validación, el estado crudo resultó mejor que el AE (Wilcoxon pareado, p_Holm = 0,016, δ de Cliff = −0,17) y que el VAE (p_Holm = 1,2·10⁻⁴, δ = −0,59). Por ello los tres modelos temporales trabajan directamente sobre el estado normalizado; la compresión latente, indispensable con imágenes, no se justifica con un estado numérico de 91 variables."));

  add(H2("E. Arquitectura general del World Model"));
  add(P("El flujo experimental es: SUMO → estado del tráfico → normalización → modelo temporal → predicción del siguiente estado y de la recompensa → imaginación → controlador → acción semafórica → SUMO (Figura 2). El bloque de modelo temporal es el único que se sustituye entre experimentos; datos, representación, ventana, controlador y métricas permanecen constantes (Tabla 3)."));
  add(figure("image1.png", "Arquitectura general del World Model para el control de semáforos en SUMO. Fase 1: aprendizaje del World Model. Fase 2: entrenamiento del controlador PPO en el entorno imaginado (Dream Environment) y evaluación final en SUMO. El autoencoder es opcional y, según el Experimento 0, no se utiliza."));
  add(table(["Componente", "Entrada", "Salida", "Propósito"],
    [["SUMO + TraCI/libsumo", "Red, demanda y acción", "Estado del tráfico", "Simular la red urbana y ejecutar las acciones de control."],
     ["Representación", "Estado 7×13", "Vector normalizado (91)", "z-score por intersección y variable (sin compresión latente)."],
     ["Modelo temporal (LSTM / TSMixer / Transformer)", "Ventana de 12 estados y acciones", "Δŝₜ₊₁ y r̂ₜ", "Aprender la dinámica; único bloque que cambia entre experimentos."],
     ["Dream Environment", "Predicciones del modelo", "Trayectorias imaginadas", "Entrenar PPO y evaluar acciones sin consultar SUMO."],
     ["Controlador (PPO, Stable-Baselines3)", "Estado normalizado", "Acción semafórica", "Seleccionar la acción; opcionalmente con planificación."]],
    [2.2, 1.6, 1.6, 3.0],
    "Componentes del World Model y su función en el pipeline experimental."));

  add(H2("F. Modelos temporales"));
  add(P("Los tres modelos comparten la interfaz f_θ(xₜ₋₁₁…xₜ) = (Δŝₜ₊₁, r̂ₜ) y el siguiente estado se obtiene como ŝₜ₊₁ = sₜ + Δŝₜ₊₁. Para que la comparación sea justa, cada arquitectura tiene una dimensión libre que se ajusta automáticamente para que el total de parámetros quede en 150 000 ± 10 %: el tamaño oculto en la LSTM y el ancho del MLP interno en TSMixer y Transformer (Tabla 4)."));
  add(P("**LSTM.** Dos capas LSTM seguidas de una capa densa sobre el último estado oculto, sin componentes probabilísticos (Mixture Density Network) (Figura 3). Con dos capas, el presupuesto fija un tamaño oculto de 93."));
  add(figure("image2.png", "Arquitectura de una red LSTM: capas de celdas LSTM (compuertas de olvido, entrada y salida) y capa densa de salida.", 520));
  add(P("**TSMixer.** Una proyección de entrada seguida de N bloques que alternan mezcla temporal (una capa densa sobre el eje de los 12 pasos) y mezcla de variables (MLP sobre el eje de características), con normalización y conexiones residuales; al final, una proyección temporal 12 → 1 y una cabeza densa (Figura 4). No usa recurrencia ni atención."));
  add(figure("image3.png", "Arquitectura TSMixer: bloques de mezcla temporal y mezcla de variables basados en MLP, con normalización y conexiones residuales.", 520));
  add(P("**Transformer.** Embedding lineal de cada paso más codificación posicional sinusoidal, N capas de codificador con normalización previa (pre-norm), atención multi-cabeza con máscara causal y MLP, y una cabeza densa sobre el último token. Solo se usa la parte de codificador de la arquitectura de la Figura 5: no hay decodificador porque la salida es una única predicción por ventana."));
  add(figure("image4.png", "Arquitectura Transformer de referencia [6]. En este trabajo se usa únicamente el codificador (atención causal sobre la ventana de 12 pasos) y una cabeza densa sobre el último token.", 520));
  add(P("La Figura 6 muestra las tres arquitecturas finales, tras la búsqueda de hiperparámetros, con sus dimensiones reales."));
  add(figure("arquitecturas_implementadas.png", "Arquitecturas implementadas (configuración final de cada modelo). Entrada y salida son idénticas; solo cambia el bloque central, con ≈150 000 parámetros en los tres casos."));
  add(table(["Modelo", "Configuración final", "Dimensión ajustada", "Parámetros", "lr / weight decay"],
    [["LSTM", "2 capas, dropout 0", "oculto = 93", "150 944", "3·10⁻³ / 10⁻⁴"],
     ["TSMixer", "6 bloques, d = 64, BatchNorm, dropout 0,1", "ff = 174", "150 251", "3·10⁻³ / 0"],
     ["Transformer", "4 capas, d = 64, 4 cabezas, dropout 0", "ff = 134", "149 818", "3·10⁻³ / 10⁻³"]],
    [1.2, 2.8, 1.5, 1.1, 1.6],
    "Configuración elegida para cada modelo por búsqueda aleatoria (12 configuraciones, la mejor pérdida de validación)."));

  add(H2("G. Protocolo de entrenamiento común"));
  add(P("Los tres modelos se entrenan con el mismo protocolo. La pérdida combina el error del estado y el de la recompensa con λ = 7/91, que da el mismo peso a cada una de las 98 salidas, y se calcula de forma multi-paso: el modelo avanza k = 5 pasos reutilizando sus propias predicciones como entrada (con las acciones reales), lo que expone el error compuesto durante el entrenamiento:"));
  add(EQ("L = (1/k) · Σⱼ₌₁ᵏ [ MSE(ŝₜ₊ⱼ, sₜ₊ⱼ) + λ · MSE(r̂ₜ₊ⱼ₋₁, rₜ₊ⱼ₋₁) ],   k = 5,   λ = 7/91"));
  add(P("Se usa AdamW, lotes de 256 ventanas, épocas de 11 040 ventanas muestreadas de entrenamiento, hasta 200 épocas y parada temprana con paciencia 10 sobre la pérdida de validación. El presupuesto de ajuste es idéntico: 12 configuraciones por modelo, muestreadas al azar con la misma semilla y entrenadas 15 épocas, y la mejor se reentrena con 5 semillas (0–4). Todo se ejecutó en CPU (AMD Ryzen 5 3500U, 4 núcleos, 6 GB de RAM, PyTorch 2.2)."));

  add(H2("H. Dream Environment, PPO y planificación"));
  add(P("El Dream Environment reemplaza a SUMO por el modelo temporal: se reinicia con una ventana real de 12 pasos tomada del conjunto de entrenamiento y avanza con el modelo, aplicando las mismas reglas del semáforo que el entorno real (acción efectiva, verdes mínimo y máximo) y proyectando el estado imaginado a rangos físicos válidos (valores no negativos, fase one-hot). Los episodios imaginados duran 40 pasos (200 s) para limitar el error compuesto, y 16 episodios avanzan en paralelo con una sola pasada del modelo. Sobre él se entrena PPO [17] (Stable-Baselines3 [18]) con los mismos hiperparámetros para los tres World Models y para el PPO directo en SUMO (Tabla 5). La política de la semilla k se entrena en el sueño del modelo de la semilla k."));
  add(P("La planificación por imaginación es complementaria: en cada decisión se imaginan 16 acciones conjuntas candidatas (mantener todo, la acción propuesta por la política PPO del sueño y 14 aleatorias) durante 5 pasos con el modelo temporal, y se ejecuta la de mayor retorno imaginado."));
  add(table(["Parámetro", "Valor"],
    [["Tasa de aprendizaje / γ / λ_GAE", "3·10⁻⁴ / 0,99 / 0,95"],
     ["Transiciones por actualización / lote / épocas", "2 048 / 256 / 10"],
     ["Clip / coef. entropía / coef. valor", "0,2 / 0,01 / 0,5"],
     ["Red (política y valor)", "MLP 128–128; observación: estado normalizado (91)"],
     ["Recompensa de PPO", "Σᵢ rₜⁱ / 10 (igual en sueño y en SUMO)"],
     ["PPO en el sueño", "400 000 pasos imaginados; 0 pasos de SUMO; 5 semillas × 3 modelos"],
     ["PPO directo en SUMO", "57 600 pasos de SUMO (D1–D4, semillas 1–4); 3 semillas"],
     ["Planificación", "16 candidatas, horizonte 5 pasos"]],
    [3.0, 5.0],
    "Hiperparámetros de PPO, idénticos en el entorno imaginado y en SUMO."));

  // ============================================================ IV. Diseño experimental
  add(H1("IV. Diseño Experimental"));
  add(P("El diseño aísla el efecto del modelo temporal sobre la predicción y sobre el control. Las cinco condiciones del plan —tiempo fijo, RL directo y World Model con LSTM, TSMixer y Transformer— se complementan con dos referencias adaptativas sin aprendizaje (control actuado de SUMO y Max-Pressure) y con la variante con planificación de cada World Model, para un total de diez condiciones (Tabla 6). No se asume de antemano cuál arquitectura será superior."));
  add(table(["Condición", "Modelo temporal", "Pasos de SUMO para aprender", "Semillas"],
    [["Tiempo fijo (ciclo 90 s)", "—", "0", "—"],
     ["Actuado (SUMO, verde 10–60 s)", "—", "0", "—"],
     ["Max-Pressure", "—", "0", "—"],
     ["PPO directo", "—", "57 600 por semilla", "3"],
     ["WM + LSTM / TSMixer / Transformer", "el indicado", "57 600 (dataset) + 0", "5 cada uno"],
     ["WM + modelo + planificación", "el indicado", "57 600 (dataset) + 0", "5 cada uno"]],
    [2.8, 1.5, 2.0, 1.2],
    "Condiciones evaluadas en SUMO. Los World Models solo usan las transiciones del conjunto de datos (entrenamiento y validación)."));
  add(P("**Métricas de predicción.** RMSE del estado normalizado a un paso y en rollout autorregresivo hasta h = 20 pasos (100 s) con las acciones reales, área bajo la curva RMSE(h), RMSE de la recompensa, exactitud de la fase predicha y RMSE en régimen congestionado (cola mayor que el percentil 75 de entrenamiento). Se incluyen cuatro baselines: persistencia, media móvil, regresión ridge sobre la ventana aplanada y un MLP sin estructura temporal entrenado con el mismo protocolo. También se mide el costo de entrenamiento y de inferencia."));
  add(P("**Métricas de control.** Cada condición se ejecuta sobre los mismos episodios de SUMO: demandas D1–D5 con las semillas 9 y 10, nunca vistas durante el entrenamiento (10 escenarios). Las condiciones aprendidas se evalúan con cada una de sus semillas de entrenamiento y se promedian por escenario (360 episodios en total). Se reportan tiempo de espera y de viaje por vehículo, cola media y máxima, throughput, paradas, CO₂, recompensa y teletransportes."));
  add(P("**Fidelidad.** Se compara el retorno imaginado a 40 pasos con el real sobre 64 segmentos de test con las acciones registradas, y el ranking de 10 acciones candidatas imaginado a 5 pasos con el obtenido en SUMO, en 15 estados de decisión (τ de Kendall)."));
  add(P("**Estadística.** Las comparaciones usan la prueba de Wilcoxon pareada por episodio (predicción, n = 16) o por escenario (control, n = 10), con corrección de Holm [21] para comparaciones múltiples, δ de Cliff [22] como tamaño de efecto e intervalos de confianza del 95 % por bootstrap (10 000 remuestreos)."));

  // ============================================================ V. Resultados
  add(H1("V. Resultados"));
  add(H2("A. Precisión predictiva"));
  add(P("Los tres modelos temporales superan a todos los baselines simples (Tabla 7). Frente a ridge, la mejora es significativa a un paso (p_Holm ≤ 0,002, δ entre −0,41 y −0,44) y a 20 pasos (p_Holm ≤ 0,003, δ entre −0,45 y −0,59); frente a la persistencia, δ = −1 en todos los casos. El MLP sin estructura temporal es tan preciso como los modelos temporales a un paso (0,203), pero su error crece más rápido con el horizonte: a 20 pasos LSTM y TSMixer lo superan de forma significativa (p_Holm < 0,001, δ ≈ −0,43), mientras que el Transformer queda empatado con él (p_Holm = 0,9)."));
  add(table(["Modelo", "RMSE h=1", "RMSE h=5", "RMSE h=20", "AUC RMSE(h)", "RMSE recompensa", "Exactitud de fase"],
    [["LSTM", "**0,204** ± 0,002", "**0,265** ± 0,003", "**0,335** ± 0,003", "**5,58**", "1,95", "0,982"],
     ["TSMixer", "0,205 ± 0,003", "0,267 ± 0,004", "0,336 ± 0,006", "5,60", "**1,92**", "0,980"],
     ["Transformer", "0,208 ± 0,004", "0,283 ± 0,004", "0,375 ± 0,008", "6,05", "1,99", "0,979"],
     ["MLP", "0,203", "0,267", "0,374", "5,88", "1,94", "0,981"],
     ["Ridge", "0,245", "0,361", "0,442", "7,47", "2,25", "0,886"],
     ["Media móvil", "0,561", "0,619", "0,605", "11,14", "2,96", "0,193"],
     ["Persistencia", "0,477", "0,786", "0,740", "13,86", "2,96", "0,748"]],
    [1.3, 1.3, 1.3, 1.3, 1.1, 1.2, 1.1],
    "Precisión predictiva en el conjunto de prueba (estado y recompensa normalizados). Modelos temporales: media ± desviación estándar sobre 5 semillas."));
  add(P("La Figura 7 muestra el error autorregresivo según el horizonte. LSTM y TSMixer son indistinguibles en todo el horizonte (a 20 pasos p_Holm = 0,30, δ = −0,05), mientras que el Transformer empieza a separarse a partir de h ≈ 3 y a 20 pasos es claramente peor que ambos (p_Holm = 6·10⁻⁵, δ = −0,48 frente a LSTM y −0,45 frente a TSMixer). A un paso las diferencias entre modelos son estadísticamente detectables pero despreciables (|δ| ≤ 0,12). En el escenario fuera de distribución D5 el orden se mantiene (RMSE a 20 pasos: LSTM 0,348, TSMixer 0,355, Transformer 0,394, ridge 0,470), lo que indica que ningún modelo sobreajusta a las demandas de entrenamiento."));
  add(figure("rmse_vs_horizonte.png", "Error autorregresivo (RMSE del estado normalizado) en el conjunto de prueba según el horizonte de predicción. Las bandas muestran ±1 desviación estándar sobre 5 semillas; las líneas discontinuas son los baselines."));

  add(H2("B. Costo computacional"));
  add(P("Con el mismo número de parámetros, el costo difiere de forma notable (Tabla 8). La LSTM es la más barata en entrenamiento (13 s por época y 20 min por semilla) y en inferencia (0,8 ms por paso y 6 164 pasos imaginados por segundo en lote de 64), gracias a que PyTorch implementa la recurrencia con núcleos optimizados. TSMixer, pese a ser un modelo solo de capas densas, fue el más lento en esta implementación (6 bloques con BatchNorm y proyecciones temporales). La Figura 8 muestra las curvas de validación: la LSTM converge en menos épocas y con menor variabilidad entre semillas."));
  add(table(["Modelo", "s / época", "min / semilla", "RSS pico (MB)", "Latencia 1 paso (ms)", "Rollout 40 pasos (ms)", "Pasos imaginados / s (lote 64)"],
    [["LSTM", "12,9", "20,0", "660", "0,84", "40", "6 164"],
     ["TSMixer", "39,5", "69,6", "852", "2,31", "107", "3 484"],
     ["Transformer", "31,8", "48,3", "639", "1,28", "58", "3 226"]],
    [1.3, 1.0, 1.1, 1.1, 1.3, 1.3, 1.6],
    "Costo computacional medido en el mismo equipo (CPU, 1 hilo en inferencia)."));
  add(figure("entrenamiento_modelos_temporales.png", "Pérdida de validación por época para las 5 semillas finales de cada modelo (misma escala vertical)."));

  add(H2("C. Aprendizaje en el entorno imaginado"));
  add(P("Las 15 políticas PPO entrenadas en el sueño (3 modelos × 5 semillas) aumentan de forma sostenida su retorno imaginado hasta estabilizarse a partir de unos 200 000 pasos (Figura 9, derecha), y cada una tardó entre 6 y 8 minutos. El retorno imaginado final fue 4,3 ± 2,0 (LSTM), 3,6 ± 0,9 (TSMixer) y 3,9 ± 1,0 (Transformer); este valor mide qué tan bien la política explota el modelo, no su desempeño real. El PPO directo (Figura 9, izquierda) consume 57 600 pasos de SUMO por semilla (≈17 min) con una curva mucho más ruidosa; tarda en promedio 34 080 pasos en alcanzar el 90 % de su mejora final. Los World Models, en cambio, no consumen pasos de simulador adicionales a los del conjunto de datos, que además se comparte entre los tres modelos y todas las semillas."));
  add(figure("curvas_ppo.png", "Izquierda: retorno por episodio del PPO directo frente a los pasos de SUMO consumidos (3 semillas). Derecha: retorno imaginado de PPO en el sueño de cada World Model (media ± desviación sobre 5 semillas), sin pasos de SUMO."));

  add(H2("D. Desempeño del control en SUMO"));
  add(P("La Tabla 9 resume las diez condiciones sobre los mismos diez escenarios de evaluación y la Figura 10 desglosa la espera por demanda. El control actuado es claramente el mejor (48,0 s de espera, 156,6 s de viaje) y supera a todas las condiciones con World Model (p_Holm = 0,018, δ entre 0,92 y 1,0). El PPO directo reduce la espera respecto al tiempo fijo de 100,9 s a 77,9 s (p_Holm = 0,002, δ = −0,42)."));
  add(table(["Condición", "Espera (s)", "Viaje (s)", "Cola media (m)", "Throughput", "CO₂ (g/veh)"],
    [["Tiempo fijo", "100,9 [81,9; 121,9]", "214,9", "142,3", "4 449", "525"],
     ["Actuado", "**48,0** [43,2; 53,0]", "**156,6**", "**66,8**", "**4 576**", "**373**"],
     ["Max-Pressure", "82,6 [75,8; 89,8]", "196,2", "111,2", "4 557", "479"],
     ["PPO directo", "77,9 [61,9; 95,7]", "191,8", "110,1", "4 506", "466"],
     ["WM + LSTM", "103,9 [95,4; 112,4]", "219,7", "137,9", "4 366", "538"],
     ["WM + TSMixer", "96,6 [78,2; 115,6]", "214,4", "132,1", "4 373", "524"],
     ["WM + Transformer", "96,2 [83,0; 109,1]", "212,0", "131,5", "4 430", "518"],
     ["WM + LSTM + planificación", "85,1 [78,1; 92,2]", "196,7", "115,7", "4 554", "479"],
     ["WM + TSMixer + planificación", "93,2 [80,4; 107,4]", "206,4", "128,7", "4 506", "504"],
     ["WM + Transformer + planificación", "99,3 [85,4; 113,4]", "212,9", "139,8", "4 422", "520"]],
    [2.6, 1.8, 1.0, 1.1, 1.0, 1.0],
    "Control en SUMO: media sobre 10 escenarios (D1–D5 × semillas 9 y 10), con IC 95 % bootstrap para la espera. Las condiciones aprendidas promedian sus semillas de entrenamiento en cada escenario."));
  add(P("Las políticas entrenadas únicamente en el sueño quedan a la par del tiempo fijo: WM + TSMixer (96,6 s) y WM + Transformer (96,2 s) lo mejoran ligeramente y WM + LSTM (103,9 s) queda ligeramente por encima, pero ninguna diferencia es significativa (p_Holm ≥ 0,83) y tampoco lo son las diferencias entre los tres World Models (p_Holm ≥ 0,26). La planificación por imaginación mejora a la política del sueño con la LSTM y con TSMixer (96,6 → 93,2 s, no significativo), pero no con el Transformer (96,2 → 99,3 s). Con la LSTM la mejora es significativa: WM + LSTM + planificación reduce la espera de 103,9 s a 85,1 s (p_Holm = 0,018, δ = −0,70), queda a solo 2,5 s de Max-Pressure (p_Holm = 0,08) y a 7 s del PPO directo (p_Holm = 0,49). Por demanda (Figura 10), las mayores ganancias de la planificación aparecen en D3, la demanda cercana a la saturación, donde WM + LSTM + planificación (102,8 s) iguala a Max-Pressure (102,0 s) y mejora en 55 s al tiempo fijo (158,3 s). En D1 y D2, con poca congestión, las diferencias frente al tiempo fijo son pequeñas y de signo variable."));
  add(figure("control_espera_por_demanda.png", "Tiempo de espera medio por vehículo en SUMO por demanda (media de las semillas de evaluación 9 y 10). Las barras rayadas son las variantes con planificación por imaginación; cada World Model conserva su color."));

  add(H2("E. Fidelidad del World Model"));
  add(P("La Tabla 10 cuantifica qué tan bien predice cada modelo el retorno que importa para el control. Con las acciones registradas, la correlación entre el retorno imaginado a 40 pasos y el real es moderada (Pearson 0,48–0,58) y la brecha absoluta media es grande respecto al retorno medio real (−8,6): el Transformer es el más optimista (retorno imaginado −3,5) y TSMixer el más pesimista (−10,3). En cambio, para ordenar acciones a 5 pasos los tres modelos son bastante fiables: τ de Kendall de 0,60–0,64 y τ > 0 en el 93–100 % de los estados; TSMixer acierta la mejor acción en el 77 % de los estados."));
  add(table(["Modelo", "Pearson (40 pasos)", "Spearman", "Brecha media", "Brecha abs. media", "τ de Kendall (h=5)", "Estados con τ > 0", "Acierto top-1"],
    [["LSTM", "**0,58**", "**0,49**", "+2,5", "**22,2**", "0,64", "93 %", "67 %"],
     ["TSMixer", "0,52", "0,35", "**−1,7**", "23,2", "**0,64**", "93 %", "**77 %**"],
     ["Transformer", "0,48", "0,40", "+5,0", "24,2", "0,60", "**100 %**", "68 %"]],
    [1.2, 1.1, 1.0, 1.0, 1.1, 1.1, 1.1, 1.0],
    "Fidelidad del World Model (media sobre 5 semillas). Retorno medio real de los segmentos: −8,6."));

  // ============================================================ VI. Discusión
  add(H1("VI. Discusión"));
  add(P("**Qué modelo temporal elegir.** En predicción, LSTM y TSMixer son equivalentes y ambos superan al Transformer en horizontes largos, que es lo que usa la imaginación. La ventaja de la atención para dependencias de largo alcance no se materializa con ventanas de 12 pasos y ≈150 000 parámetros; con la máscara causal y pocos datos por ventana, el Transformer acumula más error compuesto. Considerando también el costo —la LSTM entrena 3,5 veces más rápido que TSMixer y genera casi el doble de pasos imaginados por segundo— y la mejor fidelidad del retorno, la LSTM ofrece la mejor relación precisión–costo en este escenario, seguida de cerca por TSMixer."));
  add(P("**Por qué las políticas del sueño no superan al tiempo fijo.** Que un modelo tenga poco error a un paso no garantiza que la política entrenada dentro de él funcione en el simulador. PPO maximiza el retorno imaginado y explota los errores sistemáticos del modelo; la correlación moderada entre retorno imaginado y real (≈0,5) y el sesgo de cada modelo (optimista en el Transformer, pesimista en TSMixer) son consistentes con esa brecha entre sueño y realidad. Además, el conjunto de datos proviene de cuatro políticas de recolección, por lo que el modelo es menos fiable en las regiones del espacio de estados que visita una política nueva. La planificación mitiga el problema porque solo usa el modelo a 5 pasos, donde el error es menor y el ranking de acciones es fiable (τ ≈ 0,6), y porque vuelve a partir del estado real en cada decisión. Esto es coherente con que la mejora más clara aparezca con la LSTM, el modelo con mejor fidelidad del retorno, y con que el Transformer —el de mayor error a varios pasos y retorno imaginado más sesgado— no se beneficie de la planificación."));
  add(P("**Eficiencia muestral.** El atractivo de los World Models se mantiene: WM + LSTM + planificación se acerca al nivel de Max-Pressure sin consumir un solo paso de SUMO adicional al conjunto de datos, que además se reutiliza para los tres modelos y las 15 políticas, mientras que el PPO directo necesita 57 600 pasos de simulador por semilla. Sin embargo, con este presupuesto el PPO directo sigue siendo mejor, y el control actuado —que usa detectores y conocimiento del sistema— es una referencia difícil de superar en esta red."));
  add(P("**Limitaciones.** (i) La evaluación de control usa 2 semillas por demanda (10 escenarios) en lugar de las 10 previstas, por costo de cómputo en CPU; con n = 10 el menor p alcanzable tras la corrección de Holm es ≈0,018, por lo que diferencias moderadas (p. ej. WM + LSTM + planificación frente a Max-Pressure) pueden no detectarse. (ii) El PPO directo usa 3 semillas en lugar de 5. (iii) Se estudia una sola red de siete intersecciones y demandas sintéticas. (iv) Los modelos son deterministas; no se modela la incertidumbre de la predicción, que podría usarse para penalizar la explotación de errores del modelo. (v) Todas las mediciones de costo corresponden a un único equipo sin GPU."));

  // ============================================================ VII. Conclusiones
  add(H1("VII. Conclusiones"));
  add(P("Se comparó LSTM, TSMixer y Transformer como bloque de dinámica de un mismo World Model para el control de siete semáforos en SUMO, con igual número de parámetros, igual presupuesto de búsqueda y cinco semillas por modelo. Los tres superan a los baselines simples en predicción. LSTM y TSMixer son estadísticamente equivalentes y ambos superan al Transformer en horizontes largos; la LSTM es además la más económica en entrenamiento e inferencia y la que mejor predice el retorno."));
  add(P("En control, las políticas PPO entrenadas solo dentro del modelo aprendido rinden como el control de tiempo fijo, y la planificación por imaginación con la LSTM mejora significativamente a su política base hasta acercarse a Max-Pressure, sin pasos de simulador adicionales. El PPO entrenado directamente en SUMO y, sobre todo, el control actuado siguen siendo superiores. La lección principal es que la precisión a un paso no basta para elegir el modelo de un World Model de control: el error compuesto y la fidelidad del retorno imaginado determinan cuánto sirve el modelo para decidir."));
  add(P("Como trabajo futuro se plantea: ampliar la evaluación a las semillas 9–18 para aumentar la potencia estadística; modelos probabilísticos o ensambles que estimen la incertidumbre y penalicen su explotación; recolectar datos de forma iterativa con la propia política del sueño (estilo Dreamer) para cerrar la brecha entre sueño y realidad; episodios imaginados más cortos o combinados con planificación durante el entrenamiento; e incorporar la estructura de la red mediante grafos, redes de mayor tamaño y datos reales de tráfico."));
  add(P("**Disponibilidad.** El código, la configuración, el conjunto de datos (con su manifiesto de checksums) y todos los resultados de este artículo están versionados en el repositorio del proyecto *world-model-semaforos-sumo*; cada tabla y figura se regenera con scripts/run_pipeline.py."));

  // ============================================================ Referencias
  add(H1("Referencias"));
  const refs = [
    "MinSalud Colombia y otros organismos de movilidad urbana, informes sobre congestión vehicular y su impacto en la eficiencia de las redes viales, 2018-2022.",
    "D. Hafner, T. Lillicrap, J. Ba y M. Norouzi, \"Dream to Control: Learning Behaviors by Latent Imagination,\" 2020.",
    "D. Ha y J. Schmidhuber, \"World Models,\" 2018.",
    "R. S. Sutton y A. G. Barto, Reinforcement Learning: An Introduction, MIT Press, 2018.",
    "S.-A. Chen et al., \"TSMixer: An All-MLP Architecture for Time Series Forecasting,\" 2023.",
    "A. Vaswani et al., \"Attention Is All You Need,\" Advances in Neural Information Processing Systems, 2017.",
    "Q. Li, Y. Lin, Q. Luo y L. Yu, \"DreamerV3 for Traffic Signal Control: Hyperparameter Tuning and Performance,\" Advances in Transdisciplinary Engineering, DOI: 10.3233/atde250554, 2025.",
    "X. Huang, D. Wu, M. Jenkin y B. Boulet, \"ModelLight: Model-Based Meta-Reinforcement Learning for Traffic Signal Control,\" 2021.",
    "P. Jaggi, X. Wang, N. Carrara, S. Sanner y B. Abdulhai, \"Microscopic Model-Based RL Approaches for Traffic Signal Control Generalize Better than Model-Free RL Approaches,\" DOI: 10.1109/ITSC48978.2021.9564528, 2021.",
    "J. Hu, X. Li, C. Ye, Y. Zhang, J. Sun y H. Zhao, \"TrafficWise: Leveraging World Models for Generalized and Interpretable Traffic Control,\" IEEE Intelligent Transportation Systems Magazine, DOI: 10.1109/mits.2025.3543446.",
    "\"Traffic Signal Control Based on Deep Reinforcement Learning A2C Integrated with State Prediction,\" DOI: 10.19678/j.issn.1000-3428.0069478, 2025.",
    "C. Sun, Y. Yang, J. Li, W. Fang y P. Zhang, \"A Multi-Agent Regional Traffic Signal Control System Integrating Traffic Flow Prediction and Graph Attention Networks,\" Systems, DOI: 10.3390/systems14010047, 2025.",
    "X. Peng, S. Chen y H. M. Zhang, \"A Hierarchical Signal Coordination and Control System Using a Hybrid Model-based and Reinforcement Learning Approach,\" DOI: 10.48550/arxiv.2508.20102, 2025.",
    "Z. Huang, Y. Liu, C. Liang y G. Zheng, \"Planning Under Observation Mismatch for Traffic Signal Control via Adaptive Modular World Models,\" arXiv:2501.02548, 2025.",
    "H. Goel, Y. Zhang, M. Damani y G. Sartoretti, \"SocialLight: Distributed Cooperation Learning towards Network-Wide Traffic Signal Control,\" DOI: 10.65109/gifg9402, 2023.",
    "X. Huang, D. Wu, M. Jenkin y B. Boulet, \"Sample-Efficient Meta-RL for Traffic Signal Control,\" DOI: 10.1109/ccece59415.2024.10667211, 2024.",
    "J. Schulman, F. Wolski, P. Dhariwal, A. Radford y O. Klimov, \"Proximal Policy Optimization Algorithms,\" arXiv:1707.06347, 2017.",
    "A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus y N. Dormann, \"Stable-Baselines3: Reliable Reinforcement Learning Implementations,\" Journal of Machine Learning Research, vol. 22, no. 268, 2021.",
    "P. A. Lopez et al., \"Microscopic Traffic Simulation using SUMO,\" IEEE International Conference on Intelligent Transportation Systems (ITSC), 2018.",
    "P. Varaiya, \"Max pressure control of a network of signalized intersections,\" Transportation Research Part C, vol. 36, pp. 177–195, 2013.",
    "S. Holm, \"A Simple Sequentially Rejective Multiple Test Procedure,\" Scandinavian Journal of Statistics, vol. 6, no. 2, pp. 65–70, 1979.",
    "N. Cliff, \"Dominance statistics: Ordinal analyses to answer ordinal questions,\" Psychological Bulletin, vol. 114, no. 3, pp. 494–509, 1993.",
    "S. Hochreiter y J. Schmidhuber, \"Long Short-Term Memory,\" Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997.",
  ];
  refs.forEach((r, i) => add(new Paragraph({ children: [new TextRun({ text: `[${i + 1}] ${r}`, size: 20 })],
    alignment: AlignmentType.LEFT, spacing: { after: 80 }, indent: { left: 440, hanging: 440 } })));
  return out;
};
