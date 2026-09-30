| horizonte   | modelo      | contra       |   mediana dif. RMSE |   p (Holm) |   Cliff δ | efecto       | modelo mejor   |
|:------------|:------------|:-------------|--------------------:|-----------:|----------:|:-------------|:---------------|
| 1 paso      | LSTM        | Persistencia |          -0.2744    |  0.0001221 |  -1       | grande       | sí             |
| 1 paso      | LSTM        | Media móvil  |          -0.367     |  0.0001221 |  -1       | grande       | sí             |
| 1 paso      | LSTM        | Ridge        |          -0.0404    |  0.001526  |  -0.4375  | mediano      | sí             |
| 1 paso      | LSTM        | MLP          |           0.0001347 |  0.4037    |   0.02344 | despreciable | no             |
| 20 pasos    | LSTM        | Persistencia |          -0.4021    |  0.0001221 |  -1       | grande       | sí             |
| 20 pasos    | LSTM        | Media móvil  |          -0.2846    |  0.0001221 |  -1       | grande       | sí             |
| 20 pasos    | LSTM        | Ridge        |          -0.1238    |  0.0001221 |  -0.5859  | grande       | sí             |
| 20 pasos    | LSTM        | MLP          |          -0.04684   |  0.0001221 |  -0.4453  | mediano      | sí             |
| 1 paso      | TSMixer     | Persistencia |          -0.2721    |  0.0001221 |  -1       | grande       | sí             |
| 1 paso      | TSMixer     | Media móvil  |          -0.3653    |  0.0001221 |  -1       | grande       | sí             |
| 1 paso      | TSMixer     | Ridge        |          -0.03592   |  0.00116   |  -0.4375  | mediano      | sí             |
| 1 paso      | TSMixer     | MLP          |           0.0008837 |  0.01309   |   0.07812 | despreciable | no             |
| 20 pasos    | TSMixer     | Persistencia |          -0.3946    |  0.0001221 |  -1       | grande       | sí             |
| 20 pasos    | TSMixer     | Media móvil  |          -0.2807    |  0.0001221 |  -1       | grande       | sí             |
| 20 pasos    | TSMixer     | Ridge        |          -0.1233    |  0.0005798 |  -0.5547  | grande       | sí             |
| 20 pasos    | TSMixer     | MLP          |          -0.04438   |  0.0003052 |  -0.4219  | mediano      | sí             |
| 1 paso      | Transformer | Persistencia |          -0.2727    |  0.0001221 |  -1       | grande       | sí             |
| 1 paso      | Transformer | Media móvil  |          -0.3642    |  0.0001221 |  -1       | grande       | sí             |
| 1 paso      | Transformer | Ridge        |          -0.03661   |  0.001678  |  -0.4062  | mediano      | sí             |
| 1 paso      | Transformer | MLP          |           0.003939  |  0.0001221 |   0.1406  | despreciable | no             |
| 20 pasos    | Transformer | Persistencia |          -0.3639    |  0.0001221 |  -1       | grande       | sí             |
| 20 pasos    | Transformer | Media móvil  |          -0.2422    |  0.0001221 |  -0.9922  | grande       | sí             |
| 20 pasos    | Transformer | Ridge        |          -0.07706   |  0.002625  |  -0.4531  | mediano      | sí             |
| 20 pasos    | Transformer | MLP          |          -0.002238  |  0.8999    |  -0.02344 | despreciable | sí             |
