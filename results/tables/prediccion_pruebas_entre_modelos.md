| horizonte   | modelo   | contra      |   mediana dif. RMSE |   p (Holm) |   Cliff δ | efecto       | modelo mejor   |
|:------------|:---------|:------------|--------------------:|-----------:|----------:|:-------------|:---------------|
| 1 paso      | TSMixer  | Transformer |           -0.00197  |  0.005371  |  -0.08594 | despreciable | sí             |
| 1 paso      | LSTM     | TSMixer     |           -0.001177 |  0.4037    |  -0.07031 | despreciable | sí             |
| 1 paso      | LSTM     | Transformer |           -0.003373 |  0.002625  |  -0.1172  | despreciable | sí             |
| 20 pasos    | TSMixer  | Transformer |           -0.03735  |  6.104e-05 |  -0.4531  | mediano      | sí             |
| 20 pasos    | LSTM     | TSMixer     |           -0.005218 |  0.2979    |  -0.04688 | despreciable | sí             |
| 20 pasos    | LSTM     | Transformer |           -0.04217  |  6.104e-05 |  -0.4766  | grande       | sí             |
