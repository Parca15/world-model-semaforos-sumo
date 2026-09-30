| métrica        | referencia                       | contra                           |   n |   mediana_dif |   p_holm |   cliffs_delta | efecto       | referencia_mejor   |
|:---------------|:---------------------------------|:---------------------------------|----:|--------------:|---------:|---------------:|:-------------|:-------------------|
| waiting_time_s | WM + LSTM                        | Tiempo fijo                      |  10 |       12.98   | 0.8262   |           0.2  | pequeño      | False              |
| waiting_time_s | WM + LSTM                        | Actuado                          |  10 |       57.02   | 0.01758  |           1    | grande       | False              |
| waiting_time_s | WM + LSTM                        | Max-Pressure                     |  10 |       20.97   | 0.01758  |           0.72 | grande       | False              |
| waiting_time_s | WM + LSTM                        | PPO directo                      |  10 |       33.29   | 0.05859  |           0.56 | grande       | False              |
| waiting_time_s | WM + LSTM                        | WM + TSMixer                     |  10 |        8.822  | 0.8262   |           0.22 | pequeño      | False              |
| waiting_time_s | WM + LSTM                        | WM + Transformer                 |  10 |        8.283  | 0.2578   |           0.2  | pequeño      | False              |
| waiting_time_s | WM + LSTM                        | WM + LSTM + planificación        |  10 |       19.43   | 0.01758  |           0.7  | grande       | False              |
| waiting_time_s | WM + LSTM                        | WM + TSMixer + planificación     |  10 |       14.37   | 0.1367   |           0.44 | mediano      | False              |
| waiting_time_s | WM + LSTM                        | WM + Transformer + planificación |  10 |        7.393  | 0.8262   |           0.14 | despreciable | False              |
| waiting_time_s | WM + TSMixer                     | Tiempo fijo                      |  10 |       -5.973  | 0.9668   |          -0.12 | despreciable | True               |
| waiting_time_s | WM + TSMixer                     | Actuado                          |  10 |       50.66   | 0.01758  |           0.92 | grande       | False              |
| waiting_time_s | WM + TSMixer                     | Max-Pressure                     |  10 |       16.39   | 0.5879   |           0.26 | pequeño      | False              |
| waiting_time_s | WM + TSMixer                     | PPO directo                      |  10 |       17.36   | 0.01758  |           0.4  | mediano      | False              |
| waiting_time_s | WM + TSMixer                     | WM + LSTM                        |  10 |       -8.822  | 1        |          -0.22 | pequeño      | True               |
| waiting_time_s | WM + TSMixer                     | WM + Transformer                 |  10 |       -1.186  | 1        |          -0.04 | despreciable | True               |
| waiting_time_s | WM + TSMixer                     | WM + LSTM + planificación        |  10 |       11.86   | 0.7852   |           0.22 | pequeño      | False              |
| waiting_time_s | WM + TSMixer                     | WM + TSMixer + planificación     |  10 |        3.028  | 1        |           0.1  | despreciable | False              |
| waiting_time_s | WM + TSMixer                     | WM + Transformer + planificación |  10 |       -6.3    | 1        |          -0.08 | despreciable | True               |
| waiting_time_s | WM + Transformer                 | Tiempo fijo                      |  10 |       -0.582  | 1        |          -0.02 | despreciable | True               |
| waiting_time_s | WM + Transformer                 | Actuado                          |  10 |       51.2    | 0.01758  |           1    | grande       | False              |
| waiting_time_s | WM + Transformer                 | Max-Pressure                     |  10 |       16.93   | 0.07812  |           0.32 | pequeño      | False              |
| waiting_time_s | WM + Transformer                 | PPO directo                      |  10 |       21.46   | 0.07812  |           0.42 | mediano      | False              |
| waiting_time_s | WM + Transformer                 | WM + LSTM                        |  10 |       -8.283  | 0.3223   |          -0.2  | pequeño      | True               |
| waiting_time_s | WM + Transformer                 | WM + TSMixer                     |  10 |        1.186  | 1        |           0.04 | despreciable | False              |
| waiting_time_s | WM + Transformer                 | WM + LSTM + planificación        |  10 |       11.57   | 0.1172   |           0.28 | pequeño      | False              |
| waiting_time_s | WM + Transformer                 | WM + TSMixer + planificación     |  10 |        1.389  | 1        |           0.16 | pequeño      | False              |
| waiting_time_s | WM + Transformer                 | WM + Transformer + planificación |  10 |       -1.627  | 1        |          -0.04 | despreciable | True               |
| waiting_time_s | WM + LSTM + planificación        | Tiempo fijo                      |  10 |       -4.58   | 0.3926   |          -0.18 | pequeño      | True               |
| waiting_time_s | WM + LSTM + planificación        | Actuado                          |  10 |       37.05   | 0.01758  |           1    | grande       | False              |
| waiting_time_s | WM + LSTM + planificación        | Max-Pressure                     |  10 |        2.366  | 0.08203  |           0.2  | pequeño      | False              |
| waiting_time_s | WM + LSTM + planificación        | PPO directo                      |  10 |       16.44   | 0.4922   |           0.3  | pequeño      | False              |
| waiting_time_s | WM + LSTM + planificación        | WM + LSTM                        |  10 |      -19.43   | 0.01758  |          -0.7  | grande       | True               |
| waiting_time_s | WM + LSTM + planificación        | WM + TSMixer                     |  10 |      -11.86   | 0.3926   |          -0.22 | pequeño      | True               |
| waiting_time_s | WM + LSTM + planificación        | WM + Transformer                 |  10 |      -11.57   | 0.09766  |          -0.28 | pequeño      | True               |
| waiting_time_s | WM + LSTM + planificación        | WM + TSMixer + planificación     |  10 |       -3.736  | 0.09766  |          -0.22 | pequeño      | True               |
| waiting_time_s | WM + LSTM + planificación        | WM + Transformer + planificación |  10 |      -13.03   | 0.06836  |          -0.38 | mediano      | True               |
| waiting_time_s | WM + TSMixer + planificación     | Tiempo fijo                      |  10 |       -2.55   | 0.8262   |          -0.04 | despreciable | True               |
| waiting_time_s | WM + TSMixer + planificación     | Actuado                          |  10 |       40.49   | 0.01758  |           1    | grande       | False              |
| waiting_time_s | WM + TSMixer + planificación     | Max-Pressure                     |  10 |        7.354  | 0.06836  |           0.22 | pequeño      | False              |
| waiting_time_s | WM + TSMixer + planificación     | PPO directo                      |  10 |       17.76   | 0.03125  |           0.42 | mediano      | False              |
| waiting_time_s | WM + TSMixer + planificación     | WM + LSTM                        |  10 |      -14.37   | 0.1367   |          -0.44 | mediano      | True               |
| waiting_time_s | WM + TSMixer + planificación     | WM + TSMixer                     |  10 |       -3.028  | 0.8262   |          -0.1  | despreciable | True               |
| waiting_time_s | WM + TSMixer + planificación     | WM + Transformer                 |  10 |       -1.389  | 0.8262   |          -0.16 | pequeño      | True               |
| waiting_time_s | WM + TSMixer + planificación     | WM + LSTM + planificación        |  10 |        3.736  | 0.1172   |           0.22 | pequeño      | False              |
| waiting_time_s | WM + TSMixer + planificación     | WM + Transformer + planificación |  10 |       -3.746  | 0.2578   |          -0.18 | pequeño      | True               |
| waiting_time_s | WM + Transformer + planificación | Tiempo fijo                      |  10 |        0.1054 | 1        |           0.06 | despreciable | False              |
| waiting_time_s | WM + Transformer + planificación | Actuado                          |  10 |       48.58   | 0.01758  |           1    | grande       | False              |
| waiting_time_s | WM + Transformer + planificación | Max-Pressure                     |  10 |       16.1    | 0.06836  |           0.38 | mediano      | False              |
| waiting_time_s | WM + Transformer + planificación | PPO directo                      |  10 |       18.08   | 0.03125  |           0.46 | mediano      | False              |
| waiting_time_s | WM + Transformer + planificación | WM + LSTM                        |  10 |       -7.393  | 1        |          -0.14 | despreciable | True               |
| waiting_time_s | WM + Transformer + planificación | WM + TSMixer                     |  10 |        6.3    | 1        |           0.08 | despreciable | False              |
| waiting_time_s | WM + Transformer + planificación | WM + Transformer                 |  10 |        1.627  | 1        |           0.04 | despreciable | False              |
| waiting_time_s | WM + Transformer + planificación | WM + LSTM + planificación        |  10 |       13.03   | 0.06836  |           0.38 | mediano      | False              |
| waiting_time_s | WM + Transformer + planificación | WM + TSMixer + planificación     |  10 |        3.746  | 0.3223   |           0.18 | pequeño      | False              |
| waiting_time_s | PPO directo                      | Tiempo fijo                      |  10 |      -20.22   | 0.001953 |          -0.42 | mediano      | True               |
| travel_time_s  | WM + LSTM                        | Tiempo fijo                      |  10 |       16.23   | 0.8633   |           0.2  | pequeño      | False              |
| travel_time_s  | WM + LSTM                        | Actuado                          |  10 |       64.08   | 0.01758  |           1    | grande       | False              |
| travel_time_s  | WM + LSTM                        | Max-Pressure                     |  10 |       23.4    | 0.01758  |           0.64 | grande       | False              |
| travel_time_s  | WM + LSTM                        | PPO directo                      |  10 |       36.38   | 0.05859  |           0.54 | grande       | False              |
| travel_time_s  | WM + LSTM                        | WM + TSMixer                     |  10 |        6.948  | 0.8633   |           0.1  | despreciable | False              |
| travel_time_s  | WM + LSTM                        | WM + Transformer                 |  10 |        8.204  | 0.5234   |           0.16 | pequeño      | False              |
| travel_time_s  | WM + LSTM                        | WM + LSTM + planificación        |  10 |       23.88   | 0.01758  |           0.62 | grande       | False              |
| travel_time_s  | WM + LSTM                        | WM + TSMixer + planificación     |  10 |       16.36   | 0.09766  |           0.38 | mediano      | False              |
| travel_time_s  | WM + LSTM                        | WM + Transformer + planificación |  10 |       11.24   | 0.5234   |           0.18 | pequeño      | False              |
| travel_time_s  | WM + TSMixer                     | Tiempo fijo                      |  10 |       -2.764  | 1        |          -0.06 | despreciable | True               |
| travel_time_s  | WM + TSMixer                     | Actuado                          |  10 |       61.23   | 0.01758  |           0.92 | grande       | False              |
| travel_time_s  | WM + TSMixer                     | Max-Pressure                     |  10 |       21.25   | 0.5879   |           0.24 | pequeño      | False              |
| travel_time_s  | WM + TSMixer                     | PPO directo                      |  10 |       22.34   | 0.01758  |           0.32 | pequeño      | False              |
| travel_time_s  | WM + TSMixer                     | WM + LSTM                        |  10 |       -6.948  | 1        |          -0.1  | despreciable | True               |
| travel_time_s  | WM + TSMixer                     | WM + Transformer                 |  10 |        0.4939 | 1        |           0.02 | despreciable | False              |
| travel_time_s  | WM + TSMixer                     | WM + LSTM + planificación        |  10 |       18.46   | 0.6328   |           0.24 | pequeño      | False              |
| travel_time_s  | WM + TSMixer                     | WM + TSMixer + planificación     |  10 |        8.983  | 0.9668   |           0.12 | despreciable | False              |
| travel_time_s  | WM + TSMixer                     | WM + Transformer + planificación |  10 |       -2.784  | 1        |          -0.02 | despreciable | True               |
| travel_time_s  | WM + Transformer                 | Tiempo fijo                      |  10 |        1.703  | 1        |           0.02 | despreciable | False              |
| travel_time_s  | WM + Transformer                 | Actuado                          |  10 |       59.97   | 0.01758  |           0.96 | grande       | False              |
| travel_time_s  | WM + Transformer                 | Max-Pressure                     |  10 |       19.95   | 0.07812  |           0.26 | pequeño      | False              |
| travel_time_s  | WM + Transformer                 | PPO directo                      |  10 |       23.91   | 0.07812  |           0.36 | mediano      | False              |
| travel_time_s  | WM + Transformer                 | WM + LSTM                        |  10 |       -8.204  | 0.5273   |          -0.16 | pequeño      | True               |
| travel_time_s  | WM + Transformer                 | WM + TSMixer                     |  10 |       -0.4939 | 1        |          -0.02 | despreciable | True               |
| travel_time_s  | WM + Transformer                 | WM + LSTM + planificación        |  10 |       16.04   | 0.08203  |           0.28 | pequeño      | False              |
| travel_time_s  | WM + Transformer                 | WM + TSMixer + planificación     |  10 |        3.994  | 0.5273   |           0.18 | pequeño      | False              |
| travel_time_s  | WM + Transformer                 | WM + Transformer + planificación |  10 |       -0.4424 | 1        |           0    | despreciable | True               |
| travel_time_s  | WM + LSTM + planificación        | Tiempo fijo                      |  10 |       -4.641  | 0.6973   |          -0.14 | despreciable | True               |
| travel_time_s  | WM + LSTM + planificación        | Actuado                          |  10 |       40.18   | 0.01758  |           0.98 | grande       | False              |
| travel_time_s  | WM + LSTM + planificación        | Max-Pressure                     |  10 |        0.1604 | 0.9844   |           0.08 | despreciable | False              |
| travel_time_s  | WM + LSTM + planificación        | PPO directo                      |  10 |       16.23   | 0.9844   |           0.28 | pequeño      | False              |
| travel_time_s  | WM + LSTM + planificación        | WM + LSTM                        |  10 |      -23.88   | 0.01758  |          -0.62 | grande       | True               |
| travel_time_s  | WM + LSTM + planificación        | WM + TSMixer                     |  10 |      -18.46   | 0.4219   |          -0.24 | pequeño      | True               |
| travel_time_s  | WM + LSTM + planificación        | WM + Transformer                 |  10 |      -16.04   | 0.06836  |          -0.28 | pequeño      | True               |
| travel_time_s  | WM + LSTM + planificación        | WM + TSMixer + planificación     |  10 |       -4.739  | 0.06836  |          -0.22 | pequeño      | True               |
| travel_time_s  | WM + LSTM + planificación        | WM + Transformer + planificación |  10 |      -14.74   | 0.06836  |          -0.36 | mediano      | True               |
| travel_time_s  | WM + TSMixer + planificación     | Tiempo fijo                      |  10 |       -1.909  | 0.3867   |          -0.04 | despreciable | True               |
| travel_time_s  | WM + TSMixer + planificación     | Actuado                          |  10 |       45.13   | 0.01758  |           0.98 | grande       | False              |
| travel_time_s  | WM + TSMixer + planificación     | Max-Pressure                     |  10 |        6.556  | 0.07812  |           0.22 | pequeño      | False              |
| travel_time_s  | WM + TSMixer + planificación     | PPO directo                      |  10 |       18.24   | 0.07812  |           0.36 | mediano      | False              |
| travel_time_s  | WM + TSMixer + planificación     | WM + LSTM                        |  10 |      -16.36   | 0.09766  |          -0.38 | mediano      | True               |
| travel_time_s  | WM + TSMixer + planificación     | WM + TSMixer                     |  10 |       -8.983  | 0.3867   |          -0.12 | despreciable | True               |
| travel_time_s  | WM + TSMixer + planificación     | WM + Transformer                 |  10 |       -3.994  | 0.3164   |          -0.18 | pequeño      | True               |
| travel_time_s  | WM + TSMixer + planificación     | WM + LSTM + planificación        |  10 |        4.739  | 0.07812  |           0.22 | pequeño      | False              |
| travel_time_s  | WM + TSMixer + planificación     | WM + Transformer + planificación |  10 |       -3.859  | 0.1953   |          -0.18 | pequeño      | True               |
| travel_time_s  | WM + Transformer + planificación | Tiempo fijo                      |  10 |        1.196  | 1        |           0.06 | despreciable | False              |
| travel_time_s  | WM + Transformer + planificación | Actuado                          |  10 |       53.42   | 0.01758  |           0.96 | grande       | False              |
| travel_time_s  | WM + Transformer + planificación | Max-Pressure                     |  10 |       14.99   | 0.07812  |           0.38 | mediano      | False              |
| travel_time_s  | WM + Transformer + planificación | PPO directo                      |  10 |       18.26   | 0.07812  |           0.42 | mediano      | False              |
| travel_time_s  | WM + Transformer + planificación | WM + LSTM                        |  10 |      -11.24   | 0.5234   |          -0.18 | pequeño      | True               |
| travel_time_s  | WM + Transformer + planificación | WM + TSMixer                     |  10 |        2.784  | 1        |           0.02 | despreciable | False              |
| travel_time_s  | WM + Transformer + planificación | WM + Transformer                 |  10 |        0.4424 | 1        |           0    | despreciable | False              |
| travel_time_s  | WM + Transformer + planificación | WM + LSTM + planificación        |  10 |       14.74   | 0.07812  |           0.36 | mediano      | False              |
| travel_time_s  | WM + Transformer + planificación | WM + TSMixer + planificación     |  10 |        3.859  | 0.2441   |           0.18 | pequeño      | False              |
| travel_time_s  | PPO directo                      | Tiempo fijo                      |  10 |      -19.66   | 0.001953 |          -0.38 | mediano      | True               |
