"""Registro de los modelos temporales del World Model (LSTM, TSMixer, Transformer).

Los tres comparten la interfaz forward(x [B, W, input_dim]) -> (Δŝ [B, state_dim], r̂ [B, n_tls]) y un
`build(target_params, **arch)` que ajusta su dimensión libre (ancho oculto o del MLP) al presupuesto de
parámetros. El resto del código (entrenamiento, evaluación, DreamEnv, PPO) solo conoce el nombre del modelo.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from torch import nn

from wm.models.lstm import LSTMConfig, LSTMModel, build_lstm
from wm.models.transformer import TransformerConfig, TransformerModel, build_transformer
from wm.models.tsmixer import TSMixer, TSMixerConfig, build_tsmixer


@dataclass(frozen=True)
class TemporalModel:
    label: str                         # nombre en tablas y figuras
    config: type
    model: type[nn.Module]
    build: Callable[..., nn.Module]    # build(target_params, input_dim=, state_dim=, n_tls=, window=, **arch)


TEMPORAL_MODELS = {
    "lstm": TemporalModel("LSTM", LSTMConfig, LSTMModel, build_lstm),
    "tsmixer": TemporalModel("TSMixer", TSMixerConfig, TSMixer, build_tsmixer),
    "transformer": TemporalModel("Transformer", TransformerConfig, TransformerModel, build_transformer),
}


def build_temporal(kind: str, target_params: int | None = None, **kwargs) -> nn.Module:
    return TEMPORAL_MODELS[kind].build(target_params, **kwargs)


def from_config(kind: str, config: dict) -> nn.Module:
    spec = TEMPORAL_MODELS[kind]
    return spec.model(spec.config(**config))
