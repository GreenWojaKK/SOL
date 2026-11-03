"""BiGRU decoder: pooled 시퀀스의 다층 양방향 GRU. 크기는 runtime hyperparameters의 dl_gru_hidden·dl_gru_layers·dl_dropout이다.

GenerationNet 속성 이름은 gru이다(MK3·MK5 checkpoint의 parameter 이름 gru.layers.*가 그대로 맞는다).
"""
from __future__ import annotations

import torch
from torch import nn

KIND = "bigru"
ATTRIBUTE = "gru"
VERSION = 1
DEFAULTS: dict = {}


def resolve(spec: dict) -> dict:
    if set(spec) - {"kind"}:
        raise ValueError(f"BiGRU decoder의 크기는 runtime hyperparameters(dl_gru_*)로 정합니다: {sorted(spec)}")
    return {"kind": KIND}


def build(spec: dict | None, n_in: int, settings: dict) -> "StackedBiGRU":
    return StackedBiGRU(n_in, settings["dl_gru_hidden"], settings["dl_gru_layers"], settings["dl_dropout"])


class StackedBiGRU(nn.Module):
    """다층 양방향 GRU. 층 사이 dropout은 nn.Dropout으로 둔다(cuDNN RNN 내부 dropout은 이 실행 환경에서 종료 시 비정상 종료를 일으킨다)."""

    def __init__(self, n_in: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([nn.GRU(n_in if i == 0 else 2 * hidden, hidden, batch_first=True, bidirectional=True) for i in range(layers)])
        self.dropout = nn.Dropout(dropout)
        self.out_width = 2 * hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            if i:
                x = self.dropout(x)
            x, _ = layer(x)
        return x
