"""Shared-memory looped attention decoder: 같은 attention·FFN block을 T번 반복하고, 매 반복에서 초기 상태의 K/V(shared memory)와
현재 상태의 K/V를 하나의 softmax로 함께 읽는다. 원래 입력을 반복마다 다시 더하지 않는다.

    X = input_proj(pooled) + hour_embedding;  H = X
    Z0 = ln_attn(X + loop_embedding[0]);  K_shared, V_shared = k_proj(Z0), v_proj(Z0)   (한 forward에서 한 번)
    r = 0..T−1:  Z = Z0 (r = 0) 또는 ln_attn(H + loop_embedding[r]);  Q = q_proj(Z)
                 K, V = K_shared, V_shared (r = 0, 24개) 또는 [K_shared; k_proj(Z)], [V_shared; v_proj(Z)] (48개)
                 U = H + dropout(out_proj(attention(Q, K, V)));  H = ln_rec(U + dropout(ffn(ln_ffn(U))))

T(loops)는 첫 pass를 포함한 전체 pass 수이다. 반복 embedding table은 loop_table행으로 고정해, T가 달라도 parameter 수와 초기화 순서가
같다. attention은 24시간 양방향이고 dropout은 train/eval mode를 따른다. 참조: reports/looped-decoder-2026-10-21/plan.md.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

KIND = "shared-memory-looped"
ATTRIBUTE = "decoder"
VERSION = 1
DEFAULTS = {"loops": 4, "width": 128, "heads": 4, "ffn": 256, "dropout": 0.1, "loop_table": 8}
HOURS = 24


def resolve(spec: dict) -> dict:
    if set(spec) - {"kind", *DEFAULTS}:
        raise ValueError(f"{KIND} decoder에 정의되지 않은 항목이 있습니다: {sorted(set(spec) - {'kind', *DEFAULTS})}")
    out = {"kind": KIND, **{k: spec.get(k, v) for k, v in DEFAULTS.items()}}
    for k in ("loops", "width", "heads", "ffn", "loop_table"):
        value = out[k]
        if isinstance(value, bool) or float(value) != int(value) or int(value) < 1:
            raise ValueError(f"{KIND} decoder의 {k}는 1 이상의 정수입니다: {value}")
        out[k] = int(value)
    out["dropout"] = float(out["dropout"])
    if not 0.0 <= out["dropout"] < 1.0:
        raise ValueError(f"{KIND} decoder의 dropout은 [0, 1)입니다: {out['dropout']}")
    if out["loops"] > out["loop_table"]:
        raise ValueError(f"{KIND} decoder의 loops({out['loops']})는 반복 embedding table 행 수({out['loop_table']}) 이하입니다.")
    if out["width"] % out["heads"]:
        raise ValueError(f"{KIND} decoder의 width는 heads로 나누어떨어져야 합니다.")
    return out


def build(spec: dict, n_in: int, settings: dict) -> "SharedMemoryLoopedDecoder":
    return SharedMemoryLoopedDecoder(n_in, **{k: spec[k] for k in DEFAULTS})


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dropout_p: float) -> torch.Tensor:
    """multi-head attention 한 번. q [B, heads, 24, d], k·v [B, heads, key 수, d]이고 key 전체를 하나의 softmax로 정규화한다."""
    return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)


class SharedMemoryLoopedDecoder(nn.Module):
    def __init__(self, n_in: int, loops: int, width: int, heads: int, ffn: int, dropout: float, loop_table: int):
        super().__init__()
        self.loops, self.heads, self.attention_dropout, self.out_width = loops, heads, dropout, width
        self.input_proj = nn.Linear(n_in, width)
        self.hour_embedding = nn.Parameter(torch.empty(HOURS, width))
        self.loop_embedding = nn.Parameter(torch.empty(loop_table, width))
        nn.init.normal_(self.hour_embedding, std=0.02)
        nn.init.normal_(self.loop_embedding, std=0.02)
        self.ln_attn = nn.LayerNorm(width)
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)
        self.ln_ffn = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, ffn), nn.GELU(), nn.Linear(ffn, width))
        self.ln_rec = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 시각, width] → [B, heads, 시각, width/heads]."""
        b, t, w = x.shape
        return x.reshape(b, t, self.heads, w // self.heads).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        b, h, t, d = x.shape
        return x.transpose(1, 2).reshape(b, t, h * d)

    def forward(self, pooled: torch.Tensor, detach_shared: bool = False) -> torch.Tensor:
        """pooled [B, 24, n_in] → 최종 상태 H [B, 24, width]. detach_shared는 shared memory 기울기 경로를 끊어 보는 진단용이다."""
        x = self.input_proj(pooled) + self.hour_embedding
        z0 = self.ln_attn(x + self.loop_embedding[0])
        k_shared, v_shared = self._split(self.k_proj(z0)), self._split(self.v_proj(z0))
        if detach_shared:
            k_shared, v_shared = k_shared.detach(), v_shared.detach()
        p = self.attention_dropout if self.training else 0.0
        h = x
        for r in range(self.loops):
            z = z0 if r == 0 else self.ln_attn(h + self.loop_embedding[r])
            q = self._split(self.q_proj(z))
            if r == 0:              # 첫 pass는 shared와 current가 같은 표현이라 한 번만 읽는다
                k, v = k_shared, v_shared
            else:
                k = torch.cat([k_shared, self._split(self.k_proj(z))], dim=2)
                v = torch.cat([v_shared, self._split(self.v_proj(z))], dim=2)
            u = h + self.dropout(self.out_proj(self._merge(attend(q, k, v, p))))
            h = self.ln_rec(u + self.dropout(self.ffn(self.ln_ffn(u))))
        return h
