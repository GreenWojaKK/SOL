"""Residual inverted attention: 관측소 하나의 24시간 provider 표현 전체를 token으로 두고, 같은 대상일의 모든 관측소(WA·WB) token 사이
attention으로 만든 update를 원래 표현에 더한다.

    t_s = token_in(flatten_hours(e_p[s])) + provider_embedding[p] + station_embedding_p[s]      [U, S_wa + S_wb, width]
    V = t + dropout(attention(LN1(t)));  H = V + dropout(FFN(LN2(V)))
    e'_p[s] = e_p[s] + reshape(token_out(LN_out(H_s)), [24, R])

비활성 제공사의 token은 key/value에서 빼고 그 제공사의 residual은 0이다. 참조: reports/provider-context-2026-10-21/plan.md.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

KIND = "inverted-attention"
VERSION = 1
DEFAULTS = {"width": 64, "heads": 4, "ffn": 128, "dropout": 0.1, "shared_projection": True}
HOURS = 24


def resolve(spec: dict) -> dict:
    if set(spec) - {"kind", *DEFAULTS}:
        raise ValueError(f"{KIND} provider context에 정의되지 않은 항목이 있습니다: {sorted(set(spec) - {'kind', *DEFAULTS})}")
    out = {"kind": KIND, **{k: spec.get(k, v) for k, v in DEFAULTS.items()}}
    for k in ("width", "heads", "ffn"):
        value = out[k]
        if isinstance(value, bool) or float(value) != int(value) or int(value) < 1:
            raise ValueError(f"{KIND} provider context의 {k}는 1 이상의 정수입니다: {value}")
        out[k] = int(value)
    out["dropout"] = float(out["dropout"])
    if not 0.0 <= out["dropout"] < 1.0:
        raise ValueError(f"{KIND} provider context의 dropout은 [0, 1)입니다: {out['dropout']}")
    if out["width"] % out["heads"]:
        raise ValueError(f"{KIND} provider context의 width는 heads로 나누어떨어져야 합니다.")
    if not isinstance(out["shared_projection"], bool):
        raise ValueError(f"{KIND} provider context의 shared_projection은 true 또는 false입니다.")
    return out


def build(spec: dict, providers: tuple, n_stations: dict, rep_width: int, active: dict) -> "InvertedStationAttention":
    return InvertedStationAttention(providers, n_stations, rep_width, active, **{k: spec[k] for k in DEFAULTS})


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """관측소 축 multi-head attention 한 번. q·k·v [U, heads, 관측소, d], mask는 key에 더할 값 [관측소](비활성 제공사는 매우 큰 음수)이다."""
    return F.scaled_dot_product_attention(q, k, v, attn_mask=None if mask is None else mask.view(1, 1, 1, -1))


class InvertedStationAttention(nn.Module):
    def __init__(self, providers: tuple, n_stations: dict, rep_width: int, active: dict, width: int, heads: int, ffn: int, dropout: float,
                 shared_projection: bool):
        super().__init__()
        self.providers, self.heads, self.shared = tuple(providers), heads, shared_projection
        self.sizes = [int(n_stations[p]) for p in self.providers]
        flat = HOURS * rep_width
        if shared_projection:
            self.token_in, self.token_out = nn.Linear(flat, width), nn.Linear(width, flat)
        else:
            self.token_in = nn.ModuleDict({p: nn.Linear(flat, width) for p in self.providers})
            self.token_out = nn.ModuleDict({p: nn.Linear(width, flat) for p in self.providers})
        self.provider_embedding = nn.Parameter(torch.empty(len(self.providers), width))
        self.station_embedding = nn.ParameterDict({p: nn.Parameter(torch.empty(n_stations[p], width)) for p in self.providers})
        nn.init.normal_(self.provider_embedding, std=0.02)
        for p in self.providers:
            nn.init.normal_(self.station_embedding[p], std=0.02)
        self.ln1 = nn.LayerNorm(width)
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)
        self.ln2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, ffn), nn.GELU(), nn.Linear(ffn, width))
        self.ln_out = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        keep = torch.cat([torch.full((s,), bool(active[p])) for p, s in zip(self.providers, self.sizes)])
        self.masked = not bool(keep.all())
        self.register_buffer("key_bias", torch.where(keep, 0.0, -1e9), persistent=False)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        """[U, 관측소, width] → [U, heads, 관측소, width/heads]."""
        u, s, w = x.shape
        return x.reshape(u, s, self.heads, w // self.heads).transpose(1, 2)

    def forward(self, reps: dict, active: dict) -> dict:
        """reps {제공사: [U, S_p, 24, R]}, active {제공사: 0/1 tensor} → context를 더한 같은 모양의 표현."""
        tokens = []
        for i, p in enumerate(self.providers):
            e = reps[p]
            flat = e.reshape(e.shape[0], e.shape[1], -1)
            projected = self.token_in(flat) if self.shared else self.token_in[p](flat)
            tokens.append(projected + self.provider_embedding[i] + self.station_embedding[p])
        t = torch.cat(tokens, dim=1)
        z = self.ln1(t)
        a = attend(self._split(self.q_proj(z)), self._split(self.k_proj(z)), self._split(self.v_proj(z)), self.key_bias if self.masked else None)
        v = t + self.dropout(self.out_proj(a.transpose(1, 2).reshape(t.shape)))
        h = self.ln_out(v + self.dropout(self.ffn(self.ln2(v))))
        out, start = {}, 0
        for p, s in zip(self.providers, self.sizes):
            part = h[:, start:start + s]
            start += s
            delta = self.token_out(part) if self.shared else self.token_out[p](part)
            out[p] = reps[p] + delta.reshape(reps[p].shape) * active[p]
        return out
