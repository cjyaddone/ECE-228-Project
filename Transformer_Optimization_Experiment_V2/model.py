"""Model variants for Transformer Optimization Experiment V2.

Key improvements over V1:
- TrilineTransformerV3: CLS + last-position hybrid pooling (recency bias)
- TrilineTransformerV3_ALiBi: ALiBi attention + CLS+last (no learned positional emb)
- TrilineLSTM6L: Stronger 6-layer LSTM baseline at 512 dim
"""

from __future__ import annotations

import math

import torch
from torch import nn


# ═══════════════════════════════════════════════════════════════════════════
# Shared building blocks
# ═══════════════════════════════════════════════════════════════════════════

def _head(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, 128),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(128, 64),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(64, out_dim),
    )


# ═══════════════════════════════════════════════════════════════════════════
# LSTM baselines
# ═══════════════════════════════════════════════════════════════════════════

class TrilineLSTM(nn.Module):
    """Original 2-layer LSTM baseline (128 dim)."""

    def __init__(
        self,
        n_features: int,
        n_birds: int = 0,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_projection = nn.Linear(n_features, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fly_head = _head(hidden_dim, 1, dropout)
        self.distance_head = _head(hidden_dim, 1, dropout)
        self.direction_head = _head(hidden_dim, 2, dropout)

    def forward(self, features: torch.Tensor, bird_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        x = self.feature_projection(features)
        encoded, _ = self.lstm(x)
        pooled = encoded[:, -1, :]
        return {
            "fly_logit": self.fly_head(pooled).squeeze(-1),
            "log_distance": self.distance_head(pooled).squeeze(-1),
            "direction": self.direction_head(pooled),
        }


class TrilineLSTM4L(nn.Module):
    """4-layer LSTM (256 dim)."""

    def __init__(
        self,
        n_features: int,
        n_birds: int = 0,
        hidden_dim: int = 256,
        n_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_projection = nn.Linear(n_features, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_layers=n_layers, batch_first=True, dropout=dropout,
        )
        self.fly_head = _head(hidden_dim, 1, dropout)
        self.distance_head = _head(hidden_dim, 1, dropout)
        self.direction_head = _head(hidden_dim, 2, dropout)

    def forward(self, features: torch.Tensor, bird_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        x = self.feature_projection(features)
        encoded, _ = self.lstm(x)
        pooled = encoded[:, -1, :]
        return {
            "fly_logit": self.fly_head(pooled).squeeze(-1),
            "log_distance": self.distance_head(pooled).squeeze(-1),
            "direction": self.direction_head(pooled),
        }


class TrilineLSTM6L(nn.Module):
    """6-layer LSTM (512 dim) — strongest LSTM baseline."""

    def __init__(
        self,
        n_features: int,
        n_birds: int = 0,
        hidden_dim: int = 512,
        n_layers: int = 6,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.feature_projection = nn.Linear(n_features, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_layers=n_layers, batch_first=True, dropout=dropout,
        )
        self.fly_head = _head(hidden_dim, 1, dropout)
        self.distance_head = _head(hidden_dim, 1, dropout)
        self.direction_head = _head(hidden_dim, 2, dropout)

    def forward(self, features: torch.Tensor, bird_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        x = self.feature_projection(features)
        encoded, _ = self.lstm(x)
        pooled = encoded[:, -1, :]
        return {
            "fly_logit": self.fly_head(pooled).squeeze(-1),
            "log_distance": self.distance_head(pooled).squeeze(-1),
            "direction": self.direction_head(pooled),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Transformer V2  (CLS token baseline, from experiment V1)
# ═══════════════════════════════════════════════════════════════════════════

class TrilineTransformerV2(nn.Module):
    """CLS-token transformer — d_model=256, 4 layers, 8 heads."""

    def __init__(
        self,
        n_features: int,
        n_birds: int = 0,
        max_k: int = 30,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.feature_projection = nn.Linear(n_features, d_model)
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_k + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.fly_head = _head(d_model, 1, dropout)
        self.distance_head = _head(d_model, 1, dropout)
        self.direction_head = _head(d_model, 2, dropout)

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.02)

    def forward(self, features: torch.Tensor, bird_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        B, S, _ = features.shape
        x = self.feature_projection(features)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.positional_embedding[:, : S + 1, :]
        encoded = self.encoder(x)
        cls_out = encoded[:, 0, :]
        return {
            "fly_logit": self.fly_head(cls_out).squeeze(-1),
            "log_distance": self.distance_head(cls_out).squeeze(-1),
            "direction": self.direction_head(cls_out),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Transformer V3  — CLS + last-position hybrid pooling
# ═══════════════════════════════════════════════════════════════════════════

class TrilineTransformerV3(nn.Module):
    """CLS + last-position concat for recency-aware prediction.

    The LSTM wins on short sequences because ``pooled = encoded[:, -1, :]``
    naturally focuses on the most recent day.  This variant gives the
    transformer the same advantage by concatenating the CLS output (global
    context) with the encoding of the most recent input position (recency).
    """

    def __init__(
        self,
        n_features: int,
        n_birds: int = 0,
        max_k: int = 30,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.feature_projection = nn.Linear(n_features, d_model)
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_k + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Hybrid representation: CLS (global) + last position (recency)
        # Input dim = 2 * d_model
        self.fly_head = _head(d_model * 2, 1, dropout)
        self.distance_head = _head(d_model * 2, 1, dropout)
        self.direction_head = _head(d_model * 2, 2, dropout)

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.02)

    def forward(self, features: torch.Tensor, bird_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        B, S, _ = features.shape
        x = self.feature_projection(features)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.positional_embedding[:, : S + 1, :]
        encoded = self.encoder(x)

        # Hybrid: CLS (position 0) + last real position (position S)
        cls_out = encoded[:, 0, :]       # global context
        last_out = encoded[:, S, :]      # most recent day  (CLS is pos 0, so day k is pos S)
        hybrid = torch.cat([cls_out, last_out], dim=-1)

        return {
            "fly_logit": self.fly_head(hybrid).squeeze(-1),
            "log_distance": self.distance_head(hybrid).squeeze(-1),
            "direction": self.direction_head(hybrid),
        }


# ═══════════════════════════════════════════════════════════════════════════
# ALiBi building blocks  (custom, avoids fighting with PyTorch's encoder API)
# ═══════════════════════════════════════════════════════════════════════════

class _ALiBiAttention(nn.Module):
    """Multi-head self-attention with ALiBi (Attention with Linear Biases).

    Adds a fixed, non-learned bias to attention scores that decays with
    distance — giving the model a natural recency preference without
    learning positional embeddings.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # ALiBi slopes (one per head)
        slopes = torch.tensor(
            [2.0 ** (-8.0 * i / n_heads) for i in range(n_heads)]
        )
        # Precompute symmetric distance bias for max_seq_len
        pos = torch.arange(max_seq_len, dtype=torch.float32)
        dist = (pos[:, None] - pos[None, :]).abs()          # (S, S)
        alibi = -slopes[:, None, None] * dist[None, :, :]   # (H, S, S)
        self.register_buffer("alibi", alibi, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        H, d = self.n_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, S, 3, H, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                     # (B, H, S, d)

        attn = (q @ k.transpose(-2, -1)) * self.scale         # (B, H, S, S)
        attn = attn + self.alibi[:, :S, :S].unsqueeze(0)      # add ALiBi
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = attn @ v                                         # (B, H, S, d)
        out = out.transpose(1, 2).reshape(B, S, D)
        return self.out_proj(out)


class _PreNormBlock(nn.Module):
    """Pre-norm transformer block with ALiBi attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _ALiBiAttention(d_model, n_heads, max_seq_len, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


# ═══════════════════════════════════════════════════════════════════════════
# Transformer V3 + ALiBi
# ═══════════════════════════════════════════════════════════════════════════

class TrilineTransformerV3_ALiBi(nn.Module):
    """CLS + last-position hybrid pooling with ALiBi instead of learned pos emb.

    ALiBi provides a natural recency bias in every attention layer without
    learning positional embeddings.  Combined with CLS+last hybrid pooling,
    this gives the transformer strong inductive bias toward recent days.
    """

    def __init__(
        self,
        n_features: int,
        n_birds: int = 0,
        max_k: int = 30,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        max_seq_len = max_k + 1  # CLS + up to max_k positions
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.feature_projection = nn.Linear(n_features, d_model)

        # No learned positional embedding — ALiBi handles position
        self.blocks = nn.ModuleList([
            _PreNormBlock(d_model, n_heads, max_seq_len, ff_mult=4, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Hybrid pooling: CLS + last position → 2 * d_model
        self.fly_head = _head(d_model * 2, 1, dropout)
        self.distance_head = _head(d_model * 2, 1, dropout)
        self.direction_head = _head(d_model * 2, 2, dropout)

        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, features: torch.Tensor, bird_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        B, S, _ = features.shape
        x = self.feature_projection(features)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)    # (B, 1+S, d_model)
        # No positional embedding — ALiBi in each block provides position signal

        for block in self.blocks:
            x = block(x)

        cls_out = x[:, 0, :]                      # global context
        last_out = x[:, S, :]                     # most recent real day
        hybrid = torch.cat([cls_out, last_out], dim=-1)

        return {
            "fly_logit": self.fly_head(hybrid).squeeze(-1),
            "log_distance": self.distance_head(hybrid).squeeze(-1),
            "direction": self.direction_head(hybrid),
        }
