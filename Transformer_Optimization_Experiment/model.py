"""Self-contained model definitions for the Transformer Optimization Experiment.

All models accept ``bird_ids`` for API compatibility with existing dataloaders
but ignore it — bird embeddings are disabled by design (year-grouped split
makes them counterproductive).
"""

from __future__ import annotations

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

def _head(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    """Three-layer head: LN → 128 → GELU → Drop → 64 → GELU → Drop → out."""
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


# ---------------------------------------------------------------------------
# Baseline: 2-layer LSTM (matches the original experiment)
# ---------------------------------------------------------------------------

class TrilineLSTM(nn.Module):
    """Original 2-layer LSTM baseline (no bird embedding)."""

    def __init__(
        self,
        n_features: int,
        n_birds: int = 0,           # ignored — kept for API compatibility
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_projection = nn.Linear(n_features, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fly_head = _head(hidden_dim, 1, dropout)
        self.distance_head = _head(hidden_dim, 1, dropout)
        self.direction_head = _head(hidden_dim, 2, dropout)

    def forward(self, features: torch.Tensor, bird_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        # bird_ids is ignored
        x = self.feature_projection(features)
        encoded, _ = self.lstm(x)
        pooled = encoded[:, -1, :]
        return {
            "fly_logit": self.fly_head(pooled).squeeze(-1),
            "log_distance": self.distance_head(pooled).squeeze(-1),
            "direction": self.direction_head(pooled),
        }


# ---------------------------------------------------------------------------
# 4-layer LSTM — fair depth comparison with the 4-layer Transformer
# ---------------------------------------------------------------------------

class TrilineLSTM4L(nn.Module):
    """Wider 4-layer LSTM for fair comparison with deep transformer."""

    def __init__(
        self,
        n_features: int,
        n_birds: int = 0,           # ignored
        hidden_dim: int = 256,
        n_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_projection = nn.Linear(n_features, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
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


# ---------------------------------------------------------------------------
# Transformer V2 — CLS token, deeper, wider, 4 layers
# ---------------------------------------------------------------------------

class TrilineTransformerV2(nn.Module):
    """Improved transformer with CLS token, 4 layers, d_model=256, 8 heads.

    Key changes over the original TrilineTransformer:
    - CLS token participates in all encoder layers (not post-hoc pooling)
    - d_model 128 → 256, heads 4 → 8, layers 2 → 4
    - Deeper output heads (128 → 64 instead of single 64)
    """

    def __init__(
        self,
        n_features: int,
        n_birds: int = 0,            # ignored
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
        # +1 position for CLS token
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_k + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Deeper output heads
        self.fly_head = _head(d_model, 1, dropout)
        self.distance_head = _head(d_model, 1, dropout)
        self.direction_head = _head(d_model, 2, dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.02)

    def forward(self, features: torch.Tensor, bird_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        B, S, _ = features.shape
        # Project and prepend CLS token
        x = self.feature_projection(features)                     # (B, S, d_model)
        cls_tokens = self.cls_token.expand(B, -1, -1)             # (B, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)                     # (B, 1+S, d_model)
        x = x + self.positional_embedding[:, : S + 1, :]
        encoded = self.encoder(x)
        cls_out = encoded[:, 0, :]                                 # CLS token output
        return {
            "fly_logit": self.fly_head(cls_out).squeeze(-1),
            "log_distance": self.distance_head(cls_out).squeeze(-1),
            "direction": self.direction_head(cls_out),
        }
