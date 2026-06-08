from __future__ import annotations

import torch
from torch import nn


class TrilineTransformer(nn.Module):
    """Transformer V2 — CLS token, 4 layers, d_model=256, 8 heads.

    Key improvements over the original:
    - CLS token participates in all encoder layers (not post-hoc pooling)
    - d_model 128 → 256, heads 4 → 8, layers 2 → 4
    - Deeper output heads (128 → 64 → out instead of 64 → out)
    """

    def __init__(
        self,
        n_features: int,
        n_birds: int,
        max_k: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        bird_embedding_dim: int = 2,
        use_bird_id: bool = False,
    ) -> None:
        super().__init__()
        self.use_bird_id = use_bird_id
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.feature_projection = nn.Linear(n_features, d_model)
        # +1 position for CLS token
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_k + 1, d_model))

        if self.use_bird_id:
            self.bird_embedding = nn.Embedding(n_birds, bird_embedding_dim)
            self.bird_projection = nn.Linear(bird_embedding_dim, d_model)

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

        self.fly_head = self._head(d_model, 1, dropout)
        self.distance_head = self._head(d_model, 1, dropout)
        self.direction_head = self._head(d_model, 2, dropout)

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.02)

    @staticmethod
    def _head(d_model: int, out_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim),
        )

    def forward(self, features: torch.Tensor, bird_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        B, S, _ = features.shape
        x = self.feature_projection(features)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)                     # (B, 1+S, d_model)
        x = x + self.positional_embedding[:, : S + 1, :]
        if self.use_bird_id:
            x = x + self.bird_projection(self.bird_embedding(bird_ids)).unsqueeze(1)
        encoded = self.encoder(x)
        cls_out = encoded[:, 0, :]                                 # CLS token output
        return {
            "fly_logit": self.fly_head(cls_out).squeeze(-1),
            "log_distance": self.distance_head(cls_out).squeeze(-1),
            "direction": self.direction_head(cls_out),
        }


TrilineFlyTransformer = TrilineTransformer
