from __future__ import annotations

import torch
from torch import nn


class TrilineTransformer(nn.Module):
    """Transformer encoder with fly, distance, and heading heads.

    Uses learned attention pooling over the encoded sequence so the model
    can weight recent days more heavily than distant ones, rather than
    averaging all positions equally.
    """

    def __init__(
        self,
        n_features: int,
        n_birds: int,
        max_k: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        bird_embedding_dim: int = 2,
        use_bird_id: bool = True,
    ) -> None:
        super().__init__()
        self.use_bird_id = use_bird_id
        self.feature_projection = nn.Linear(n_features, d_model)
        if self.use_bird_id:
            self.bird_embedding = nn.Embedding(n_birds, bird_embedding_dim)
            self.bird_projection = nn.Linear(bird_embedding_dim, d_model)
        else:
            self.bird_embedding = None
            self.bird_projection = None
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_k, d_model))

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

        # Learned attention pooling: each position gets a scalar score,
        # softmax-normalized across the sequence, then weighted sum.
        self.pool_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )

        self.fly_head = self._head(d_model, 1, dropout)
        self.distance_head = self._head(d_model, 1, dropout)
        self.direction_head = self._head(d_model, 2, dropout)

        nn.init.normal_(self.positional_embedding, mean=0.0, std=0.02)

    @staticmethod
    def _head(d_model: int, out_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim),
        )

    def forward(self, features: torch.Tensor, bird_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        seq_len = features.shape[1]
        x = self.feature_projection(features)
        x = x + self.positional_embedding[:, :seq_len, :]
        if self.use_bird_id:
            x = x + self.bird_projection(self.bird_embedding(bird_ids)).unsqueeze(1)
        encoded = self.encoder(x)
        # Learned attention pooling over sequence positions
        attn_scores = self.pool_attention(encoded).squeeze(-1)  # (B, seq_len)
        attn_weights = torch.softmax(attn_scores, dim=-1)       # (B, seq_len)
        pooled = (encoded * attn_weights.unsqueeze(-1)).sum(dim=1)
        return {
            "fly_logit": self.fly_head(pooled).squeeze(-1),
            "log_distance": self.distance_head(pooled).squeeze(-1),
            "direction": self.direction_head(pooled),
        }


TrilineFlyTransformer = TrilineTransformer
