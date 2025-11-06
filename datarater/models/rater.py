from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from datarater.utils.attention import sdpa_ctx


@dataclass(frozen=True)
class DataRaterConfig:
    """Configuration for the :class:`DataRater` transformer encoder."""

    vocab_size: int
    seq_len: int
    d_model: int
    n_heads: int
    d_ff: int
    num_layers: int
    tau: float
    w_min: float
    w_max: float
    dropout: float = 0.0


class DataRater(nn.Module):
    """Transformer encoder that produces per-sequence weights in (w_min, w_max)."""

    def __init__(self, config: DataRaterConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Parameter(torch.zeros(config.seq_len, config.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 1),
        )
        self.register_buffer("clamp_min", torch.tensor(config.w_min, dtype=torch.float32))
        self.register_buffer("clamp_max", torch.tensor(config.w_max, dtype=torch.float32))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be of shape (batch, seq_len)")
        if input_ids.size(1) > self.config.seq_len:
            raise ValueError("sequence length exceeds configured maximum")

        embeddings = self.embedding(input_ids)
        position = self.position_embedding[: embeddings.size(1)]
        hidden = embeddings + position
        # Use first-order backend (Flash/Efficient) for standard forward pass
        # Note: Flash Attention requires bf16/fp16 - ensure model/dtype is set appropriately
        with sdpa_ctx("first_order"):
            encoded = self.encoder(hidden)
        pooled = self.norm(encoded.mean(dim=1))
        logits = self.head(pooled).squeeze(-1)
        scores = torch.sigmoid(logits / self.config.tau)
        return torch.clamp(scores, min=self.clamp_min.item(), max=self.clamp_max.item())
