from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TinyLMConfig:
    vocab_size: int
    d_model: int
    n_heads: int
    num_layers: int
    d_ff: int
    max_seq_len: int
    dropout: float = 0.0


class TinyLM(nn.Module):
    """Tiny causal language model used in the inner training loop."""

    def __init__(self, config: TinyLMConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Parameter(torch.zeros(config.max_seq_len, config.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be of shape (batch, seq_len)")
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError("sequence length exceeds configured maximum")

        seq_len = input_ids.size(1)
        positions = self.position_embedding[:seq_len]
        hidden = self.embedding(input_ids) + positions
        mask = self._causal_mask(seq_len, hidden.device)
        encoded = self.blocks(hidden, mask=mask)
        return self.lm_head(self.norm(encoded))

    def compute_sequence_loss(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits_to_use = logits if logits is not None else self.forward(input_ids)
        if target_ids.shape != input_ids.shape:
            raise ValueError("target_ids must match input_ids shape")
        per_token = nn.functional.cross_entropy(
            logits_to_use.view(-1, logits_to_use.size(-1)), target_ids.view(-1), reduction="none"
        ).view(input_ids.size(0), -1)
        return per_token.mean(dim=1)
