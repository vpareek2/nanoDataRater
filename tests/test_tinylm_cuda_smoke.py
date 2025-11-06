from __future__ import annotations

import math

import pytest
import torch

from datarater.data.datamodule import SequenceBatch
from datarater.models.tiny_lm import TinyLM, TinyLMConfig
from datarater.utils.seed import seed_everything


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_tinylm_training_step() -> None:
    """Smoke test: TinyLM forward, loss computation, backward, and optimizer step on CUDA."""
    seed_everything(42)
    device = torch.device("cuda")

    config = TinyLMConfig(
        vocab_size=16,
        d_model=32,
        n_heads=4,
        num_layers=2,
        d_ff=64,
        max_seq_len=8,
        dropout=0.0,
    )
    model = TinyLM(config).to(device).to(torch.bfloat16)

    # Create a small batch on CUDA
    batch_size = 4
    seq_len = 6
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)
    target_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)

    # Forward pass
    logits = model(input_ids)
    assert logits.shape == (batch_size, seq_len, config.vocab_size)

    # Compute loss
    loss = model.compute_sequence_loss(input_ids, target_ids, logits=logits)
    assert loss.shape == (batch_size,)
    assert torch.all(torch.isfinite(loss))

    # Backward pass
    total_loss = loss.mean()
    total_loss.backward()

    # Check gradients exist
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient missing for {name}"
        assert torch.all(torch.isfinite(param.grad)), f"Non-finite gradient for {name}"

    # Optimizer step
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.step()
    optimizer.zero_grad()

    # Verify loss is still finite after step
    assert math.isfinite(total_loss.item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_tinylm_with_sequence_batch() -> None:
    """Smoke test: TinyLM works with SequenceBatch dataclass."""
    device = torch.device("cuda")

    config = TinyLMConfig(
        vocab_size=8,
        d_model=16,
        n_heads=2,
        num_layers=1,
        d_ff=32,
        max_seq_len=4,
        dropout=0.0,
    )
    model = TinyLM(config).to(device).to(torch.bfloat16)

    batch = SequenceBatch(
        input_ids=torch.randint(0, config.vocab_size, (2, 4), dtype=torch.long, device=device),
        target_ids=torch.randint(0, config.vocab_size, (2, 4), dtype=torch.long, device=device),
    )

    logits = model(batch.input_ids)
    loss = model.compute_sequence_loss(batch.input_ids, batch.target_ids, logits=logits)

    assert logits.shape == (2, 4, config.vocab_size)
    assert loss.shape == (2,)
    assert torch.all(torch.isfinite(loss))

