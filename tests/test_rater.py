import pytest
import torch

from datarater.models.rater import DataRater, DataRaterConfig


@pytest.mark.parametrize("device_str", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"))])
def test_rater_score_range(device_str: str) -> None:
    device = torch.device(device_str)
    config = DataRaterConfig(
        vocab_size=32,
        seq_len=16,
        d_model=32,
        n_heads=4,
        d_ff=64,
        num_layers=2,
        tau=0.9,
        w_min=0.05,
        w_max=0.9,
        dropout=0.0,
    )
    model = DataRater(config).to(device)
    if device_str == "cuda":
        model = model.to(torch.bfloat16)
    tokens = torch.randint(0, config.vocab_size, (8, config.seq_len), dtype=torch.long, device=device)

    scores = model(tokens)

    assert scores.shape == (8,)
    assert torch.all(scores >= config.w_min - 1e-6)
    assert torch.all(scores <= config.w_max + 1e-6)
