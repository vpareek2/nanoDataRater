import torch

from datarater.models.rater import DataRater, DataRaterConfig


def test_rater_score_range() -> None:
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
    model = DataRater(config)
    tokens = torch.randint(0, config.vocab_size, (8, config.seq_len), dtype=torch.long)

    scores = model(tokens)

    assert scores.shape == (8,)
    assert torch.all(scores >= config.w_min - 1e-6)
    assert torch.all(scores <= config.w_max + 1e-6)
