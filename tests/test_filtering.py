import torch

from datarater.train.filtering import filter_batch


def test_filter_batch_filter_mode() -> None:
    weights = torch.tensor([0.1, 0.9, 0.3, 0.8], dtype=torch.float32)
    losses = torch.arange(4, dtype=torch.float32)

    kept_losses, kept_indices = filter_batch(weights, losses, keep_rate=0.5, mode="filter")

    assert kept_indices.tolist() == [1, 3]
    assert torch.allclose(kept_losses, torch.tensor([1.0, 3.0]))


def test_filter_batch_reweight_mode() -> None:
    weights = torch.tensor([0.1, 0.9, 0.3, 0.7], dtype=torch.float32)
    losses = torch.ones(4, dtype=torch.float32)

    adjusted_losses, kept_indices = filter_batch(weights, losses, keep_rate=1.0, mode="reweight")

    assert kept_indices.tolist() == [0, 1, 2, 3]
    assert torch.allclose(adjusted_losses.sum(), torch.tensor(4.0))
    assert torch.allclose(adjusted_losses[1], losses[1] * weights[1] / weights.mean())
