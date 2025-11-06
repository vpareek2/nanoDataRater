from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pytest
import torch

from datarater.data.datamodule import InMemorySequenceDataModule, SequenceBatch
from datarater.models.rater import DataRater, DataRaterConfig
from datarater.models.tiny_lm import TinyLM, TinyLMConfig
from datarater.train.meta_trainer import InnerLoopConfig, MetaTrainer, MetaTrainerConfig
from datarater.utils.seed import seed_everything


@dataclass
class SyntheticSequence:
    tokens: List[int]

    def to_batch(self) -> SequenceBatch:
        input_ids = torch.tensor(self.tokens, dtype=torch.long)
        target_ids = torch.tensor(self.tokens[1:] + [self.tokens[-1]], dtype=torch.long)
        return SequenceBatch(input_ids=input_ids, target_ids=target_ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_meta_trainer_minimal_smoke() -> None:
    """Minimal smoke test: MetaTrainer runs end-to-end on CUDA with minimal config."""
    seed_everything(123)
    device = torch.device("cuda")

    # Create minimal synthetic data
    seq1 = SyntheticSequence([1, 2, 3, 4])
    seq2 = SyntheticSequence([5, 6, 7, 8])
    train_batches = [seq1.to_batch(), seq2.to_batch()]
    val_batches = [seq1.to_batch()]

    datamodule = InMemorySequenceDataModule(
        train_batches=train_batches,
        val_batches=val_batches,
        batch_size=2,
        shuffle=False,
    )

    # Minimal model configs
    rater_config = DataRaterConfig(
        vocab_size=10,
        seq_len=4,
        d_model=16,
        n_heads=2,
        d_ff=32,
        num_layers=1,
        tau=1.0,
        w_min=0.1,
        w_max=1.0,
        dropout=0.0,
    )
    rater = DataRater(rater_config).to(device).to(torch.bfloat16)

    lm_config = TinyLMConfig(
        vocab_size=10,
        d_model=16,
        n_heads=2,
        num_layers=1,
        d_ff=32,
        max_seq_len=4,
        dropout=0.0,
    )
    language_model = TinyLM(lm_config).to(device).to(torch.bfloat16)

    # Minimal training config
    inner_config = InnerLoopConfig(steps=1, lr=0.1)
    meta_config = MetaTrainerConfig(rounds=1, population=1, val_batches=1)

    trainer = MetaTrainer(
        datamodule=datamodule,
        rater=rater,
        language_model=language_model,
        inner_config=inner_config,
        meta_config=meta_config,
    )

    # Run training
    history = trainer.run()

    # Verify history structure
    assert len(history.round_val_losses) == 1
    assert len(history.round_train_losses) == 1

    # Verify losses are finite
    assert torch.isfinite(torch.tensor(history.round_val_losses[0]))
    assert torch.isfinite(torch.tensor(history.round_train_losses[0]))

    # Verify losses are non-negative
    assert history.round_val_losses[0] >= 0.0
    assert history.round_train_losses[0] >= 0.0

