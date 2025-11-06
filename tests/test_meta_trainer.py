from __future__ import annotations

from dataclasses import dataclass
from typing import List

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


def build_datamodule() -> InMemorySequenceDataModule:
    good = SyntheticSequence([1, 1, 1, 1])
    bad = SyntheticSequence([2, 2, 2, 2])

    train_batches = [good.to_batch() for _ in range(16)] + [bad.to_batch() for _ in range(16)]
    val_batches = [good.to_batch() for _ in range(8)]

    return InMemorySequenceDataModule(train_batches=train_batches, val_batches=val_batches, batch_size=4)


def test_meta_trainer_upweights_good_sequences() -> None:
    seed_everything(0)
    datamodule = build_datamodule()

    rater_config = DataRaterConfig(
        vocab_size=8,
        seq_len=4,
        d_model=16,
        n_heads=4,
        d_ff=32,
        num_layers=2,
        tau=1.0,
        w_min=0.1,
        w_max=1.0,
        dropout=0.0,
    )
    rater = DataRater(rater_config)

    lm_config = TinyLMConfig(
        vocab_size=8,
        d_model=16,
        n_heads=4,
        num_layers=2,
        d_ff=32,
        max_seq_len=4,
        dropout=0.0,
    )
    language_model = TinyLM(lm_config)

    inner_config = InnerLoopConfig(steps=2, lr=0.5)
    meta_config = MetaTrainerConfig(rounds=4, population=1, val_batches=2)

    trainer = MetaTrainer(
        datamodule=datamodule,
        rater=rater,
        language_model=language_model,
        inner_config=inner_config,
        meta_config=meta_config,
    )

    history = trainer.run()

    assert history.round_val_losses[0] > history.round_val_losses[-1]

    with torch.no_grad():
        good_batch = datamodule.train_batches[0]
        bad_batch = datamodule.train_batches[-1]
        good_score = rater(good_batch.input_ids.unsqueeze(0)).mean().item()
        bad_score = rater(bad_batch.input_ids.unsqueeze(0)).mean().item()

    assert good_score > bad_score
