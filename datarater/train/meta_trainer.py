from __future__ import annotations

from dataclasses import dataclass, field
from itertools import cycle, islice
from typing import List

import torch

from datarater.data.datamodule import InMemorySequenceDataModule, SequenceBatch
from datarater.models.rater import DataRater
from datarater.models.tiny_lm import TinyLM
from datarater.train.inner_step import InnerLoopConfig, run_inner_loop


@dataclass(frozen=True)
class MetaTrainerConfig:
    rounds: int
    population: int
    val_batches: int
    outer_lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float | None = None


@dataclass
class MetaTrainingHistory:
    round_val_losses: List[float] = field(default_factory=list)
    round_train_losses: List[float] = field(default_factory=list)

    def append(self, train_loss: float, val_loss: float) -> None:
        self.round_train_losses.append(train_loss)
        self.round_val_losses.append(val_loss)


class MetaTrainer:
    """Outer-loop optimizer coordinating differentiable inner updates."""

    def __init__(
        self,
        datamodule: InMemorySequenceDataModule,
        rater: DataRater,
        language_model: TinyLM,
        inner_config: InnerLoopConfig,
        meta_config: MetaTrainerConfig,
    ) -> None:
        self.datamodule = datamodule
        self.rater = rater
        self.language_model = language_model
        self.inner_config = inner_config
        self.meta_config = meta_config
        self.optimizer = torch.optim.AdamW(rater.parameters(), lr=meta_config.outer_lr, weight_decay=meta_config.weight_decay)

    def _prepare_validation_batches(self) -> List[SequenceBatch]:
        val_iterable = list(self.datamodule.val_dataloader())
        if len(val_iterable) < self.meta_config.val_batches:
            raise ValueError("Not enough validation batches available")
        return list(islice(cycle(val_iterable), self.meta_config.val_batches))

    def _prepare_train_batches(self) -> List[SequenceBatch]:
        train_batches = list(self.datamodule.train_dataloader())
        if not train_batches:
            raise ValueError("No training batches available")
        return train_batches

    def run(self) -> MetaTrainingHistory:
        history = MetaTrainingHistory()
        val_batches = self._prepare_validation_batches()
        train_batches = self._prepare_train_batches()

        for _ in range(self.meta_config.rounds):
            self.optimizer.zero_grad()
            meta_loss = None
            train_loss_accum = 0.0
            train_iterator = cycle(train_batches)
            for _ in range(self.meta_config.population):
                result = run_inner_loop(
                    model=self.language_model,
                    rater=self.rater,
                    train_batches=train_iterator,
                    val_batches=val_batches,
                    config=self.inner_config,
                )
                meta_loss = result.val_loss if meta_loss is None else meta_loss + result.val_loss
                train_loss_accum += result.train_loss.item()
            assert meta_loss is not None
            meta_loss = meta_loss / float(self.meta_config.population)
            meta_loss.backward()
            if self.meta_config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.rater.parameters(), self.meta_config.grad_clip)
            self.optimizer.step()
            history.append(train_loss=train_loss_accum / float(self.meta_config.population), val_loss=meta_loss.item())
        return history
