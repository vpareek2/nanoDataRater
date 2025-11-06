from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence

import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class SequenceBatch:
    """A batch of tokenized sequences for language modelling."""

    input_ids: torch.Tensor
    target_ids: torch.Tensor

    def to(self, device: torch.device | str) -> "SequenceBatch":
        return SequenceBatch(input_ids=self.input_ids.to(device), target_ids=self.target_ids.to(device))


class InMemorySequenceDataset(Dataset[SequenceBatch]):
    """Simple dataset backed by a list of :class:`SequenceBatch` records."""

    def __init__(self, records: Sequence[SequenceBatch]) -> None:
        self._records: List[SequenceBatch] = list(records)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> SequenceBatch:
        return self._records[index]


def _collate_sequence_batches(records: Sequence[SequenceBatch]) -> SequenceBatch:
    inputs = torch.stack([record.input_ids for record in records], dim=0)
    targets = torch.stack([record.target_ids for record in records], dim=0)
    return SequenceBatch(input_ids=inputs, target_ids=targets)


class InMemorySequenceDataModule:
    """DataModule that serves pre-constructed batches for the meta-learning loops."""

    def __init__(
        self,
        train_batches: Sequence[SequenceBatch],
        val_batches: Sequence[SequenceBatch],
        batch_size: int,
        shuffle: bool = True,
    ) -> None:
        self.train_batches = list(train_batches)
        self.val_batches = list(val_batches)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def _dataloader(self, records: Sequence[SequenceBatch], shuffle: bool) -> DataLoader[SequenceBatch]:
        dataset = InMemorySequenceDataset(records)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, collate_fn=_collate_sequence_batches)

    def train_dataloader(self) -> Iterator[SequenceBatch]:
        yield from self._dataloader(self.train_batches, shuffle=self.shuffle)

    def val_dataloader(self) -> Iterator[SequenceBatch]:
        yield from self._dataloader(self.val_batches, shuffle=False)


def load_jsonl_sequences(path_glob: str, input_key: str = "input_ids", target_key: str = "target_ids") -> List[SequenceBatch]:
    """Load token sequences from JSONL files into :class:`SequenceBatch` records."""

    batches: List[SequenceBatch] = []
    for path in sorted(Path().glob(path_glob)):
        if not path.is_file():
            continue
        frame = pl.read_ndjson(path)
        for row in frame.iter_rows():
            record = row.as_dict()
            inputs = torch.tensor(record[input_key], dtype=torch.long)
            targets = torch.tensor(record[target_key], dtype=torch.long)
            batches.append(SequenceBatch(input_ids=inputs, target_ids=targets))
    return batches
