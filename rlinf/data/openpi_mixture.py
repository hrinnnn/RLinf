"""Deterministic source-balanced batching for ordinary OpenPI SFT.

This is deliberately independent of AWBC metadata.  It is used when a
DAgger-style expert suffix dataset must be mixed with a fixed expert replay
without changing the SFT loss.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler


class MaskPreservingOpenPIDataLoader:
    """Expose LeRobot temporal padding metadata to the RLinf SFT worker."""

    def __init__(self, openpi_data_loader: Any):
        self._openpi_data_loader = openpi_data_loader

    def data_config(self):
        return self._openpi_data_loader.data_config()

    def __getattr__(self, name: str):
        return getattr(self._openpi_data_loader, name)

    def __iter__(self):
        from openpi.models import model as _model

        torch_loader = self._openpi_data_loader._data_loader
        for batch in torch_loader:
            observation = _model.Observation.from_dict(batch)
            yield observation, batch["actions"], batch.get("action_valid_mask")


class SourceBalancedBatchSampler(Sampler[list[int]]):
    """Draw an exact, shuffled number of examples from every source per batch."""

    def __init__(self, source_lengths: Sequence[int], batch_size: int, *, seed: int = 0):
        if not source_lengths or any(int(length) <= 0 for length in source_lengths):
            raise ValueError("every source must be non-empty")
        if batch_size <= 0 or batch_size % len(source_lengths):
            raise ValueError("batch_size must divide evenly across all sources")
        self.source_lengths = tuple(int(length) for length in source_lengths)
        self.batch_size = int(batch_size)
        self.per_source = self.batch_size // len(self.source_lengths)
        self.seed = int(seed)
        self.epoch = 0
        offsets = [0]
        for length in self.source_lengths:
            offsets.append(offsets[-1] + length)
        self._offsets = tuple(offsets)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return max(math.ceil(length / self.per_source) for length in self.source_lengths)

    @staticmethod
    def _draw(length: int, count: int, generator: torch.Generator) -> list[int]:
        output: list[int] = []
        while len(output) < count:
            output.extend(torch.randperm(length, generator=generator).tolist())
        return output[:count]

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        total_per_source = len(self) * self.per_source
        draws = [
            self._draw(length, total_per_source, generator)
            for length in self.source_lengths
        ]
        for batch_index in range(len(self)):
            batch = []
            for source_index, source_draws in enumerate(draws):
                start = batch_index * self.per_source
                batch.extend(
                    self._offsets[source_index] + item
                    for item in source_draws[start : start + self.per_source]
                )
            order = torch.randperm(len(batch), generator=generator).tolist()
            yield [batch[index] for index in order]


def _unwrap_pytorch_loader(openpi_data_loader: Any) -> tuple[Any, DataLoader]:
    torch_wrapper = getattr(openpi_data_loader, "_data_loader", None)
    pytorch_loader = getattr(torch_wrapper, "_data_loader", None) or getattr(
        torch_wrapper, "torch_loader", None
    )
    if torch_wrapper is None or not isinstance(pytorch_loader, DataLoader):
        raise TypeError("OpenPI dataloader does not expose its PyTorch DataLoader")
    return torch_wrapper, pytorch_loader


def attach_source_balanced_openpi_dataloader(
    openpi_data_loader: Any,
    *,
    datasets: Sequence[Dataset],
    seed: int,
) -> Any:
    """Replace OpenPI's inner loader with exact source-balanced ordinary BC batches."""

    if len(datasets) < 2:
        raise ValueError("source-balanced OpenPI SFT needs at least two datasets")
    torch_wrapper, pytorch_loader = _unwrap_pytorch_loader(openpi_data_loader)
    batch_size = pytorch_loader.batch_size
    if batch_size is None:
        batch_size = getattr(pytorch_loader.batch_sampler, "batch_size", None)
    if batch_size is None:
        raise TypeError("Cannot determine OpenPI SFT micro-batch size")
    sampler = SourceBalancedBatchSampler(
        [len(dataset) for dataset in datasets], int(batch_size), seed=seed
    )
    kwargs: dict[str, Any] = {
        "batch_sampler": sampler,
        "num_workers": pytorch_loader.num_workers,
        "collate_fn": pytorch_loader.collate_fn,
        "pin_memory": pytorch_loader.pin_memory,
        "timeout": pytorch_loader.timeout,
        "worker_init_fn": pytorch_loader.worker_init_fn,
    }
    if pytorch_loader.num_workers > 0:
        kwargs["persistent_workers"] = pytorch_loader.persistent_workers
        kwargs["multiprocessing_context"] = pytorch_loader.multiprocessing_context
        if pytorch_loader.prefetch_factor is not None:
            kwargs["prefetch_factor"] = pytorch_loader.prefetch_factor
    torch_wrapper._data_loader = DataLoader(ConcatDataset(list(datasets)), **kwargs)
    return openpi_data_loader
