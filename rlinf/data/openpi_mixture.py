"""Deterministic source-balanced batching for ordinary OpenPI SFT.

This is deliberately independent of AWBC metadata.  It is used when a
DAgger-style expert suffix dataset must be mixed with a fixed expert replay
without changing the SFT loss.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Iterator, Sequence
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler


class ValidActionHorizonDataset(Dataset):
    """Expose only starts with an entirely in-episode action target.

    LeRobot clamps future action indices at episode boundaries and emits an
    ``actions_is_pad`` flag.  OpenPI's transformed sample deliberately drops
    that flag, so ordinary flow-matching SFT would otherwise optimize against
    repeated terminal actions.  This wrapper filters those starts before a
    batch reaches the model.
    """

    def __init__(self, dataset: Dataset, *, action_horizon: int):
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        self.dataset = dataset
        self.action_horizon = int(action_horizon)
        raw_dataset = dataset
        while hasattr(raw_dataset, "_dataset"):
            raw_dataset = raw_dataset._dataset
        episode_index = getattr(raw_dataset, "episode_data_index", None)
        if (
            episode_index is None
            or "from" not in episode_index
            or "to" not in episode_index
        ):
            raise TypeError(
                "valid action-horizon filtering requires a LeRobot-style "
                "dataset with episode_data_index"
            )

        valid_indices: list[int] = []
        for start, end in zip(episode_index["from"], episode_index["to"], strict=True):
            start = int(start)
            end = int(end)
            # ``end`` is exclusive.  An anchor at i needs actions i..i+H-1.
            valid_indices.extend(range(start, max(start, end - self.action_horizon + 1)))
        if not valid_indices:
            raise ValueError(
                f"no episode contains a complete {self.action_horizon}-step action target"
            )
        self.raw_indices = tuple(valid_indices)

    def __len__(self) -> int:
        return len(self.raw_indices)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[self.raw_indices[index]]


class ActionHorizonMaskDataset(Dataset):
    """Attach a temporal validity mask without removing episode-tail anchors.

    LeRobot uses ``actions_is_pad`` when a requested action horizon crosses an
    episode boundary. OpenPI's ``RepackTransform`` creates a smaller
    model-input dictionary and drops that metadata. This wrapper reconstructs
    the same validity semantics from LeRobot's episode ranges after the
    OpenPI transforms have run.
    """

    def __init__(self, dataset: Dataset, *, action_horizon: int):
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        self.dataset = dataset
        self.action_horizon = int(action_horizon)
        raw_dataset = dataset
        while hasattr(raw_dataset, "_dataset"):
            raw_dataset = raw_dataset._dataset
        episode_index = getattr(raw_dataset, "episode_data_index", None)
        if episode_index is None or "from" not in episode_index or "to" not in episode_index:
            raise TypeError(
                "action-horizon masking requires a LeRobot-style dataset with "
                "episode_data_index"
            )
        self._starts = tuple(int(value) for value in episode_index["from"])
        self._ends = tuple(int(value) for value in episode_index["to"])
        if not self._starts or len(self._starts) != len(self._ends):
            raise ValueError("invalid LeRobot episode_data_index")

    def __len__(self) -> int:
        return len(self.dataset)

    def _episode_end(self, raw_index: int) -> int:
        episode = bisect_right(self._starts, raw_index) - 1
        if episode < 0 or raw_index >= self._ends[episode]:
            raise IndexError(f"dataset index {raw_index} does not belong to an episode")
        return self._ends[episode]

    def __getitem__(self, index: int) -> Any:
        raw_index = int(index)
        sample = self.dataset[raw_index]
        if not isinstance(sample, dict):
            raise TypeError("OpenPI transformed samples must be dictionaries")
        if "action_valid_mask" in sample:
            raise KeyError("sample already contains action_valid_mask")
        valid_steps = max(
            0, min(self.action_horizon, self._episode_end(raw_index) - raw_index)
        )
        sample = dict(sample)
        # This is the inverse of LeRobot's actions_is_pad for this horizon.
        sample["action_valid_mask"] = torch.arange(self.action_horizon) < valid_steps
        return sample


class OpenPIActionMaskDataLoader:
    """Yield OpenPI batches as dictionaries while preserving the temporal mask.

    ``openpi.training.data_loader.DataLoaderImpl`` converts a transformed
    dictionary into ``(Observation, actions)`` and therefore cannot forward
    extra dataset metadata. This narrow adapter uses the same internal torch
    loader and the same ``Observation.from_dict`` conversion, but keeps
    ``action_valid_mask`` alongside the official model inputs.
    """

    def __init__(self, openpi_data_loader: Any):
        self._openpi_data_loader = openpi_data_loader

    @property
    def pytorch_loader(self) -> DataLoader:
        _, loader = _unwrap_pytorch_loader(self._openpi_data_loader)
        return loader

    def data_config(self) -> Any:
        return self._openpi_data_loader.data_config()

    def __iter__(self) -> Iterator[dict[str, Any]]:
        from openpi.models import model as openpi_model

        for batch in self._openpi_data_loader._data_loader:
            if "action_valid_mask" not in batch:
                raise KeyError("masked OpenPI batch is missing action_valid_mask")
            yield {
                "observation": openpi_model.Observation.from_dict(batch),
                "actions": batch["actions"],
                "action_valid_mask": batch["action_valid_mask"],
            }


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
