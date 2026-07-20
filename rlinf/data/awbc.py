# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


@dataclass(frozen=True)
class AWBCProgressRecord:
    dataset_index: int
    episode_index: int
    frame_index: int
    next_frame_index: int
    phi: float | None
    phi_next: float | None
    valid: bool
    confidence: float
    episode_length_chunks: float
    source: str
    success: bool

    @property
    def delta_phi(self) -> float:
        if not self.valid or self.phi is None or self.phi_next is None:
            return 0.0
        return self.phi_next - self.phi

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AWBCProgressRecord":
        valid = bool(data["valid"])
        phi = None if data.get("phi") is None else float(data["phi"])
        phi_next = None if data.get("phi_next") is None else float(data["phi_next"])
        record = cls(
            dataset_index=int(data["dataset_index"]),
            episode_index=int(data["episode_index"]),
            frame_index=int(data["frame_index"]),
            next_frame_index=int(data.get("next_frame_index", data["frame_index"])),
            phi=phi,
            phi_next=phi_next,
            valid=valid,
            confidence=float(data.get("confidence", 1.0)),
            episode_length_chunks=float(data["episode_length_chunks"]),
            source=str(data["source"]),
            success=bool(data.get("success", False)),
        )
        record.validate()
        supplied_delta = data.get("delta_phi")
        if supplied_delta is not None and valid and not math.isclose(
            float(supplied_delta), record.delta_phi, rel_tol=1e-6, abs_tol=1e-6
        ):
            raise ValueError(
                f"AWBC record {record.dataset_index} has inconsistent delta_phi"
            )
        return record

    def validate(self) -> None:
        if self.dataset_index < 0 or self.episode_index < 0 or self.frame_index < 0:
            raise ValueError("AWBC indices must be non-negative")
        if self.next_frame_index < self.frame_index:
            raise ValueError("next_frame_index must not precede frame_index")
        if self.episode_length_chunks <= 0:
            raise ValueError("episode_length_chunks must be positive")
        if self.source not in {"expert", "policy"}:
            raise ValueError("AWBC source must be 'expert' or 'policy'")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("AWBC confidence must be in [0, 1]")
        if self.valid:
            if self.phi is None or self.phi_next is None:
                raise ValueError("valid AWBC records require phi and phi_next")
            if not 0.0 <= self.phi <= 1.0 or not 0.0 <= self.phi_next <= 1.0:
                raise ValueError("AWBC phi values must be in [0, 1]")
            if self.next_frame_index <= self.frame_index:
                raise ValueError(
                    "valid AWBC records require next_frame_index > frame_index"
                )


class AWBCProgressManifest(Sequence[AWBCProgressRecord]):
    def __init__(
        self,
        records: Sequence[AWBCProgressRecord],
        *,
        expected_dataset_size: int | None = None,
    ):
        by_index: dict[int, AWBCProgressRecord] = {}
        for record in records:
            if record.dataset_index in by_index:
                raise ValueError(
                    f"duplicate AWBC dataset_index={record.dataset_index}"
                )
            by_index[record.dataset_index] = record

        if expected_dataset_size is not None:
            expected = set(range(expected_dataset_size))
            actual = set(by_index)
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            if missing:
                raise ValueError(f"AWBC manifest is missing dataset indices: {missing[:8]}")
            if extra:
                raise ValueError(f"AWBC manifest has out-of-range indices: {extra[:8]}")

        self._records = tuple(by_index[index] for index in sorted(by_index))
        self._by_index = by_index

    @classmethod
    def load(
        cls, path: str | Path, *, expected_dataset_size: int | None = None
    ) -> "AWBCProgressManifest":
        path = Path(path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"AWBC progress manifest does not exist: {path}")
        with path.open(encoding="utf-8") as file:
            if path.suffix == ".jsonl":
                raw_records = [json.loads(line) for line in file if line.strip()]
            else:
                raw_records = json.load(file)
        if not isinstance(raw_records, list):
            raise ValueError("AWBC progress manifest must contain a list of records")
        return cls(
            [AWBCProgressRecord.from_dict(item) for item in raw_records],
            expected_dataset_size=expected_dataset_size,
        )

    def __getitem__(self, index: int) -> AWBCProgressRecord:
        try:
            return self._by_index[index]
        except KeyError as exc:
            raise IndexError(f"AWBC manifest has no dataset_index={index}") from exc

    def __len__(self) -> int:
        return len(self._records)


class AWBCSidecarDataset(Dataset):
    def __init__(self, dataset: Dataset, manifest: AWBCProgressManifest):
        if len(dataset) != len(manifest):
            raise ValueError(
                f"AWBC manifest size {len(manifest)} does not match dataset size {len(dataset)}"
            )
        self.dataset = dataset
        self.manifest = manifest

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.dataset[index]
        if not isinstance(item, dict):
            raise TypeError("AWBC OpenPI dataset items must be dictionaries")
        record = self.manifest[index]
        result = dict(item)
        result.update(
            {
                "awbc_dataset_index": np.int64(record.dataset_index),
                "awbc_delta_phi": np.float32(record.delta_phi),
                "awbc_episode_length": np.float32(record.episode_length_chunks),
                "awbc_valid": np.bool_(record.valid),
                "awbc_confidence": np.float32(record.confidence),
                "awbc_source": np.int64(1 if record.source == "expert" else 0),
            }
        )
        return result


class BalancedSourceBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        manifest: AWBCProgressManifest,
        batch_size: int,
        expert_ratio: float,
        *,
        seed: int = 0,
        valid_only: bool = True,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0.0 <= expert_ratio <= 1.0:
            raise ValueError("expert_ratio must be in [0, 1]")
        self.batch_size = batch_size
        self.expert_count = int(round(batch_size * expert_ratio))
        self.policy_count = batch_size - self.expert_count
        self.seed = seed
        self.epoch = 0
        self.expert_indices = [
            record.dataset_index
            for record in manifest
            if record.source == "expert" and (record.valid or not valid_only)
        ]
        self.policy_indices = [
            record.dataset_index
            for record in manifest
            if record.source == "policy" and (record.valid or not valid_only)
        ]
        if self.expert_count and not self.expert_indices:
            raise ValueError("expert_ratio requires expert records in the AWBC manifest")
        if self.policy_count and not self.policy_indices:
            raise ValueError("expert_ratio requires policy records in the AWBC manifest")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        counts = []
        if self.expert_count:
            counts.append(math.ceil(len(self.expert_indices) / self.expert_count))
        if self.policy_count:
            counts.append(math.ceil(len(self.policy_indices) / self.policy_count))
        return max(counts)

    @staticmethod
    def _draw(
        pool: list[int], count: int, generator: torch.Generator
    ) -> list[int]:
        if count == 0:
            return []
        output = []
        while len(output) < count:
            permutation = torch.randperm(len(pool), generator=generator).tolist()
            output.extend(pool[index] for index in permutation)
        return output[:count]

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        expert = self._draw(
            self.expert_indices, len(self) * self.expert_count, generator
        )
        policy = self._draw(
            self.policy_indices, len(self) * self.policy_count, generator
        )
        for batch_index in range(len(self)):
            batch = expert[
                batch_index * self.expert_count : (batch_index + 1) * self.expert_count
            ]
            batch += policy[
                batch_index * self.policy_count : (batch_index + 1) * self.policy_count
            ]
            order = torch.randperm(len(batch), generator=generator).tolist()
            yield [batch[index] for index in order]


class UniformValidBatchSampler(Sampler[list[int]]):
    """Shuffle all eligible rows without enforcing a source ratio."""

    def __init__(
        self,
        manifest: AWBCProgressManifest,
        batch_size: int,
        *,
        seed: int = 0,
        valid_only: bool = True,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.indices = [
            record.dataset_index for record in manifest if record.valid or not valid_only
        ]
        if not self.indices:
            raise ValueError("AWBC sampler has no eligible records")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return math.ceil(len(self.indices) / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.indices), generator=generator).tolist()
        shuffled = [self.indices[index] for index in order]
        for start in range(0, len(shuffled), self.batch_size):
            yield shuffled[start : start + self.batch_size]


class AWBCOpenPiDataLoader:
    """Preserve AWBC metadata that OpenPI's DataLoaderImpl normally drops."""

    _METADATA_KEYS = (
        "awbc_dataset_index",
        "awbc_delta_phi",
        "awbc_episode_length",
        "awbc_valid",
        "awbc_confidence",
        "awbc_source",
    )

    def __init__(self, openpi_data_loader: Any):
        self._openpi_data_loader = openpi_data_loader
        self._data_loader = openpi_data_loader._data_loader

    def data_config(self):
        return self._openpi_data_loader.data_config()

    def __iter__(self):
        from openpi.models import model as openpi_model

        for batch in self._data_loader:
            result = {
                "observation": openpi_model.Observation.from_dict(batch),
                "actions": batch["actions"],
            }
            result.update({key: batch[key] for key in self._METADATA_KEYS})
            yield result


def attach_awbc_to_openpi_dataloader(
    openpi_data_loader: Any,
    *,
    manifest_path: str | Path,
    expert_sampling_ratio: float | None,
    seed: int,
    dataset_override: Dataset | None = None,
    valid_only: bool = True,
) -> AWBCOpenPiDataLoader:
    torch_wrapper = getattr(openpi_data_loader, "_data_loader", None)
    pytorch_loader = getattr(torch_wrapper, "_data_loader", None) or getattr(
        torch_wrapper, "torch_loader", None
    )
    if pytorch_loader is None:
        raise TypeError("OpenPI dataloader does not expose its PyTorch DataLoader")

    base_dataset = dataset_override or pytorch_loader.dataset
    manifest = AWBCProgressManifest.load(
        manifest_path, expected_dataset_size=len(base_dataset)
    )
    dataset = AWBCSidecarDataset(base_dataset, manifest)
    batch_size = pytorch_loader.batch_size
    if batch_size is None:
        batch_size = getattr(pytorch_loader.batch_sampler, "batch_size", None)
    if batch_size is None:
        raise TypeError("Cannot determine OpenPI AWBC batch size")
    if expert_sampling_ratio is None:
        batch_sampler: Sampler[list[int]] = UniformValidBatchSampler(
            manifest,
            int(batch_size),
            seed=seed,
            valid_only=valid_only,
        )
    else:
        batch_sampler = BalancedSourceBatchSampler(
            manifest,
            int(batch_size),
            expert_sampling_ratio,
            seed=seed,
            valid_only=valid_only,
        )

    kwargs: dict[str, Any] = {
        "batch_sampler": batch_sampler,
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

    torch_wrapper._data_loader = DataLoader(dataset, **kwargs)
    return AWBCOpenPiDataLoader(openpi_data_loader)
