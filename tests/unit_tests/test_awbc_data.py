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

import json

import pytest
import torch
from torch.utils.data import Dataset

from rlinf.data.awbc import (
    AWBCProgressManifest,
    AWBCProgressRecord,
    AWBCSidecarDataset,
    BalancedSourceBatchSampler,
    UniformValidBatchSampler,
)


def _record(index, *, source="expert", valid=True, phi=None, phi_next=None):
    phi = index / 20 if phi is None else phi
    phi_next = phi + 0.1 if phi_next is None else phi_next
    return AWBCProgressRecord(
        dataset_index=index,
        episode_index=index // 5,
        frame_index=index,
        next_frame_index=index + 1,
        phi=phi if valid else None,
        phi_next=phi_next if valid else None,
        valid=valid,
        confidence=0.8,
        episode_length_chunks=5.0,
        source=source,
        success=False,
    )


class _MarkerDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {"marker": index}


def test_manifest_rejects_duplicate_missing_and_out_of_range_indices(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        AWBCProgressManifest([_record(0), _record(0)])
    with pytest.raises(ValueError, match="missing"):
        AWBCProgressManifest([_record(0), _record(2)], expected_dataset_size=3)
    with pytest.raises(ValueError, match="out-of-range"):
        AWBCProgressManifest(
            [_record(0), _record(1), _record(2)], expected_dataset_size=2
        )

    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps(
            {
                "dataset_index": 0,
                "episode_index": 0,
                "frame_index": 0,
                "next_frame_index": 1,
                "phi": 0.2,
                "phi_next": 0.4,
                "delta_phi": 0.7,
                "valid": True,
                "episode_length_chunks": 5,
                "source": "expert",
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="inconsistent delta_phi"):
        AWBCProgressManifest.load(path)


def test_invalid_record_does_not_turn_parser_failure_into_zero_phi():
    record = _record(0, valid=False)
    assert record.phi is None
    assert record.phi_next is None
    assert record.delta_phi == 0.0


def test_sidecar_alignment_survives_shuffled_indices():
    manifest = AWBCProgressManifest([_record(i) for i in range(6)])
    dataset = AWBCSidecarDataset(_MarkerDataset(6), manifest)
    order = [4, 1, 5, 0, 2, 3]

    for index in order:
        item = dataset[index]
        assert item["marker"] == index
        assert item["awbc_dataset_index"] == index
        assert item["awbc_delta_phi"] == pytest.approx(0.1)


def test_sidecar_requires_one_record_for_every_dataset_item():
    manifest = AWBCProgressManifest([_record(0)])
    with pytest.raises(ValueError, match="does not match"):
        AWBCSidecarDataset(_MarkerDataset(2), manifest)


def test_balanced_sampler_emits_exact_source_ratio_and_is_deterministic():
    records = [_record(i, source="expert") for i in range(2)] + [
        _record(i, source="policy") for i in range(2, 10)
    ]
    manifest = AWBCProgressManifest(records)
    sampler = BalancedSourceBatchSampler(
        manifest, batch_size=4, expert_ratio=0.5, seed=7
    )
    batches = list(sampler)

    assert len(batches) == 4
    for batch in batches:
        sources = [manifest[index].source for index in batch]
        assert sources.count("expert") == 2
        assert sources.count("policy") == 2
    assert batches == list(
        BalancedSourceBatchSampler(
            manifest, batch_size=4, expert_ratio=0.5, seed=7
        )
    )

    sampler.set_epoch(1)
    assert batches != list(sampler)


def test_balanced_sampler_rejects_missing_required_source():
    manifest = AWBCProgressManifest([_record(i) for i in range(4)])
    with pytest.raises(ValueError, match="policy records"):
        BalancedSourceBatchSampler(manifest, batch_size=4, expert_ratio=0.5)


def test_balanced_sampler_excludes_invalid_rows_by_default():
    records = [
        _record(0, source="expert", valid=True),
        _record(1, source="expert", valid=False),
        _record(2, source="policy", valid=True),
        _record(3, source="policy", valid=False),
    ]
    sampler = BalancedSourceBatchSampler(
        AWBCProgressManifest(records), batch_size=2, expert_ratio=0.5
    )

    assert list(sampler) == [[2, 0]] or list(sampler) == [[0, 2]]


def test_uniform_valid_sampler_preserves_natural_source_counts():
    records = [
        _record(0, source="expert", valid=True),
        _record(1, source="expert", valid=True),
        _record(2, source="policy", valid=True),
        _record(3, source="policy", valid=False),
    ]
    sampler = UniformValidBatchSampler(AWBCProgressManifest(records), batch_size=2, seed=3)
    assert sorted(index for batch in sampler for index in batch) == [0, 1, 2]


def test_manifest_round_trip_jsonl(tmp_path):
    records = []
    for index in range(3):
        record = _record(index, source="expert" if index == 0 else "policy")
        records.append(
            {
                "dataset_index": record.dataset_index,
                "episode_index": record.episode_index,
                "frame_index": record.frame_index,
                "next_frame_index": record.next_frame_index,
                "phi": record.phi,
                "phi_next": record.phi_next,
                "delta_phi": record.delta_phi,
                "valid": record.valid,
                "confidence": record.confidence,
                "episode_length_chunks": record.episode_length_chunks,
                "source": record.source,
                "success": record.success,
            }
        )
    path = tmp_path / "progress.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    manifest = AWBCProgressManifest.load(path, expected_dataset_size=3)
    assert len(manifest) == 3
    assert manifest[2].source == "policy"
    assert torch.tensor([manifest[i].delta_phi for i in range(3)]).shape == (3,)
