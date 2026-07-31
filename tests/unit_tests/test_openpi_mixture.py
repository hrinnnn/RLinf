import torch
from torch.utils.data import Dataset

from rlinf.data.openpi_mixture import (
    SourceBalancedBatchSampler,
    ValidActionHorizonDataset,
)


class _RawEpisodeDataset(Dataset):
    def __init__(self):
        self.episode_data_index = {
            "from": torch.tensor([0, 20]),
            "to": torch.tensor([20, 50]),
        }

    def __len__(self):
        return 50

    def __getitem__(self, index):
        return index


class _TransformWrapper(Dataset):
    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        return {"source_index": self._dataset[index]}


def test_source_balanced_batches_have_equal_source_counts():
    sampler = SourceBalancedBatchSampler([3, 7], batch_size=4, seed=7)
    batches = list(sampler)
    assert len(batches) == 4
    for batch in batches:
        assert len(batch) == 4
        assert sum(index < 3 for index in batch) == 2
        assert sum(index >= 3 for index in batch) == 2


def test_source_balanced_sampler_is_reproducible_per_epoch():
    sampler = SourceBalancedBatchSampler([5, 5], batch_size=4, seed=11)
    first = list(sampler)
    assert first == list(sampler)
    sampler.set_epoch(1)
    assert first != list(sampler)


def test_valid_action_horizon_dataset_excludes_episode_tail_padding():
    dataset = ValidActionHorizonDataset(
        _TransformWrapper(_RawEpisodeDataset()), action_horizon=10
    )
    # Episode [0, 20) contributes starts 0..10; [20, 50) contributes 20..40.
    assert dataset.raw_indices == tuple(range(0, 11)) + tuple(range(20, 41))
    assert len(dataset) == 32
    assert dataset[10] == {"source_index": 10}
    assert dataset[11] == {"source_index": 20}
