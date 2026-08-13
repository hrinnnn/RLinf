import torch
from torch.utils.data import Dataset

from rlinf.data.openpi_mixture import (
    SourceBalancedBatchSampler,
)
from rlinf.algorithms.awbc import weighted_flow_matching_loss
from rlinf.models.embodiment.openpi.policies.maniskill_policy import (
    action_valid_mask_from_padding,
)


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


def test_temporal_mask_keeps_final_anchor_with_one_real_target():
    mask = action_valid_mask_from_padding([False, True, True, True])
    assert mask.tolist() == [True, False, False, False]


def test_masked_flow_loss_ignores_repeated_terminal_actions():
    element_loss = torch.tensor([[[1.0], [3.0], [100.0], [200.0]]])
    mask = torch.tensor([[True, True, False, False]])
    loss, per_sample = weighted_flow_matching_loss(
        element_loss, element_mask=mask
    )
    assert loss.item() == 2.0
    assert per_sample.item() == 2.0
