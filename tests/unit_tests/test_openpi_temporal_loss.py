import pytest
import torch

from rlinf.algorithms.awbc import weighted_flow_matching_loss


def test_temporal_mask_excludes_padded_action_positions():
    element_loss = torch.tensor(
        [
            [[1.0, 3.0], [100.0, 100.0], [100.0, 100.0]],
            [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]],
        ]
    )
    mask = torch.tensor([[True, False, False], [True, True, True]])
    loss, per_sample = weighted_flow_matching_loss(element_loss, element_mask=mask)
    assert torch.allclose(per_sample, torch.tensor([2.0, 7.0]))
    assert torch.allclose(loss, torch.tensor(4.5))


def test_temporal_mask_and_sample_weights_preserve_weighted_reduction():
    element_loss = torch.tensor([[[2.0], [99.0]], [[4.0], [6.0]]])
    mask = torch.tensor([[True, False], [True, True]])
    loss, per_sample = weighted_flow_matching_loss(
        element_loss, torch.tensor([1.0, 3.0]), element_mask=mask
    )
    assert torch.allclose(per_sample, torch.tensor([2.0, 5.0]))
    assert torch.allclose(loss, torch.tensor(4.25))


def test_temporal_mask_rejects_samples_without_a_real_action():
    with pytest.raises(ValueError, match="valid action target"):
        weighted_flow_matching_loss(
            torch.ones(1, 2, 3), element_mask=torch.zeros(1, 2, dtype=torch.bool)
        )
