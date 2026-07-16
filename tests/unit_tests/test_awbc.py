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

import pytest
import torch

from rlinf.algorithms.awbc import (
    compute_arm_awbc_weights,
    weighted_flow_matching_loss,
)


def test_arm_formula_matches_hand_calculation():
    result = compute_arm_awbc_weights(
        delta_phi=[0.1, 0.2, 0.3],
        episode_lengths=[10.0, 10.0, 10.0],
    )

    gains = torch.tensor([0.1, 0.2, 0.3])
    mean = gains.mean()
    std = gains.std(correction=0)
    expected = ((gains - (mean - 2 * std)) / (4 * std + 1e-6)).clamp(0, 1)
    torch.testing.assert_close(result.gains, gains)
    torch.testing.assert_close(result.weights, expected)


def test_episode_length_scaling_uses_valid_batch_mean():
    result = compute_arm_awbc_weights(
        delta_phi=[0.2, 0.2, 99.0],
        episode_lengths=[10.0, 20.0, 100.0],
        valid=[True, True, False],
    )

    assert result.mean_episode_length.item() == pytest.approx(15.0)
    torch.testing.assert_close(
        result.gains, torch.tensor([2.0 / 15.0, 4.0 / 15.0, 660.0])
    )
    assert result.valid_count == 2
    assert result.weights[2].item() == 0.0


@pytest.mark.parametrize(
    ("valid", "expected"),
    [
        ([False, False], [0.0, 0.0]),
        ([True, False], [1.0, 0.0]),
        ([True, True], [1.0, 1.0]),
    ],
)
def test_degenerate_batches_fallback_without_nan(valid, expected):
    result = compute_arm_awbc_weights(
        delta_phi=[0.2, 0.2], episode_lengths=[10, 10], valid=valid
    )

    assert result.used_fallback
    assert not torch.isnan(result.weights).any()
    torch.testing.assert_close(result.weights, torch.tensor(expected))


def test_negative_progress_is_continuous_in_exact_mode():
    result = compute_arm_awbc_weights(
        delta_phi=[-0.1, 0.0, 0.2], episode_lengths=[10, 10, 10]
    )
    assert 0.0 < result.weights[0].item() < result.weights[1].item()


def test_robust_controls_zero_negative_and_apply_confidence_and_floor():
    result = compute_arm_awbc_weights(
        delta_phi=[-0.1, 0.0, 0.2],
        episode_lengths=[10, 10, 10],
        confidence=[1.0, 0.25, 1.0],
        confidence_power=1.0,
        weight_floor=0.1,
        negative_delta_policy="zero",
    )

    assert result.weights[0].item() == 0.0
    assert result.weights[1].item() >= 0.1
    assert result.weights[1].item() < result.weights[2].item()


def test_preaggregated_global_statistics_match_combined_batch():
    global_delta = torch.tensor([0.0, 0.1, 0.2, 0.3])
    global_lengths = torch.tensor([5.0, 10.0, 15.0, 20.0])
    combined = compute_arm_awbc_weights(global_delta, global_lengths)

    rank0 = compute_arm_awbc_weights(
        global_delta[:2],
        global_lengths[:2],
        global_delta_phi=global_delta,
        global_episode_lengths=global_lengths,
    )
    rank1 = compute_arm_awbc_weights(
        global_delta[2:],
        global_lengths[2:],
        global_delta_phi=global_delta,
        global_episode_lengths=global_lengths,
    )

    torch.testing.assert_close(
        torch.cat([rank0.weights, rank1.weights]), combined.weights
    )
    assert rank0.gain_mean.item() == pytest.approx(rank1.gain_mean.item())
    assert rank0.gain_std.item() == pytest.approx(rank1.gain_std.item())


def test_all_one_weights_equal_previous_global_mean_loss():
    element_loss = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    weighted, per_sample = weighted_flow_matching_loss(
        element_loss, torch.ones(2)
    )

    torch.testing.assert_close(weighted, element_loss.mean())
    torch.testing.assert_close(per_sample, element_loss.mean(dim=(1, 2)))


def test_zero_weight_sample_has_no_gradient_contribution():
    element_loss = torch.tensor(
        [[[1.0, 3.0]], [[10.0, 20.0]]], requires_grad=True
    )
    loss, _ = weighted_flow_matching_loss(element_loss, torch.tensor([1.0, 0.0]))
    loss.backward()

    torch.testing.assert_close(element_loss.grad[0], torch.full((1, 2), 0.5))
    torch.testing.assert_close(element_loss.grad[1], torch.zeros((1, 2)))


def test_all_zero_weights_return_differentiable_zero():
    element_loss = torch.ones((2, 3, 4), requires_grad=True)
    loss, _ = weighted_flow_matching_loss(element_loss, torch.zeros(2))
    loss.backward()

    assert loss.item() == 0.0
    torch.testing.assert_close(element_loss.grad, torch.zeros_like(element_loss))


def test_input_validation_rejects_bad_shapes_and_lengths():
    with pytest.raises(ValueError, match="same length"):
        compute_arm_awbc_weights([0.1], [10, 20])
    with pytest.raises(ValueError, match="positive"):
        compute_arm_awbc_weights([0.1], [0])
    with pytest.raises(ValueError, match="batch dimension"):
        weighted_flow_matching_loss(torch.tensor(1.0), torch.tensor([1.0]))
