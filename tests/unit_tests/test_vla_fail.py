"""Tests for the training-free VLA-FAIL score primitives."""

from __future__ import annotations

import pytest
import torch

from rlinf.algorithms.vla_fail import (
    LLMDStatistics,
    assert_threshold_statistics_compatible,
    constant_split_conformal_threshold,
    failure_alert,
    fixed_gaussian_prior,
    fit_llmd_statistics,
    llmd_score,
    llmd_token_scores,
    velocity_normalized_acc,
)


def test_llmd_is_tokenwise_and_uses_maximum_token_score() -> None:
    features = torch.tensor(
        [
            [[0.0], [0.0]],
            [[2.0], [2.0]],
        ]
    )
    stats = fit_llmd_statistics(features, ridge=1e-6)

    scores = llmd_token_scores(torch.tensor([[[1.0], [3.0]]]), stats)

    assert scores.shape == (1, 2)
    assert scores[0, 1] > scores[0, 0]
    torch.testing.assert_close(llmd_score(torch.tensor([[[1.0], [3.0]]]), stats), scores.max(dim=1).values)


def test_llmd_rejects_hidden_or_token_shape_mismatch() -> None:
    stats = fit_llmd_statistics(torch.randn(3, 2, 4))

    with pytest.raises(ValueError, match="does not match"):
        llmd_score(torch.randn(1, 3, 4), stats)


def test_statistics_round_trip_preserves_scores() -> None:
    stats = fit_llmd_statistics(torch.randn(4, 2, 3), ridge=1e-4)
    reloaded = LLMDStatistics.from_state_dict(stats.state_dict())
    query = torch.randn(2, 2, 3)

    torch.testing.assert_close(llmd_score(query, stats), llmd_score(query, reloaded))


def test_fixed_prior_is_reproducible_and_has_one_shared_batch() -> None:
    first = fixed_gaussian_prior(action_horizon=10, action_dim=8, seed=7)
    second = fixed_gaussian_prior(action_horizon=10, action_dim=8, seed=7)

    assert first.shape == (1, 10, 8)
    torch.testing.assert_close(first, second)


def test_constant_split_conformal_uses_trajectory_maxima_and_order_statistic() -> None:
    traces = [[0.1, 1.0], [0.2, 3.0], [0.3, 2.0], [0.4, 4.0]]

    # ceil((4 + 1) * .75) = 4, so q_{.75} is the fourth trajectory maximum.
    assert constant_split_conformal_threshold(traces, delta=0.25) == pytest.approx(4.0)


def test_acc_compares_old_suffix_with_new_prefix_and_ema() -> None:
    previous = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
    )
    current = torch.tensor(
        [[2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0], [5.0, 5.0, 5.0]]
    )

    raw, ema = velocity_normalized_acc(
        previous,
        current,
        execute_horizon=2,
        min_velocity=1e-3,
        previous_ema=0.5,
        ema_alpha=0.9,
    )

    assert raw == pytest.approx(0.0)
    assert ema == pytest.approx(0.45)


def test_acc_requires_absolute_three_dimensional_points() -> None:
    with pytest.raises(ValueError, match=r"\[horizon, 3\]"):
        velocity_normalized_acc(
            torch.zeros(4, 8), torch.zeros(4, 8), execute_horizon=2, min_velocity=1e-3
        )


def test_acc_velocity_uses_only_the_overlapping_current_prefix() -> None:
    previous = torch.zeros(4, 3)
    # The overlap prefix has range 1, while the unexecuted suffix jumps to 100.
    current = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [100.0, 100.0, 100.0], [100.0, 100.0, 100.0]]
    )
    previous[2:] = 3.0

    raw, _ = velocity_normalized_acc(
        previous, current, execute_horizon=2, min_velocity=1e-3
    )

    # |3 - 0| and |3 - 1| averaged over the two overlap points, divided by 1.
    assert raw == pytest.approx(2.5)


def test_failure_fusion_is_logical_or() -> None:
    assert failure_alert(llmd_value=3.0, llmd_threshold=2.0, acc_value=0.0, acc_threshold=1.0)
    assert failure_alert(llmd_value=0.0, llmd_threshold=2.0, acc_value=2.0, acc_threshold=1.0)
    assert not failure_alert(llmd_value=0.0, llmd_threshold=2.0, acc_value=0.5, acc_threshold=1.0)


def test_threshold_manifest_rejects_statistics_from_another_detector_asset() -> None:
    assert_threshold_statistics_compatible({}, "current")
    assert_threshold_statistics_compatible({"llmd_statistics_sha256": "current"}, "current")
    with pytest.raises(ValueError, match="different LLMD statistics"):
        assert_threshold_statistics_compatible({"llmd_statistics_sha256": "old"}, "current")
