"""Tests for the training-free VLA-FAIL score primitives."""

from __future__ import annotations

import pytest
import torch

from rlinf.algorithms.vla_fail import (
    KNNStatistics,
    LLMDStatistics,
    PCAResidualStatistics,
    assert_threshold_statistics_compatible,
    constant_split_conformal_threshold,
    failure_alert,
    fit_knn_statistics,
    fit_llmd_statistics,
    fit_pca_residual_statistics,
    fixed_gaussian_prior,
    knn_score,
    knn_token_scores,
    llmd_score,
    llmd_token_scores,
    pca_residual_score,
    pool_valid_prefix_tokens,
    resolve_feature_probe_indices,
    velocity_normalized_acc,
    vim_default_principal_dim,
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


def test_multilayer_probe_fractions_select_completed_transformer_blocks() -> None:
    assert resolve_feature_probe_indices(20, (0.25, 0.5, 0.75)) == (4, 9, 14)
    # Fractions that round to the same layer must not produce duplicate probes.
    assert resolve_feature_probe_indices(3, (0.25, 0.5, 0.75, 1.0)) == (0, 1, 2)
    with pytest.raises(ValueError, match="fraction"):
        resolve_feature_probe_indices(4, (0.0,))


def test_vlm_bridge_pool_uses_only_valid_prefix_tokens() -> None:
    hidden = torch.tensor(
        [
            [[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]],
            [[2.0, 4.0], [4.0, 8.0], [6.0, 10.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [0, 1, 1]])

    pooled = pool_valid_prefix_tokens(hidden, mask)

    assert pooled.shape == (2, 1, 2)
    torch.testing.assert_close(pooled[0, 0], torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(pooled[1, 0], torch.tensor([5.0, 9.0]))


def test_vlm_bridge_pool_rejects_misaligned_mask() -> None:
    with pytest.raises(ValueError, match="must match"):
        pool_valid_prefix_tokens(torch.zeros(1, 2, 3), torch.ones(1, 3))


def test_knn_uses_normalized_kth_squared_l2_distance_and_token_maximum() -> None:
    features = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
            [[-1.0, 0.0], [0.0, -1.0]],
        ]
    )
    stats = fit_knn_statistics(features, k=2)
    query = torch.tensor([[[1.0, 0.0], [0.0, -2.0]]])

    token_scores = knn_token_scores(query, stats)

    # Both token positions have squared distances {0, 2, 4} to their ID bank,
    # so the official second-nearest score is 2 for each position.
    torch.testing.assert_close(token_scores, torch.tensor([[2.0, 2.0]]))
    torch.testing.assert_close(knn_score(query, stats), torch.tensor([2.0]))
    torch.testing.assert_close(
        knn_score(query, KNNStatistics.from_state_dict(stats.state_dict())), torch.tensor([2.0])
    )


def test_pca_residual_is_zero_in_principal_subspace_and_positive_outside() -> None:
    features = torch.tensor(
        [
            [[-2.0, 0.0, 0.0]],
            [[-1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0]],
        ]
    )
    stats = fit_pca_residual_statistics(features, principal_dim=1)
    in_subspace = torch.tensor([[[3.0, 0.0, 0.0]]])
    outside = torch.tensor([[[0.0, 4.0, 0.0]]])

    assert pca_residual_score(in_subspace, stats).item() == pytest.approx(0.0, abs=1e-6)
    assert pca_residual_score(outside, stats).item() == pytest.approx(4.0, abs=1e-6)
    torch.testing.assert_close(
        pca_residual_score(outside, PCAResidualStatistics.from_state_dict(stats.state_dict())),
        pca_residual_score(outside, stats),
    )


def test_vim_default_principal_dimension_matches_official_dimension_rule() -> None:
    assert vim_default_principal_dim(2048) == 1000
    assert vim_default_principal_dim(1024) == 512
    assert vim_default_principal_dim(256) == 128
