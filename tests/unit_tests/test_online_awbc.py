from __future__ import annotations

import pytest
import torch

from rlinf.algorithms.online_awbc import (
    FixedThresholdChunkController,
    FixedVFDThreshold,
    HysteresisChunkController,
    uniformly_spaced_chunk_indices,
)
from rlinf.data.maniskill_peg_progress import PEG_PROGRESS_LEVELS, peg_privileged_phi
from rlinf.data.online_awbc import (
    OnlineAWBCChunk,
    OnlineAWBCFrame,
    build_online_awbc_manifest,
)


def test_fixed_threshold_uses_requested_global_quantile():
    threshold = FixedVFDThreshold.calibrate([0.0, 1.0, 2.0, 3.0], quantile=0.75)
    assert threshold.threshold == pytest.approx(2.25)
    assert threshold.calibration_count == 4


def test_chunk_controller_recovers_policy_without_sticky_takeover():
    controller = FixedThresholdChunkController(
        FixedVFDThreshold(threshold=1.0, quantile=0.95, calibration_count=100)
    )
    first = controller.decide(torch.tensor([1.2, 0.9]))
    second = controller.decide(torch.tensor([0.8, 1.1]))
    assert first.controllers == ("expert", "policy")
    assert second.controllers == ("policy", "expert")


def test_hysteresis_controller_keeps_expert_until_two_clear_chunks():
    controller = HysteresisChunkController(
        FixedVFDThreshold(threshold=10.0, quantile=0.95, calibration_count=10),
        return_ratio=0.9,
        policy_release_streak=2,
    )

    assert controller.decide([10.1]).controllers == ("expert",)
    assert controller.decide([8.8]).controllers == ("expert",)
    assert controller.decide([8.7]).controllers == ("policy",)
    assert controller.decide([9.5]).controllers == ("policy",)


def test_hysteresis_controller_resets_release_streak_on_midband_score():
    controller = HysteresisChunkController(
        FixedVFDThreshold(threshold=10.0, quantile=0.95, calibration_count=10),
        return_ratio=0.9,
        policy_release_streak=2,
    )

    controller.decide([10.1])
    assert controller.decide([8.0]).controllers == ("expert",)
    assert controller.decide([9.5]).controllers == ("expert",)
    assert controller.decide([8.0]).controllers == ("expert",)
    assert controller.decide([8.0]).controllers == ("policy",)


def test_uniform_calibration_sampling_covers_episode_extremes():
    assert uniformly_spaced_chunk_indices(10, samples_per_episode=5) == (0, 2, 4, 6, 9)
    assert uniformly_spaced_chunk_indices(3, samples_per_episode=5) == (0, 1, 2)


def test_peg_phi_prefers_monotonic_once_milestones():
    assert peg_privileged_phi({}) == PEG_PROGRESS_LEVELS["ungrasped"]
    assert peg_privileged_phi({"consecutive_grasp_once": True}) == PEG_PROGRESS_LEVELS["grasped"]
    assert peg_privileged_phi({"prealign_once": True}) == PEG_PROGRESS_LEVELS["prealigned"]
    assert peg_privileged_phi({"partial_insert_once": True}) == PEG_PROGRESS_LEVELS["partially_inserted"]
    assert peg_privileged_phi({"success": True}) == PEG_PROGRESS_LEVELS["success"]


def test_online_manifest_keeps_only_chunk_anchors_valid_and_marks_terminal_success():
    frames = [
        OnlineAWBCFrame(0, 0, 0, "policy", 0.0),
        OnlineAWBCFrame(1, 0, 1, "policy", 0.0),
        OnlineAWBCFrame(2, 0, 2, "expert", 0.25),
        OnlineAWBCFrame(3, 0, 3, "expert", 0.25),
    ]
    chunks = [
        OnlineAWBCChunk(0, 0, 0, 2, "policy", 0.0, 0.25, 0.1, 0.2, False),
        OnlineAWBCChunk(2, 0, 2, 4, "expert", 0.25, 1.0, 0.4, 0.2, True),
    ]
    rows = build_online_awbc_manifest(frames, chunks)
    assert [row["valid"] for row in rows] == [True, False, True, False]
    assert rows[0]["episode_length_chunks"] == 2
    assert rows[2]["success"] is True
    assert rows[2]["next_frame_index"] == 4
