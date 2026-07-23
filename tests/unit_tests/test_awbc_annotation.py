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

from rlinf.data.awbc_annotation import (
    DatasetFrame,
    ProgressEstimate,
    build_awbc_manifest_rows,
    select_episode_anchors,
)


def _frames(episode, start_index, count):
    return [
        DatasetFrame(
            dataset_index=start_index + frame,
            episode_index=episode,
            frame_index=frame,
        )
        for frame in range(count)
    ]


def test_anchor_selection_includes_start_stride_and_terminal():
    anchors = select_episode_anchors(_frames(0, 0, 13), stride_steps=5)
    assert [anchor.frame_index for anchor in anchors] == [0, 5, 10, 12]


def test_manifest_marks_only_complete_anchor_pairs_valid():
    frames = _frames(0, 0, 7)
    estimates = [
        ProgressEstimate(0, 0, 0, 0.0, True),
        ProgressEstimate(5, 0, 5, 0.4, True, confidence=0.7),
        ProgressEstimate(6, 0, 6, 1.0, True, confidence=1.0),
    ]
    rows = build_awbc_manifest_rows(
        frames,
        estimates,
        stride_steps=5,
        source="expert",
        successful_episodes={0},
    )

    assert len(rows) == 7
    assert rows[0]["valid"] and rows[0]["delta_phi"] == 0.4
    assert rows[0]["next_frame_index"] == 5
    assert rows[5]["valid"] and rows[5]["delta_phi"] == 0.6
    assert rows[5]["success"] is True
    assert rows[6]["valid"] is False
    assert all(not rows[index]["valid"] for index in range(1, 5))


def test_invalid_estimate_does_not_create_negative_reward_artifact():
    frames = _frames(0, 0, 7)
    estimates = [
        ProgressEstimate(0, 0, 0, 0.0, True),
        ProgressEstimate(5, 0, 5, None, False, confidence=0.0),
        ProgressEstimate(6, 0, 6, 0.8, True),
    ]
    rows = build_awbc_manifest_rows(
        frames, estimates, stride_steps=5, source="policy"
    )

    assert rows[0]["valid"] is False
    assert rows[0]["phi_next"] is None
    assert rows[0]["delta_phi"] is None
    assert rows[5]["valid"] is False


def test_success_override_is_scoped_to_its_episode():
    frames = _frames(0, 0, 2) + _frames(1, 2, 2)
    estimates = [
        ProgressEstimate(0, 0, 0, 0.0, True),
        ProgressEstimate(1, 0, 1, 1.0, True),
        ProgressEstimate(2, 1, 0, 0.0, True),
        ProgressEstimate(3, 1, 1, 0.2, True),
    ]
    rows = build_awbc_manifest_rows(
        frames,
        estimates,
        stride_steps=5,
        source="policy",
        successful_episodes={0},
    )

    assert rows[0]["success"] is True
    assert rows[0]["phi_next"] == 1.0
    assert rows[2]["success"] is False
    assert rows[2]["phi"] == 0.0
    assert rows[2]["phi_next"] == 0.2


def test_manifest_can_use_ten_step_progress_with_five_step_annotations():
    frames = _frames(0, 0, 16)
    estimates = [
        ProgressEstimate(0, 0, 0, 0.0, True),
        ProgressEstimate(5, 0, 5, 0.1, True),
        ProgressEstimate(10, 0, 10, 0.6, True),
        ProgressEstimate(15, 0, 15, 0.9, True),
    ]
    rows = build_awbc_manifest_rows(
        frames,
        estimates,
        stride_steps=5,
        lookahead_steps=10,
        source="policy",
    )

    assert rows[0]["valid"] and rows[0]["next_frame_index"] == 10
    assert rows[0]["delta_phi"] == 0.6
    assert rows[5]["valid"] and rows[5]["next_frame_index"] == 15
    assert rows[10]["valid"] is False
