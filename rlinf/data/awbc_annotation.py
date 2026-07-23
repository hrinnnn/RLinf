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

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass(frozen=True)
class ProgressEstimate:
    dataset_index: int
    episode_index: int
    frame_index: int
    phi: float | None
    valid: bool
    confidence: float = 1.0
    mode_phis: dict[str, float | None] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetFrame:
    dataset_index: int
    episode_index: int
    frame_index: int


def select_episode_anchors(
    frames: Sequence[DatasetFrame], stride_steps: int
) -> list[DatasetFrame]:
    if stride_steps <= 0:
        raise ValueError("stride_steps must be positive")
    if not frames:
        return []
    ordered = sorted(frames, key=lambda frame: frame.frame_index)
    anchors = [ordered[0]]
    last_anchor_frame = ordered[0].frame_index
    for frame in ordered[1:-1]:
        if frame.frame_index - last_anchor_frame >= stride_steps:
            anchors.append(frame)
            last_anchor_frame = frame.frame_index
    if ordered[-1].dataset_index != anchors[-1].dataset_index:
        anchors.append(ordered[-1])
    return anchors


def build_awbc_manifest_rows(
    frames: Sequence[DatasetFrame],
    estimates: Sequence[ProgressEstimate],
    *,
    stride_steps: int,
    source: str,
    successful_episodes: set[int] | None = None,
    lookahead_steps: int | None = None,
) -> list[dict[str, Any]]:
    if source not in {"expert", "policy"}:
        raise ValueError("source must be 'expert' or 'policy'")
    successful_episodes = successful_episodes or set()
    if lookahead_steps is not None and lookahead_steps <= 0:
        raise ValueError("lookahead_steps must be positive")
    estimate_by_index = {estimate.dataset_index: estimate for estimate in estimates}
    if len(estimate_by_index) != len(estimates):
        raise ValueError("duplicate progress estimates")

    episodes: dict[int, list[DatasetFrame]] = {}
    for frame in frames:
        episodes.setdefault(frame.episode_index, []).append(frame)

    rows: list[dict[str, Any]] = []
    for episode_index, episode_frames in sorted(episodes.items()):
        ordered = sorted(episode_frames, key=lambda frame: frame.frame_index)
        anchors = select_episode_anchors(ordered, stride_steps)
        anchor_positions = {
            anchor.dataset_index: position for position, anchor in enumerate(anchors)
        }
        episode_length_chunks = max(1, len(anchors) - 1)

        for frame in ordered:
            valid = False
            phi = None
            phi_next = None
            confidence = 0.0
            next_frame_index = frame.frame_index
            mode_phis: dict[str, float | None] = {}
            position = anchor_positions.get(frame.dataset_index)
            transition_success = False
            next_position = None
            if position is not None:
                if lookahead_steps is None:
                    next_position = (
                        position + 1 if position + 1 < len(anchors) else None
                    )
                else:
                    next_position = next(
                        (
                            candidate
                            for candidate in range(position + 1, len(anchors))
                            if anchors[candidate].frame_index - frame.frame_index
                            >= lookahead_steps
                        ),
                        None,
                    )
            if position is not None and next_position is not None:
                next_anchor = anchors[next_position]
                current_estimate = estimate_by_index.get(frame.dataset_index)
                next_estimate = estimate_by_index.get(next_anchor.dataset_index)
                valid = bool(
                    current_estimate is not None
                    and next_estimate is not None
                    and current_estimate.valid
                    and next_estimate.valid
                    and current_estimate.phi is not None
                    and next_estimate.phi is not None
                )
                next_frame_index = next_anchor.frame_index
                if valid:
                    phi = float(current_estimate.phi)
                    phi_next = float(next_estimate.phi)
                    confidence = float(next_estimate.confidence)
                    mode_phis = dict(next_estimate.mode_phis)
                    transition_success = bool(
                        episode_index in successful_episodes
                        and next_position == len(anchors) - 1
                    )

            rows.append(
                {
                    "dataset_index": frame.dataset_index,
                    "episode_index": episode_index,
                    "frame_index": frame.frame_index,
                    "next_frame_index": next_frame_index,
                    "phi": phi,
                    "phi_next": phi_next,
                    "delta_phi": None if not valid else phi_next - phi,
                    "valid": valid,
                    "confidence": confidence,
                    "episode_length_chunks": episode_length_chunks,
                    "source": source,
                    "success": transition_success,
                    "phi_incremental": mode_phis.get("incremental"),
                    "phi_forward": mode_phis.get("forward"),
                    "phi_backward": mode_phis.get("backward"),
                }
            )
    return sorted(rows, key=lambda row: row["dataset_index"])
