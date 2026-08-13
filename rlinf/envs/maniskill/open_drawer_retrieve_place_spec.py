"""Controlled distributions for OpenDrawerRetrievePlace.

Each OOD split changes exactly one task stage. All other sampled factors use
the same random draws, so matching seeds remain paired across splits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


TaskSplit = Literal["id", "handle_ood", "grasp_ood", "goal_ood"]

ENV_IDS: dict[TaskSplit, str] = {
    "id": "RLinfOpenDrawerRetrievePlaceID-v1",
    "handle_ood": "RLinfOpenDrawerRetrievePlaceHandleOOD-v1",
    "grasp_ood": "RLinfOpenDrawerRetrievePlaceGraspOOD-v1",
    "goal_ood": "RLinfOpenDrawerRetrievePlaceGoalOOD-v1",
}

TASK_INSTRUCTION = "open the drawer, retrieve the blue object, and place it in the green tray"

DRAWER_ORIGIN = np.array([0.28, 0.0, 0.0], dtype=np.float64)
DRAWER_TRAVEL = 0.30
DRAWER_OPEN_THRESHOLD = 0.26
DRAWER_OBJECT_CENTER_LOCAL = np.array([0.025, 0.0], dtype=np.float64)
DRAWER_OBJECT_XY_JITTER = np.array([0.012, 0.035], dtype=np.float64)
DRAWER_START_QPOS_RANGE = (-0.004, 0.0)

HANDLE_OFFSET_BY_SPLIT: dict[TaskSplit, float] = {
    "id": 0.0,
    "handle_ood": 0.085,
    "grasp_ood": 0.0,
    "goal_ood": 0.0,
}

ID_OBJECT_YAW_RANGE = np.deg2rad(np.array([-10.0, 10.0], dtype=np.float64))
GRASP_OOD_YAW_RANGE = np.deg2rad(np.array([80.0, 100.0], dtype=np.float64))

ID_GOAL_CENTER = np.array([0.03, -0.30], dtype=np.float64)
GOAL_OOD_CENTER = np.array([0.03, 0.30], dtype=np.float64)
GOAL_XY_JITTER = np.array([0.018, 0.018], dtype=np.float64)


@dataclass(frozen=True)
class EpisodeParameters:
    split: TaskSplit
    handle_lateral_offset: np.ndarray
    drawer_qpos: np.ndarray
    object_local_xy: np.ndarray
    object_yaw: np.ndarray
    goal_xy: np.ndarray


def _uniform(rng: Any, low: Any, high: Any, size: tuple[int, ...]) -> np.ndarray:
    return np.asarray(rng.uniform(low, high, size=size), dtype=np.float64)


def sample_episode_parameters(
    rng: Any,
    count: int,
    *,
    split: TaskSplit,
) -> EpisodeParameters:
    """Sample paired task parameters for one controlled split."""

    if split not in ENV_IDS:
        raise ValueError(f"unknown split: {split}")
    if count < 1:
        raise ValueError("count must be positive")

    drawer_qpos = _uniform(rng, *DRAWER_START_QPOS_RANGE, size=(count,))
    object_jitter = _uniform(
        rng,
        -DRAWER_OBJECT_XY_JITTER,
        DRAWER_OBJECT_XY_JITTER,
        size=(count, 2),
    )
    yaw_unit = _uniform(rng, 0.0, 1.0, size=(count,))
    goal_jitter = _uniform(rng, -GOAL_XY_JITTER, GOAL_XY_JITTER, size=(count, 2))

    yaw_range = GRASP_OOD_YAW_RANGE if split == "grasp_ood" else ID_OBJECT_YAW_RANGE
    goal_center = GOAL_OOD_CENTER if split == "goal_ood" else ID_GOAL_CENTER
    object_yaw = yaw_range[0] + yaw_unit * (yaw_range[1] - yaw_range[0])

    return EpisodeParameters(
        split=split,
        handle_lateral_offset=np.full(count, HANDLE_OFFSET_BY_SPLIT[split]),
        drawer_qpos=drawer_qpos,
        object_local_xy=DRAWER_OBJECT_CENTER_LOCAL + object_jitter,
        object_yaw=object_yaw,
        goal_xy=goal_center + goal_jitter,
    )


def paired_common_factors(parameters: EpisodeParameters) -> dict[str, np.ndarray]:
    """Return factors that must be identical for paired ID/OOD seeds."""

    goal_center = GOAL_OOD_CENTER if parameters.split == "goal_ood" else ID_GOAL_CENTER
    yaw_range = GRASP_OOD_YAW_RANGE if parameters.split == "grasp_ood" else ID_OBJECT_YAW_RANGE
    yaw_unit = (parameters.object_yaw - yaw_range[0]) / (yaw_range[1] - yaw_range[0])
    return {
        "drawer_qpos": parameters.drawer_qpos,
        "object_local_xy": parameters.object_local_xy,
        "object_yaw_unit": yaw_unit,
        "goal_xy_jitter": parameters.goal_xy - goal_center,
    }


def reset_metadata(env: Any, *, split: TaskSplit) -> dict[str, Any]:
    """Return JSON-safe reset provenance without exposing it to the policy."""

    base = env.unwrapped

    def vector(value: Any) -> list[float]:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float64).reshape(-1).tolist()

    return {
        "split": split,
        "instruction": TASK_INSTRUCTION,
        "handle_lateral_offset": float(base.handle_lateral_offset),
        "drawer_qpos": vector(base.drawer.get_qpos()),
        "object_pose": {
            "p": vector(base.obj.pose.p),
            "q": vector(base.obj.pose.q),
        },
        "target_pose": {
            "p": vector(base.target_tray.pose.p),
            "q": vector(base.target_tray.pose.q),
        },
    }
