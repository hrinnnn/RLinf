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
ID_PROTOCOL = {
    "drawer_qpos": (-0.004, 0.0),
    "object_x": (0.293, 0.317),
    "object_y": (-0.035, 0.035),
    "object_yaw_deg": (-10.0, 10.0),
    "target_x": (0.012, 0.048),
    "target_y": (-0.318, -0.282),
}
OBJECT_HALF_SIZE = np.array([0.055, 0.024, 0.024], dtype=np.float64)
TARGET_INNER_HALF_SIZE = np.array([0.075, 0.075], dtype=np.float64)

DRAWER_ORIGIN = np.array([0.28, 0.0, 0.0], dtype=np.float64)
DRAWER_TRAVEL = 0.38
DRAWER_OPEN_THRESHOLD = 0.34
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


def _to_numpy(value: Any) -> np.ndarray:
    """Convert CPU or CUDA tensor-like values before reset-state inspection."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def reset_metadata(env: Any, *, split: TaskSplit) -> dict[str, Any]:
    """Return JSON-safe reset provenance and real reset lifecycle state."""

    base = env.unwrapped

    def vector(value: Any) -> list[float]:
        return _to_numpy(value).astype(np.float64).reshape(-1).tolist()

    def boolean(value: Any) -> bool:
        return bool(_to_numpy(value).astype(bool).reshape(-1).any())

    object_position = vector(base.obj.pose.p)
    object_quaternion = vector(base.obj.pose.q)
    target_position = vector(base.target_tray.pose.p)
    object_yaw_deg = float(np.rad2deg(2.0 * np.arctan2(object_quaternion[3], object_quaternion[0])))
    env_id = getattr(getattr(env, "spec", None), "id", None)
    if env_id is None:
        env_id = getattr(env, "env_id", None)
    control_mode = getattr(base, "control_mode", None)
    if control_mode is None:
        control_mode = getattr(getattr(env, "unwrapped", None), "control_mode", None)
    if env_id is None:
        env_id = "unknown"
    if control_mode is None:
        control_mode = "unknown"

    drawer_opened = bool(float(vector(base.drawer.get_qpos())[0]) <= -DRAWER_OPEN_THRESHOLD)
    object_grasped = boolean(base.agent.is_grasping(base.obj))
    object_lifted = bool(object_position[2] >= 0.12)
    target_delta = np.asarray(object_position[:2]) - np.asarray(target_position[:2])
    object_in_target = bool(
        abs(target_delta[0]) <= TARGET_INNER_HALF_SIZE[0]
        and abs(target_delta[1]) <= TARGET_INNER_HALF_SIZE[1]
        and OBJECT_HALF_SIZE[2] - 0.005 <= object_position[2] <= 0.085
    )
    object_released_now = not object_grasped
    ever_drawer_opened = boolean(getattr(base, "_ever_drawer_opened", False))
    ever_grasped = boolean(getattr(base, "_ever_grasped", False))
    ever_lifted = boolean(getattr(base, "_ever_lifted", False))
    is_robot_static = boolean(base.agent.is_static(0.2))
    success = bool(ever_drawer_opened and object_in_target and object_released_now and is_robot_static)
    return {
        "env_id": str(env_id),
        "split": split,
        "instruction": TASK_INSTRUCTION,
        "control_mode": str(control_mode),
        "handle_lateral_offset": float(base.handle_lateral_offset),
        "drawer_qpos": vector(base.drawer.get_qpos()),
        "object_pose": {
            "p": object_position,
            "q": object_quaternion,
            "yaw_deg": object_yaw_deg,
        },
        "target_pose": {
            "p": target_position,
            "q": vector(base.target_tray.pose.q),
        },
        "lifecycle": {
            "drawer_opened": drawer_opened,
            "object_grasped": object_grasped,
            "object_lifted": object_lifted,
            "object_in_target": object_in_target,
            "object_released_now": object_released_now,
            "release_after_grasp_event": False,
            "ever_drawer_opened": ever_drawer_opened,
            "ever_grasped": ever_grasped,
            "ever_lifted": ever_lifted,
            "is_robot_static": is_robot_static,
            "success": success,
        },
    }


def validate_reset_metadata(metadata: dict[str, Any], *, split: TaskSplit) -> list[str]:
    """Validate reset provenance before accepting an episode for a formal gate."""

    errors: list[str] = []

    def scalar(path: str, value: Any) -> float | None:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if array.size != 1 or not np.isfinite(array[0]):
            errors.append(f"{path}: expected one finite scalar")
            return None
        return float(array[0])

    def range_check(path: str, value: Any, low: float, high: float) -> None:
        number = scalar(path, value)
        if number is not None and not (low - 1e-6 <= number <= high + 1e-6):
            errors.append(f"{path}: {number} outside [{low}, {high}]")

    expected_env_id = ENV_IDS[split]
    if metadata.get("env_id") != expected_env_id:
        errors.append(f"env_id: expected {expected_env_id!r}, got {metadata.get('env_id')!r}")
    if metadata.get("split") != split:
        errors.append(f"split: expected {split!r}, got {metadata.get('split')!r}")
    if metadata.get("instruction") != TASK_INSTRUCTION:
        errors.append("instruction: does not equal TASK_INSTRUCTION")
    if metadata.get("control_mode") != "pd_joint_delta_pos":
        errors.append("control_mode: expected pd_joint_delta_pos")
    camera = metadata.get("camera")
    if not isinstance(camera, dict):
        errors.append("camera: actual camera metadata is missing")
    else:
        if camera.get("main") in (None, "unknown") or camera.get("wrist") in (None, "unknown"):
            errors.append("camera: actual main/wrist keys are required")
        for role in ("main", "wrist"):
            shape = camera.get(f"{role}_shape")
            if not isinstance(shape, list) or len(shape) != 3 or shape[-1] != 3:
                errors.append(f"camera.{role}_shape: expected [height,width,3]")
            elif camera.get("requested_size") and list(shape[:2]) != list(camera["requested_size"]):
                errors.append(f"camera.{role}_shape: does not match requested size")

    expected_handle = HANDLE_OFFSET_BY_SPLIT[split]
    range_check("handle_lateral_offset", metadata.get("handle_lateral_offset"), expected_handle - 1e-6, expected_handle + 1e-6)
    drawer_qpos = np.asarray(metadata.get("drawer_qpos", []), dtype=np.float64).reshape(-1)
    if drawer_qpos.size != 1:
        errors.append("drawer_qpos: expected one value")
    else:
        range_check("drawer_qpos", drawer_qpos[0], *DRAWER_START_QPOS_RANGE)

    object_pose = metadata.get("object_pose", {})
    object_position = np.asarray(object_pose.get("p", []), dtype=np.float64).reshape(-1)
    if object_position.size != 3:
        errors.append("object_pose.p: expected 3 values")
    else:
        range_check("object_pose.p[0]", object_position[0], *ID_PROTOCOL["object_x"])
        range_check("object_pose.p[1]", object_position[1], *ID_PROTOCOL["object_y"])
    yaw_range = GRASP_OOD_YAW_RANGE if split == "grasp_ood" else ID_OBJECT_YAW_RANGE
    range_check("object_pose.yaw_deg", object_pose.get("yaw_deg"), *np.rad2deg(yaw_range))

    target_position = np.asarray(metadata.get("target_pose", {}).get("p", []), dtype=np.float64).reshape(-1)
    if target_position.size != 3:
        errors.append("target_pose.p: expected 3 values")
    else:
        target_center = GOAL_OOD_CENTER if split == "goal_ood" else ID_GOAL_CENTER
        range_check("target_pose.p[0]", target_position[0], *(target_center[0] + np.array([-GOAL_XY_JITTER[0], GOAL_XY_JITTER[0]])))
        range_check("target_pose.p[1]", target_position[1], *(target_center[1] + np.array([-GOAL_XY_JITTER[1], GOAL_XY_JITTER[1]])))

    lifecycle = metadata.get("lifecycle")
    if not isinstance(lifecycle, dict):
        errors.append("lifecycle: reset state is missing")
    else:
        for key in ("drawer_opened", "object_grasped", "object_lifted", "object_in_target", "ever_drawer_opened", "ever_grasped", "ever_lifted", "success"):
            if lifecycle.get(key) is not False:
                errors.append(f"lifecycle.{key}: expected false at reset")
        if lifecycle.get("object_released_now") is not True:
            errors.append("lifecycle.object_released_now: expected true at reset")
        if lifecycle.get("release_after_grasp_event") is not False:
            errors.append("lifecycle.release_after_grasp_event: expected false at reset")
    return errors
