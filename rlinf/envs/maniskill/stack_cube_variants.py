"""Controlled ManiSkill StackCube distribution for fast ID learning."""

from __future__ import annotations

from typing import Any

import numpy as np


STACK_CUBE_ID_ENV_ID = "RLinfStackCubeID-v1"
STACK_CUBE_OOD_ENV_ID = "RLinfStackCubeOOD-v1"
STACK_CUBE_TASK = "stack the red cube on the green cube"
STACK_CUBE_ID_ANGLE_CENTER = np.pi / 2
STACK_CUBE_OOD_ANGLE_CENTER = -np.pi / 2
STACK_CUBE_ID_ANGLE_HALF_WIDTH = np.deg2rad(10.0)
STACK_CUBE_ID_DISTANCE_RANGE = (0.08, 0.10)
STACK_CUBE_ID_BASE_JITTER = 0.02

_REGISTERED = False


def sample_stack_cube_xy(
    rng: Any, count: int, *, split: str = "id"
) -> tuple[np.ndarray, np.ndarray]:
    """Sample green-base and red-object XY positions for a controlled split."""

    if count <= 0:
        raise ValueError("count must be positive")
    if split not in {"id", "ood"}:
        raise ValueError("split must be 'id' or 'ood'")
    angle_center = (
        STACK_CUBE_ID_ANGLE_CENTER
        if split == "id"
        else STACK_CUBE_OOD_ANGLE_CENTER
    )
    base_xy = rng.uniform(
        -STACK_CUBE_ID_BASE_JITTER,
        STACK_CUBE_ID_BASE_JITTER,
        size=(count, 2),
    )
    distance = rng.uniform(*STACK_CUBE_ID_DISTANCE_RANGE, size=count)
    angle = rng.uniform(
        angle_center - STACK_CUBE_ID_ANGLE_HALF_WIDTH,
        angle_center + STACK_CUBE_ID_ANGLE_HALF_WIDTH,
        size=count,
    )
    offset = np.stack([distance * np.cos(angle), distance * np.sin(angle)], axis=-1)
    return base_xy, base_xy + offset


def sample_stack_cube_id_xy(rng: Any, count: int) -> tuple[np.ndarray, np.ndarray]:
    return sample_stack_cube_xy(rng, count, split="id")


def stack_cube_geometry(
    cube_a_xy: Any, cube_b_xy: Any, *, split: str = "id"
) -> dict[str, np.ndarray]:
    """Return distance and signed angular offset from the split center."""

    if split not in {"id", "ood"}:
        raise ValueError("split must be 'id' or 'ood'")
    cube_a = np.asarray(cube_a_xy, dtype=np.float64)
    cube_b = np.asarray(cube_b_xy, dtype=np.float64)
    offset = cube_a - cube_b
    angle = np.arctan2(offset[..., 1], offset[..., 0])
    angle_center = (
        STACK_CUBE_ID_ANGLE_CENTER
        if split == "id"
        else STACK_CUBE_OOD_ANGLE_CENTER
    )
    angle_offset = (angle - angle_center + np.pi) % (2 * np.pi) - np.pi
    return {
        "distance": np.linalg.norm(offset, axis=-1),
        "angle": angle,
        "angle_offset": angle_offset,
    }


def stack_cube_id_geometry(cube_a_xy: Any, cube_b_xy: Any) -> dict[str, np.ndarray]:
    return stack_cube_geometry(cube_a_xy, cube_b_xy, split="id")


def geometry_is_stack_cube_id(cube_a_xy: Any, cube_b_xy: Any) -> np.ndarray:
    geometry = stack_cube_geometry(cube_a_xy, cube_b_xy, split="id")
    lower, upper = STACK_CUBE_ID_DISTANCE_RANGE
    return (
        (geometry["distance"] >= lower)
        & (geometry["distance"] <= upper)
        & (np.abs(geometry["angle_offset"]) <= STACK_CUBE_ID_ANGLE_HALF_WIDTH)
    )


def geometry_is_stack_cube_ood(cube_a_xy: Any, cube_b_xy: Any) -> np.ndarray:
    geometry = stack_cube_geometry(cube_a_xy, cube_b_xy, split="ood")
    lower, upper = STACK_CUBE_ID_DISTANCE_RANGE
    return (
        (geometry["distance"] >= lower)
        & (geometry["distance"] <= upper)
        & (np.abs(geometry["angle_offset"]) <= STACK_CUBE_ID_ANGLE_HALF_WIDTH)
    )


def reset_metadata(env: Any, *, split: str = "id") -> dict[str, Any]:
    """Create JSON-safe reset provenance for a controlled StackCube episode."""

    base = env.unwrapped

    def _array(value: Any) -> list[float]:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float64).reshape(-1).tolist()

    cube_a_pose = base.cubeA.pose
    cube_b_pose = base.cubeB.pose
    cube_a_xy = np.asarray(_array(cube_a_pose.p)[:2])
    cube_b_xy = np.asarray(_array(cube_b_pose.p)[:2])
    geometry = stack_cube_geometry(cube_a_xy, cube_b_xy, split=split)
    return {
        "split": split,
        "cube_a_pose": {"p": _array(cube_a_pose.p), "q": _array(cube_a_pose.q)},
        "cube_b_pose": {"p": _array(cube_b_pose.p), "q": _array(cube_b_pose.q)},
        "relative_distance": float(geometry["distance"]),
        "relative_angle": float(geometry["angle"]),
        "relative_angle_offset": float(geometry["angle_offset"]),
    }


def register_controlled_stack_cube_variants() -> None:
    """Register the controlled ID and opposite-direction OOD tasks."""

    global _REGISTERED
    if _REGISTERED:
        return

    import gymnasium as gym

    if (
        STACK_CUBE_ID_ENV_ID in gym.registry
        and STACK_CUBE_OOD_ENV_ID in gym.registry
    ):
        _REGISTERED = True
        return

    import torch
    from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.structs.pose import Pose

    def _initialize(controlled_env, env_idx, options: dict, *, split: str) -> None:
        StackCubeEnv._initialize_episode(controlled_env, env_idx, options)
        count = len(env_idx)
        cube_b_xy, cube_a_xy = sample_stack_cube_xy(
            controlled_env._batched_episode_rng, count, split=split
        )
        cube_a_p = controlled_env.cubeA.pose.p.clone()
        cube_b_p = controlled_env.cubeB.pose.p.clone()
        cube_a_p[env_idx, :2] = torch.as_tensor(
            cube_a_xy, dtype=cube_a_p.dtype, device=controlled_env.device
        )
        cube_b_p[env_idx, :2] = torch.as_tensor(
            cube_b_xy, dtype=cube_b_p.dtype, device=controlled_env.device
        )
        controlled_env.cubeA.set_pose(
            Pose.create_from_pq(cube_a_p, controlled_env.cubeA.pose.q.clone())
        )
        controlled_env.cubeB.set_pose(
            Pose.create_from_pq(cube_b_p, controlled_env.cubeB.pose.q.clone())
        )

    @register_env(STACK_CUBE_ID_ENV_ID, max_episode_steps=100)
    class ControlledStackCubeIDEnv(StackCubeEnv):
        def _initialize_episode(self, env_idx, options: dict) -> None:
            _initialize(self, env_idx, options, split="id")

    @register_env(STACK_CUBE_OOD_ENV_ID, max_episode_steps=100)
    class ControlledStackCubeOODEnv(StackCubeEnv):
        def _initialize_episode(self, env_idx, options: dict) -> None:
            _initialize(self, env_idx, options, split="ood")

    _REGISTERED = True


def register_controlled_stack_cube_variant() -> None:
    """Backward-compatible alias for callers that only need the ID split."""

    register_controlled_stack_cube_variants()
