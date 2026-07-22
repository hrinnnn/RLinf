"""Controlled ManiSkill StackCube distribution for fast ID learning."""

from __future__ import annotations

from typing import Any

import numpy as np


STACK_CUBE_ID_ENV_ID = "RLinfStackCubeID-v1"
STACK_CUBE_TASK = "stack the red cube on the green cube"
STACK_CUBE_ID_ANGLE_CENTER = np.pi / 2
STACK_CUBE_ID_ANGLE_HALF_WIDTH = np.deg2rad(10.0)
STACK_CUBE_ID_DISTANCE_RANGE = (0.08, 0.10)
STACK_CUBE_ID_BASE_JITTER = 0.02

_REGISTERED = False


def sample_stack_cube_id_xy(rng: Any, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample green-base and red-object XY positions for the narrow ID split."""

    if count <= 0:
        raise ValueError("count must be positive")
    base_xy = rng.uniform(
        -STACK_CUBE_ID_BASE_JITTER,
        STACK_CUBE_ID_BASE_JITTER,
        size=(count, 2),
    )
    distance = rng.uniform(*STACK_CUBE_ID_DISTANCE_RANGE, size=count)
    angle = rng.uniform(
        STACK_CUBE_ID_ANGLE_CENTER - STACK_CUBE_ID_ANGLE_HALF_WIDTH,
        STACK_CUBE_ID_ANGLE_CENTER + STACK_CUBE_ID_ANGLE_HALF_WIDTH,
        size=count,
    )
    offset = np.stack([distance * np.cos(angle), distance * np.sin(angle)], axis=-1)
    return base_xy, base_xy + offset


def stack_cube_id_geometry(cube_a_xy: Any, cube_b_xy: Any) -> dict[str, np.ndarray]:
    """Return distance and signed angular offset from the ID center direction."""

    cube_a = np.asarray(cube_a_xy, dtype=np.float64)
    cube_b = np.asarray(cube_b_xy, dtype=np.float64)
    offset = cube_a - cube_b
    angle = np.arctan2(offset[..., 1], offset[..., 0])
    angle_offset = (angle - STACK_CUBE_ID_ANGLE_CENTER + np.pi) % (2 * np.pi) - np.pi
    return {
        "distance": np.linalg.norm(offset, axis=-1),
        "angle": angle,
        "angle_offset": angle_offset,
    }


def geometry_is_stack_cube_id(cube_a_xy: Any, cube_b_xy: Any) -> np.ndarray:
    geometry = stack_cube_id_geometry(cube_a_xy, cube_b_xy)
    lower, upper = STACK_CUBE_ID_DISTANCE_RANGE
    return (
        (geometry["distance"] >= lower)
        & (geometry["distance"] <= upper)
        & (np.abs(geometry["angle_offset"]) <= STACK_CUBE_ID_ANGLE_HALF_WIDTH)
    )


def reset_metadata(env: Any) -> dict[str, Any]:
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
    geometry = stack_cube_id_geometry(cube_a_xy, cube_b_xy)
    return {
        "split": "id",
        "cube_a_pose": {"p": _array(cube_a_pose.p), "q": _array(cube_a_pose.q)},
        "cube_b_pose": {"p": _array(cube_b_pose.p), "q": _array(cube_b_pose.q)},
        "relative_distance": float(geometry["distance"]),
        "relative_angle": float(geometry["angle"]),
        "relative_angle_offset": float(geometry["angle_offset"]),
    }


def register_controlled_stack_cube_variant() -> None:
    """Register the narrow ID task lazily after ManiSkill is available."""

    global _REGISTERED
    if _REGISTERED:
        return

    import gymnasium as gym

    if STACK_CUBE_ID_ENV_ID in gym.registry:
        _REGISTERED = True
        return

    import torch
    from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.structs.pose import Pose

    @register_env(STACK_CUBE_ID_ENV_ID, max_episode_steps=100)
    class ControlledStackCubeIDEnv(StackCubeEnv):
        def _initialize_episode(self, env_idx, options: dict) -> None:
            super()._initialize_episode(env_idx, options)
            count = len(env_idx)
            cube_b_xy, cube_a_xy = sample_stack_cube_id_xy(self._batched_episode_rng, count)
            cube_a_p = self.cubeA.pose.p.clone()
            cube_b_p = self.cubeB.pose.p.clone()
            cube_a_p[env_idx, :2] = torch.as_tensor(
                cube_a_xy, dtype=cube_a_p.dtype, device=self.device
            )
            cube_b_p[env_idx, :2] = torch.as_tensor(
                cube_b_xy, dtype=cube_b_p.dtype, device=self.device
            )
            self.cubeA.set_pose(Pose.create_from_pq(cube_a_p, self.cubeA.pose.q.clone()))
            self.cubeB.set_pose(Pose.create_from_pq(cube_b_p, self.cubeB.pose.q.clone()))

    _REGISTERED = True

