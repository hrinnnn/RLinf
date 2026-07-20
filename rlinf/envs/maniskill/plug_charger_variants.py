"""Controlled ManiSkill PlugCharger distributions for ID/OOD experiments.

The official task samples a charger yaw near the insertion orientation.  This
module leaves its robot, object XY, camera, physics, and success definition
unchanged, then replaces only the charger's initial yaw relative to the
officially constructed goal pose.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np


PLUG_CHARGER_ID_ENV_ID = "RLinfPlugChargerID-v1"
PLUG_CHARGER_OOD_ENV_ID = "RLinfPlugChargerOOD-v1"
PLUG_CHARGER_ID_YAW_RANGE = (-np.pi / 12, np.pi / 12)
PLUG_CHARGER_OOD_YAW_RANGE = (11 * np.pi / 12, 13 * np.pi / 12)
PLUG_CHARGER_TASK = "plug the charger into the receptacle"

_REGISTERED = False


def wrap_yaw(angle: np.ndarray | float) -> np.ndarray | float:
    """Normalize yaw to ``[-pi, pi)`` without changing its batch shape."""

    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def yaw_from_quaternion_wxyz(quaternion: Any) -> np.ndarray:
    """Extract Z yaw from ManiSkill/SAPIEN ``wxyz`` quaternions."""

    value = quaternion.detach().cpu().numpy() if hasattr(quaternion, "detach") else quaternion
    q = np.asarray(value, dtype=np.float64)
    w, x, y, z = [q[..., index] for index in range(4)]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def relative_yaw_from_poses(charger_pose: Any, goal_pose: Any) -> np.ndarray:
    """Return charger yaw relative to the task's required insertion pose."""

    return wrap_yaw(yaw_from_quaternion_wxyz(charger_pose.q) - yaw_from_quaternion_wxyz(goal_pose.q))


def split_for_env_id(env_id: str) -> Literal["id", "ood"]:
    if env_id == PLUG_CHARGER_ID_ENV_ID:
        return "id"
    if env_id == PLUG_CHARGER_OOD_ENV_ID:
        return "ood"
    raise ValueError(f"Not a controlled PlugCharger env id: {env_id}")


def is_controlled_plug_charger_env_id(env_id: str) -> bool:
    return env_id in {PLUG_CHARGER_ID_ENV_ID, PLUG_CHARGER_OOD_ENV_ID}


def default_plug_instruction(*, num_envs: int) -> list[str]:
    return [PLUG_CHARGER_TASK for _ in range(num_envs)]


def wrap_plug_charger_openpi_joint_obs(
    raw_obs: dict[str, Any],
    *,
    task_descriptions: list[str] | None = None,
) -> dict[str, Any]:
    """Map official PlugCharger RGB observations to the pi0.5 contract."""

    import torch

    sensor_data = raw_obs.pop("sensor_data")
    raw_obs.pop("sensor_param", None)
    main_images = sensor_data["base_camera"]["rgb"]
    wrist_images = sensor_data["hand_camera"]["rgb"]
    batch_size = int(main_images.shape[0])
    return {
        "main_images": main_images,
        "wrist_images": wrist_images,
        "extra_view_images": None,
        "states": raw_obs["agent"]["qpos"],
        "task_descriptions": task_descriptions or default_plug_instruction(num_envs=batch_size),
        "task_ids": torch.zeros(batch_size, dtype=torch.long, device=main_images.device),
    }


def reset_metadata(env: Any) -> dict[str, Any]:
    """Create JSON-safe reset provenance after ``env.reset``.

    This is intentionally separate from ``evaluate`` so ManiSkill's tensor
    infos retain their original contract.
    """

    base_env = env.unwrapped
    charger_pose = base_env.charger.pose
    receptacle_pose = base_env.receptacle.pose
    goal_pose = base_env.goal_pose
    relative_yaw = relative_yaw_from_poses(charger_pose, goal_pose)

    def _array(value: Any) -> list[float]:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float64).reshape(-1).tolist()

    return {
        "split": str(base_env.rlinf_split),
        "charger_pose": {"p": _array(charger_pose.p), "q": _array(charger_pose.q)},
        "receptacle_pose": {"p": _array(receptacle_pose.p), "q": _array(receptacle_pose.q)},
        "goal_pose": {"p": _array(goal_pose.p), "q": _array(goal_pose.q)},
        "relative_yaw": float(np.asarray(relative_yaw).reshape(-1)[0]),
    }


def register_controlled_plug_charger_variants() -> None:
    """Register ID/OOD classes lazily, after ManiSkill is importable."""

    global _REGISTERED
    if _REGISTERED:
        return

    import gymnasium as gym

    if PLUG_CHARGER_ID_ENV_ID in gym.registry and PLUG_CHARGER_OOD_ENV_ID in gym.registry:
        _REGISTERED = True
        return

    import torch
    from mani_skill.envs.tasks.tabletop.plug_charger import PlugChargerEnv
    from mani_skill.utils.geometry import rotation_conversions
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.structs.pose import Pose

    class _ControlledPlugChargerMixin:
        rlinf_split: Literal["id", "ood"]
        rlinf_relative_yaw_range: tuple[float, float]

        def _initialize_episode(self, env_idx, options: dict) -> None:
            super()._initialize_episode(env_idx, options)
            count = len(env_idx)
            # Use ManiSkill's episode RNG, not NumPy global state, so reset
            # seeds remain reproducible and all non-yaw official samples stay
            # untouched.
            yaw = self._batched_episode_rng.uniform(
                self.rlinf_relative_yaw_range[0],
                self.rlinf_relative_yaw_range[1],
                size=count,
            )
            yaw = torch.as_tensor(yaw, dtype=self.goal_pose.q.dtype, device=self.device)
            relative_q = torch.zeros((count, 4), dtype=self.goal_pose.q.dtype, device=self.device)
            relative_q[:, 0] = torch.cos(yaw / 2)
            relative_q[:, 3] = torch.sin(yaw / 2)
            orientation = rotation_conversions.quaternion_multiply(self.goal_pose.q[env_idx], relative_q)
            # Actor.set_pose updates the complete merged Actor in ManiSkill
            # 3.0.  Preserve untouched vectorized environments during a
            # partial reset, then write the updated full pose once.
            position = self.charger.pose.p.clone()
            full_orientation = self.charger.pose.q.clone()
            full_orientation[env_idx] = orientation
            self.charger.set_pose(Pose.create_from_pq(position, full_orientation))
            self.rlinf_relative_yaw = yaw

    @register_env(PLUG_CHARGER_ID_ENV_ID, max_episode_steps=200)
    class ControlledPlugChargerIDEnv(_ControlledPlugChargerMixin, PlugChargerEnv):
        rlinf_split = "id"
        rlinf_relative_yaw_range = PLUG_CHARGER_ID_YAW_RANGE

    @register_env(PLUG_CHARGER_OOD_ENV_ID, max_episode_steps=200)
    class ControlledPlugChargerOODEnv(_ControlledPlugChargerMixin, PlugChargerEnv):
        rlinf_split = "ood"
        rlinf_relative_yaw_range = PLUG_CHARGER_OOD_YAW_RANGE

    _REGISTERED = True
