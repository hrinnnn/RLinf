"""Controlled PickSingleYCB airplane distributions for ID/OOD experiments.

The stock ManiSkill task changes both object identity and pose.  This module
keeps the Panda, cameras, task success condition, object/goal translation
distribution and robot noise fixed, while changing only the toy airplane's
initial yaw between the two registered splits.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np


PICK_SINGLE_YCB_AIRPLANE_ID_ENV_ID = "RLinfPickSingleYCBAirplaneID-v1"
PICK_SINGLE_YCB_AIRPLANE_OOD_ENV_ID = "RLinfPickSingleYCBAirplaneOOD-v1"
PICK_SINGLE_YCB_AIRPLANE_MODEL_ID = "072-a_toy_airplane"
PICK_SINGLE_YCB_AIRPLANE_TASK = "pick up the toy airplane and move it to the green goal"

# These intervals intentionally leave a 50 degree yaw gap on each side.
PICK_SINGLE_YCB_AIRPLANE_ID_YAW_RANGE = (np.deg2rad(-20.0), np.deg2rad(20.0))
PICK_SINGLE_YCB_AIRPLANE_OOD_YAW_RANGES = (
    (np.deg2rad(70.0), np.deg2rad(110.0)),
    (np.deg2rad(-110.0), np.deg2rad(-70.0)),
)

# Both splits draw from these exact distributions.  The x-separated centres
# guarantee a useful, reachable pick-to-goal displacement without making yaw
# a proxy for translation.
OBJECT_XY_CENTER = (-0.05, 0.0)
GOAL_XY_CENTER = (0.05, 0.0)
XY_JITTER = 0.02
GOAL_Z_OFFSET_RANGE = (0.12, 0.16)

_REGISTERED = False


def yaw_from_quaternion_wxyz(quaternion: Any) -> np.ndarray:
    """Extract Z yaw from a ManiSkill/SAPIEN wxyz quaternion."""

    value = quaternion.detach().cpu().numpy() if hasattr(quaternion, "detach") else quaternion
    q = np.asarray(value, dtype=np.float64)
    w, x, y, z = [q[..., index] for index in range(4)]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def yaw_in_ranges(yaw: float | np.ndarray, ranges: tuple[tuple[float, float], ...]) -> np.ndarray:
    """Return whether yaw belongs to one of inclusive non-wrapping ranges."""

    values = np.asarray(yaw)
    return np.logical_or.reduce([(values >= lower) & (values <= upper) for lower, upper in ranges])


def split_for_env_id(env_id: str) -> Literal["id", "ood"]:
    if env_id == PICK_SINGLE_YCB_AIRPLANE_ID_ENV_ID:
        return "id"
    if env_id == PICK_SINGLE_YCB_AIRPLANE_OOD_ENV_ID:
        return "ood"
    raise ValueError(f"Not a controlled airplane env id: {env_id}")


def sample_airplane_yaw(rng: Any, count: int, *, split: Literal["id", "ood"]) -> np.ndarray:
    """Sample a yaw while keeping every non-yaw factor split-invariant."""

    if count < 1:
        raise ValueError("count must be positive")
    if split == "id":
        return np.asarray(rng.uniform(*PICK_SINGLE_YCB_AIRPLANE_ID_YAW_RANGE, size=count), dtype=np.float64)
    if split != "ood":
        raise ValueError(f"unknown split: {split}")
    interval_index = np.asarray(rng.integers(0, len(PICK_SINGLE_YCB_AIRPLANE_OOD_YAW_RANGES), size=count))
    result = np.empty(count, dtype=np.float64)
    for index, (lower, upper) in enumerate(PICK_SINGLE_YCB_AIRPLANE_OOD_YAW_RANGES):
        mask = interval_index == index
        result[mask] = rng.uniform(lower, upper, size=int(mask.sum()))
    return result


def reset_metadata(env: Any, *, split: Literal["id", "ood"]) -> dict[str, Any]:
    """Return JSON-safe immutable reset provenance after ``env.reset``."""

    base = env.unwrapped

    def _array(value: Any) -> list[float]:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float64).reshape(-1).tolist()

    object_pose = base.obj.pose
    goal_pose = base.goal_site.pose
    return {
        "split": split,
        "object_model_id": PICK_SINGLE_YCB_AIRPLANE_MODEL_ID,
        "object_pose": {"p": _array(object_pose.p), "q": _array(object_pose.q)},
        "goal_pose": {"p": _array(goal_pose.p), "q": _array(goal_pose.q)},
        "object_yaw": float(yaw_from_quaternion_wxyz(object_pose.q).reshape(-1)[0]),
        "object_xy_center": list(OBJECT_XY_CENTER),
        "goal_xy_center": list(GOAL_XY_CENTER),
        "xy_jitter": float(XY_JITTER),
        "goal_z_offset_range": list(GOAL_Z_OFFSET_RANGE),
    }


def register_controlled_pick_single_ycb_airplane_variants() -> None:
    """Register fixed-airplane ID/OOD variants lazily after ManiSkill import."""

    global _REGISTERED
    if _REGISTERED:
        return
    import gymnasium as gym

    if PICK_SINGLE_YCB_AIRPLANE_ID_ENV_ID in gym.registry and PICK_SINGLE_YCB_AIRPLANE_OOD_ENV_ID in gym.registry:
        _REGISTERED = True
        return

    import sapien
    import torch
    from mani_skill.envs.tasks.tabletop.pick_single_ycb import PickSingleYCBEnv
    from mani_skill.utils.building import actors
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.scene_builder.table import TableSceneBuilder
    from mani_skill.utils.structs.actor import Actor
    from mani_skill.utils.structs.pose import Pose

    class _ControlledAirplaneMixin:
        rlinf_split: Literal["id", "ood"]

        def _load_scene(self, options: dict) -> None:
            # This is PickSingleYCBEnv._load_scene with just one deliberate
            # change: its random model list becomes the immutable airplane.
            self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
            self.table_scene.build()
            self._objs = []
            for index in range(self.num_envs):
                builder = actors.get_actor_builder(self.scene, id=f"ycb:{PICK_SINGLE_YCB_AIRPLANE_MODEL_ID}")
                builder.initial_pose = sapien.Pose(p=[0, 0, 0])
                builder.set_scene_idxs([index])
                actor = builder.build(name=f"{PICK_SINGLE_YCB_AIRPLANE_MODEL_ID}-{index}")
                self._objs.append(actor)
                self.remove_from_state_dict_registry(actor)
            self.obj = Actor.merge(self._objs, name="ycb_object")
            self.add_to_state_dict_registry(self.obj)
            self.goal_site = actors.build_sphere(
                self.scene, radius=self.goal_thresh, color=[0, 1, 0, 1], name="goal_site",
                body_type="kinematic", add_collision=False, initial_pose=sapien.Pose(),
            )
            self._hidden_objects.append(self.goal_site)

        def _initialize_episode(self, env_idx, options: dict) -> None:
            with torch.device(self.device):
                count = len(env_idx)
                self.table_scene.initialize(env_idx)
                rng = self._batched_episode_rng
                object_xy = rng.uniform(
                    np.asarray(OBJECT_XY_CENTER) - XY_JITTER,
                    np.asarray(OBJECT_XY_CENTER) + XY_JITTER,
                    size=(count, 2),
                )
                goal_xy = rng.uniform(
                    np.asarray(GOAL_XY_CENTER) - XY_JITTER,
                    np.asarray(GOAL_XY_CENTER) + XY_JITTER,
                    size=(count, 2),
                )
                yaw = torch.as_tensor(sample_airplane_yaw(rng, count, split=self.rlinf_split), device=self.device, dtype=torch.float32)
                object_position = self.obj.pose.p.clone()
                object_orientation = self.obj.pose.q.clone()
                object_position[env_idx, :2] = torch.as_tensor(object_xy, dtype=object_position.dtype, device=self.device)
                object_position[env_idx, 2] = self.object_zs[env_idx]
                object_orientation[env_idx] = 0
                object_orientation[env_idx, 0] = torch.cos(yaw / 2).to(object_orientation.dtype)
                object_orientation[env_idx, 3] = torch.sin(yaw / 2).to(object_orientation.dtype)
                self.obj.set_pose(Pose.create_from_pq(object_position, object_orientation))
                goal_position = self.goal_site.pose.p.clone()
                goal_position[env_idx, :2] = torch.as_tensor(goal_xy, dtype=goal_position.dtype, device=self.device)
                goal_position[env_idx, 2] = self.object_zs[env_idx] + torch.as_tensor(
                    rng.uniform(*GOAL_Z_OFFSET_RANGE, size=count), dtype=goal_position.dtype, device=self.device
                )
                self.goal_site.set_pose(Pose.create_from_pq(goal_position, self.goal_site.pose.q.clone()))
                qpos = np.array([0.0, 0, 0, -np.pi * 2 / 3, 0, np.pi * 2 / 3, np.pi / 4, 0.04, 0.04])
                qpos[:-2] += self._episode_rng.normal(0, self.robot_init_qpos_noise, len(qpos) - 2)
                self.agent.reset(qpos)
                self.agent.robot.set_root_pose(sapien.Pose([-0.615, 0, 0]))

    @register_env(PICK_SINGLE_YCB_AIRPLANE_ID_ENV_ID, max_episode_steps=50, asset_download_ids=["ycb"])
    class ControlledPickSingleYCBAirplaneIDEnv(_ControlledAirplaneMixin, PickSingleYCBEnv):
        rlinf_split = "id"

    @register_env(PICK_SINGLE_YCB_AIRPLANE_OOD_ENV_ID, max_episode_steps=50, asset_download_ids=["ycb"])
    class ControlledPickSingleYCBAirplaneOODEnv(_ControlledAirplaneMixin, PickSingleYCBEnv):
        rlinf_split = "ood"

    _REGISTERED = True
