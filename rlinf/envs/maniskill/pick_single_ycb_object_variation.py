"""Controlled PickSingleYCB object-variation distributions.

The task keeps the Panda, camera, instruction, object pose distribution, goal
distribution, robot initialization, and success predicate fixed. The only
split variable is the YCB model id.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np


PICK_SINGLE_YCB_OBJECT_ID_ENV_ID = "RLinfPickSingleYCBObjectVariationID-v1"
PICK_SINGLE_YCB_OBJECT_OOD_ENV_ID = "RLinfPickSingleYCBObjectVariationOOD-v1"
PICK_SINGLE_YCB_OBJECT_ID_MODEL_ID = "005_tomato_soup_can"
PICK_SINGLE_YCB_OBJECT_OOD_MODEL_ID = "008_pudding_box"
PICK_SINGLE_YCB_OBJECT_TASK = "pick up the object and move it to the green goal"

OBJECT_XY_CENTER = (-0.05, 0.0)
GOAL_XY_CENTER = (0.05, 0.0)
XY_JITTER = 0.02
GOAL_Z_OFFSET_RANGE = (0.16, 0.20)
OBJECT_POSE_QUATERNION_WXYZ = (1.0, 0.0, 0.0, 0.0)

_REGISTERED = False


def model_id_for_split(split: Literal["id", "ood"]) -> str:
    if split == "id":
        return PICK_SINGLE_YCB_OBJECT_ID_MODEL_ID
    if split == "ood":
        return PICK_SINGLE_YCB_OBJECT_OOD_MODEL_ID
    raise ValueError(f"unknown split: {split}")


def split_for_env_id(env_id: str) -> Literal["id", "ood"]:
    if env_id == PICK_SINGLE_YCB_OBJECT_ID_ENV_ID:
        return "id"
    if env_id == PICK_SINGLE_YCB_OBJECT_OOD_ENV_ID:
        return "ood"
    raise ValueError(f"not a controlled object-variation env id: {env_id}")


def _array(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).reshape(-1).tolist()


def reset_metadata(env: Any, *, split: Literal["id", "ood"]) -> dict[str, Any]:
    """Return JSON-safe provenance after a seeded reset."""

    base = env.unwrapped
    object_pose = base.obj.pose
    goal_pose = base.goal_site.pose
    return {
        "split": split,
        "object_model_id": model_id_for_split(split),
        "object_pose": {"p": _array(object_pose.p), "q": _array(object_pose.q)},
        "goal_pose": {"p": _array(goal_pose.p), "q": _array(goal_pose.q)},
        "object_xy_center": list(OBJECT_XY_CENTER),
        "goal_xy_center": list(GOAL_XY_CENTER),
        "xy_jitter": float(XY_JITTER),
        "goal_z_offset_range": list(GOAL_Z_OFFSET_RANGE),
        "object_pose_quaternion_wxyz": list(OBJECT_POSE_QUATERNION_WXYZ),
        "object_variation_only": True,
    }


def register_controlled_pick_single_ycb_object_variants() -> None:
    """Register fixed-ID and held-out-object variants lazily."""

    global _REGISTERED
    if _REGISTERED:
        return

    import gymnasium as gym

    required_ids = (PICK_SINGLE_YCB_OBJECT_ID_ENV_ID, PICK_SINGLE_YCB_OBJECT_OOD_ENV_ID)
    if all(env_id in gym.registry for env_id in required_ids):
        _REGISTERED = True
        return

    import sapien
    import torch
    from mani_skill.envs.tasks.tabletop.pick_single_ycb import PickSingleYCBEnv
    from mani_skill.utils import common
    from mani_skill.utils.building import actors
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.scene_builder.table import TableSceneBuilder
    from mani_skill.utils.structs.actor import Actor
    from mani_skill.utils.structs.pose import Pose

    class _ObjectVariationMixin:
        rlinf_split: Literal["id", "ood"]

        def _load_scene(self, options: dict) -> None:
            self.table_scene = TableSceneBuilder(
                env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
            )
            self.table_scene.build()
            self._objs = []
            model_id = model_id_for_split(self.rlinf_split)
            for index in range(self.num_envs):
                builder = actors.get_actor_builder(self.scene, id=f"ycb:{model_id}")
                builder.set_scene_idxs([index])
                self._objs.append(builder.build(name=f"{model_id}-{index}"))
            self.obj = Actor.merge(self._objs, name="ycb_object")
            self.goal_site = actors.build_sphere(
                self.scene,
                radius=self.goal_thresh,
                color=[0, 1, 0, 1],
                name="goal_site",
                body_type="kinematic",
                add_collision=False,
            )
            self._hidden_objects.append(self.goal_site)

        def _after_reconfigure(self, options: dict) -> None:
            self.object_zs = []
            for obj in self._objs:
                collision_mesh = obj.get_first_collision_mesh()
                self.object_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
            self.object_zs = common.to_tensor(self.object_zs, device=self.device)

        def _initialize_episode(self, env_idx: torch.Tensor, options: dict) -> None:
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
                goal_offset = rng.uniform(*GOAL_Z_OFFSET_RANGE, size=count)
                object_position = self.obj.pose.p.clone()
                object_position[env_idx, :2] = torch.as_tensor(
                    object_xy, dtype=object_position.dtype, device=self.device
                )
                object_position[env_idx, 2] = self.object_zs[env_idx]
                object_orientation = torch.zeros_like(self.obj.pose.q)
                object_orientation[..., 0] = 1.0
                self.obj.set_pose(Pose.create_from_pq(object_position, object_orientation))

                goal_position = self.goal_site.pose.p.clone()
                goal_position[env_idx, :2] = torch.as_tensor(
                    goal_xy, dtype=goal_position.dtype, device=self.device
                )
                goal_position[env_idx, 2] = self.object_zs[env_idx] + torch.as_tensor(
                    goal_offset, dtype=goal_position.dtype, device=self.device
                )
                self.goal_site.set_pose(Pose.create_from_pq(goal_position))

                qpos = np.array(
                    [0.0, 0, 0, -np.pi * 2 / 3, 0, np.pi * 2 / 3, np.pi / 4, 0.04, 0.04]
                )
                qpos[:-2] += self._episode_rng.normal(
                    0, self.robot_init_qpos_noise, len(qpos) - 2
                )
                self.agent.reset(qpos)
                self.agent.robot.set_root_pose(sapien.Pose([-0.615, 0, 0]))

    @register_env(PICK_SINGLE_YCB_OBJECT_ID_ENV_ID, max_episode_steps=200, asset_download_ids=["ycb"])
    class ControlledPickSingleYCBObjectIDEnv(_ObjectVariationMixin, PickSingleYCBEnv):
        rlinf_split = "id"

    @register_env(PICK_SINGLE_YCB_OBJECT_OOD_ENV_ID, max_episode_steps=200, asset_download_ids=["ycb"])
    class ControlledPickSingleYCBObjectOODEnv(_ObjectVariationMixin, PickSingleYCBEnv):
        rlinf_split = "ood"

    _REGISTERED = True

