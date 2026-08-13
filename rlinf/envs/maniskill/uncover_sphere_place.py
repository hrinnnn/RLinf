"""Controlled two-stage YCB uncover-and-place task for stage-localized OOD.

The task deliberately keeps the environment small and auditable: a kinematic
mug covers a dynamic sphere, and the policy must first park the mug and then
place the sphere in a bowl.  This module defines reset semantics and phase
predicates only; the motion-planning oracle is a separate follow-up gate.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import sapien
import torch

from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


UNcover_SPLITS = ("id", "handle_ood", "goal_ood")
UNCOVER_ENV_IDS = {
    "id": "RLinfUncoverSpherePlaceID-v1",
    "handle_ood": "RLinfUncoverSpherePlaceHandleOOD-v1",
    "goal_ood": "RLinfUncoverSpherePlaceGoalOOD-v1",
}

TABLE_Z = 0.02
MUG_HALF_SIZE = (0.03, 0.025, 0.025)
SPHERE_RADIUS = 0.018
BOWL_RADIUS = 0.065
PARKING_XY = np.array([-0.16, 0.16], dtype=np.float32)


def _yaw_quaternion(yaw: torch.Tensor) -> torch.Tensor:
    q = torch.zeros((*yaw.shape, 4), dtype=yaw.dtype, device=yaw.device)
    q[..., 0] = torch.cos(yaw / 2)
    q[..., 3] = torch.sin(yaw / 2)
    return q


class UncoverSpherePlaceEnv(BaseEnv):
    """Park a covering mug, then place the exposed sphere in a bowl."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Panda
    rlinf_split: Literal["id", "handle_ood", "goal_ood"] = "id"

    def __init__(self, *args, robot_uids="panda_wristcam", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.42, -0.48, 0.48], target=[0.0, 0.0, 0.10])
        return [CameraConfig("base_camera", pose, 224, 224, 1.1, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.55, -0.62, 0.55], [0.0, 0.0, 0.10])
        return CameraConfig("render_camera", pose, 512, 512, 1.05, 0.01, 100)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=0.01)
        self.table_scene.build()
        self.mug = actors.build_box(
            self.scene, half_sizes=MUG_HALF_SIZE, color=[0.72, 0.42, 0.16, 1],
            name="cover_mug", body_type="dynamic",
        )
        self.sphere = actors.build_sphere(
            self.scene, radius=SPHERE_RADIUS, color=[0.12, 0.25, 0.85, 1],
            name="target_sphere", body_type="dynamic",
        )
        self.bowl = self._build_bowl()

    def _build_bowl(self):
        builder = self.scene.create_actor_builder()
        color = [0.12, 0.72, 0.28, 1]
        builder.add_box_collision(half_size=[BOWL_RADIUS, BOWL_RADIUS, 0.008])
        builder.add_box_visual(
            half_size=[BOWL_RADIUS, BOWL_RADIUS, 0.008],
            material=sapien.render.RenderMaterial(base_color=color),
        )
        wall_height = 0.025
        wall = 0.006
        for pose, half_size in (
            (sapien.Pose(p=[BOWL_RADIUS, 0, wall_height]), [wall, BOWL_RADIUS, wall_height]),
            (sapien.Pose(p=[-BOWL_RADIUS, 0, wall_height]), [wall, BOWL_RADIUS, wall_height]),
            (sapien.Pose(p=[0, BOWL_RADIUS, wall_height]), [BOWL_RADIUS, wall, wall_height]),
            (sapien.Pose(p=[0, -BOWL_RADIUS, wall_height]), [BOWL_RADIUS, wall, wall_height]),
        ):
            builder.add_box_collision(pose=pose, half_size=half_size)
            builder.add_box_visual(
                pose=pose,
                half_size=half_size,
                material=sapien.render.RenderMaterial(base_color=color),
            )
        return builder.build_kinematic(name="target_bowl")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            b = len(env_idx)
            xy = self._batched_episode_rng.uniform(-0.035, 0.035, size=(b, 2))
            xy = torch.as_tensor(xy, dtype=self.sphere.pose.p.dtype, device=self.device)
            sphere_p = self.sphere.pose.p.clone()
            mug_p = self.mug.pose.p.clone()
            bowl_p = self.bowl.pose.p.clone()
            sphere_p[env_idx, :2] = xy
            sphere_p[env_idx, 2] = TABLE_Z + SPHERE_RADIUS
            mug_p[env_idx, :2] = xy
            # The cover rests just above the sphere rather than intersecting
            # it at reset; the sphere remains physically recoverable.
            mug_p[env_idx, 2] = TABLE_Z + 2 * SPHERE_RADIUS + MUG_HALF_SIZE[2] + 0.004
            bowl_xy = torch.tensor([0.16, -0.05], dtype=sphere_p.dtype, device=self.device)
            if self.rlinf_split == "goal_ood":
                bowl_xy = torch.tensor([0.16, 0.10], dtype=sphere_p.dtype, device=self.device)
            bowl_p[env_idx, :2] = bowl_xy
            bowl_p[env_idx, 2] = TABLE_Z
            mug_yaw = torch.zeros((b,), dtype=sphere_p.dtype, device=self.device)
            if self.rlinf_split == "handle_ood":
                mug_yaw.fill_(np.pi / 2)
            self.sphere.set_pose(Pose.create_from_pq(sphere_p, self.sphere.pose.q.clone()))
            self.sphere.set_linear_velocity(torch.zeros_like(self.sphere.linear_velocity))
            self.sphere.set_angular_velocity(torch.zeros_like(self.sphere.angular_velocity))
            self.mug.set_pose(Pose.create_from_pq(mug_p, _yaw_quaternion(mug_yaw)))
            self.bowl.set_pose(Pose.create_from_pq(bowl_p, self.bowl.pose.q.clone()))
            if not hasattr(self, "_ever_mug_parked"):
                self._ever_mug_parked = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                self._ever_sphere_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            else:
                self._ever_mug_parked[env_idx] = False
                self._ever_sphere_grasped[env_idx] = False

    def _get_obs_extra(self, info: dict):
        obs = {"tcp_pose": self.agent.tcp.pose.raw_pose, "bowl_pose": self.bowl.pose.raw_pose}
        if "state" in self.obs_mode:
            obs.update(
                mug_pose=self.mug.pose.raw_pose,
                sphere_pose=self.sphere.pose.raw_pose,
                sphere_to_bowl=self.bowl.pose.p - self.sphere.pose.p,
            )
        return obs

    def evaluate(self):
        mug_parked = torch.linalg.norm(self.mug.pose.p[:, :2] - torch.tensor(
            PARKING_XY, dtype=self.mug.pose.p.dtype, device=self.device
        ), dim=-1) <= 0.055
        sphere_grasped = self.agent.is_grasping(self.sphere)
        bowl_delta = self.sphere.pose.p[:, :2] - self.bowl.pose.p[:, :2]
        sphere_in_bowl = torch.linalg.norm(bowl_delta, dim=-1) <= BOWL_RADIUS * 0.55
        sphere_in_bowl &= torch.abs(self.sphere.pose.p[:, 2] - (TABLE_Z + SPHERE_RADIUS)) <= 0.025
        sphere_released = ~sphere_grasped
        sphere_static = self.sphere.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        self._ever_mug_parked |= mug_parked
        self._ever_sphere_grasped |= sphere_grasped
        success = self._ever_mug_parked & self._ever_sphere_grasped & sphere_in_bowl & sphere_released & sphere_static
        return {
            "success": success,
            "mug_parked": mug_parked,
            "ever_mug_parked": self._ever_mug_parked,
            "sphere_grasped": sphere_grasped,
            "ever_sphere_grasped": self._ever_sphere_grasped,
            "sphere_in_bowl": sphere_in_bowl,
            "sphere_released": sphere_released,
            "sphere_static": sphere_static,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return (
            info["ever_mug_parked"].float()
            + info["ever_sphere_grasped"].float()
            + info["sphere_in_bowl"].float()
            + info["success"].float()
        )

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        # The phase predicates are already bounded event indicators; retaining
        # the same scale keeps reward and evaluation semantics aligned.
        return self.compute_dense_reward(obs, action, info)


def register_uncover_sphere_place_variants() -> None:
    """Register ID and two single-factor stage-localized OOD variants."""

    import gymnasium as gym

    if all(env_id in gym.registry for env_id in UNCOVER_ENV_IDS.values()):
        return
    for split, env_id in UNCOVER_ENV_IDS.items():
        cls = type(
            f"UncoverSpherePlace{split.title().replace('_', '')}Env",
            (UncoverSpherePlaceEnv,),
            {"rlinf_split": split},
        )
        register_env(env_id, max_episode_steps=160)(cls)
