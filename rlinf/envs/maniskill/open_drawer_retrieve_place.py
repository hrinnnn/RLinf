"""A controlled multi-stage drawer retrieval task for robot-gated DAgger.

This module is registered explicitly by experiment entrypoints. Keeping it
outside ``tasks`` lets distribution-only tools run without importing SAPIEN.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

from rlinf.envs.maniskill.open_drawer_retrieve_place_spec import (
    DRAWER_OPEN_THRESHOLD,
    DRAWER_ORIGIN,
    DRAWER_TRAVEL,
    ENV_IDS,
    HANDLE_OFFSET_BY_SPLIT,
    TaskSplit,
    sample_episode_parameters,
)


OBJECT_HALF_SIZE = np.array([0.055, 0.024, 0.024], dtype=np.float64)
TARGET_INNER_HALF_SIZE = np.array([0.075, 0.075], dtype=np.float64)
HANDLE_LOCAL_X = -0.210
HANDLE_LOCAL_Z = 0.095


class OpenDrawerRetrievePlaceEnv(BaseEnv):
    """Open a drawer, retrieve an object, and place it in a target tray."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Panda
    rlinf_split: TaskSplit = "id"

    def __init__(
        self,
        *args,
        robot_uids: str = "panda_wristcam",
        robot_init_qpos_noise: float = 0.01,
        **kwargs,
    ):
        self.robot_init_qpos_noise = float(robot_init_qpos_noise)
        self.handle_lateral_offset = HANDLE_OFFSET_BY_SPLIT[self.rlinf_split]
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=(0.62, -0.62, 0.62), target=(0.02, 0.0, 0.08))
        return [CameraConfig("base_camera", pose, 224, 224, 1.15, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=(0.72, -0.72, 0.72), target=(0.02, 0.0, 0.08))
        return CameraConfig("render_camera", pose, 512, 512, 1.05, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    @staticmethod
    def _add_box(link, *, center, half_size, color, density=1000.0):
        pose = sapien.Pose(p=center)
        link.add_box_collision(pose=pose, half_size=half_size, density=density)
        link.add_box_visual(pose=pose, half_size=half_size, material=color)

    def _build_drawer(self):
        builder = self.scene.create_articulation_builder()
        builder.set_name("task_drawer")
        builder.disable_self_collisions = True
        builder.initial_pose = sapien.Pose(p=DRAWER_ORIGIN)

        cabinet = builder.create_link_builder()
        cabinet.set_name("cabinet")
        self._add_box(cabinet, center=(0.135, 0, 0.10), half_size=(0.015, 0.18, 0.10), color=(0.30, 0.31, 0.34))
        self._add_box(cabinet, center=(0, -0.17, 0.10), half_size=(0.15, 0.015, 0.10), color=(0.34, 0.35, 0.38))
        self._add_box(cabinet, center=(0, 0.17, 0.10), half_size=(0.15, 0.015, 0.10), color=(0.34, 0.35, 0.38))
        self._add_box(cabinet, center=(0, 0, 0.205), half_size=(0.15, 0.185, 0.015), color=(0.38, 0.39, 0.42))

        drawer = builder.create_link_builder(cabinet)
        drawer.set_name("drawer")
        drawer.set_joint_name("drawer_joint")
        drawer.set_joint_properties(
            "prismatic",
            limits=[[-DRAWER_TRAVEL, 0.0]],
            pose_in_parent=sapien.Pose(),
            pose_in_child=sapien.Pose(),
            friction=0.05,
            damping=1.0,
        )
        self._add_box(drawer, center=(0, 0, 0.022), half_size=(0.13, 0.15, 0.012), color=(0.58, 0.48, 0.36), density=450.0)
        self._add_box(drawer, center=(0, -0.142, 0.062), half_size=(0.13, 0.008, 0.052), color=(0.55, 0.45, 0.34), density=450.0)
        self._add_box(drawer, center=(0, 0.142, 0.062), half_size=(0.13, 0.008, 0.052), color=(0.55, 0.45, 0.34), density=450.0)
        self._add_box(drawer, center=(-0.13, 0, 0.075), half_size=(0.012, 0.15, 0.075), color=(0.50, 0.40, 0.30), density=450.0)
        self._add_box(
            drawer,
            center=(HANDLE_LOCAL_X, self.handle_lateral_offset, HANDLE_LOCAL_Z),
            half_size=(0.018, 0.025, 0.012),
            color=(0.88, 0.72, 0.18),
            density=900.0,
        )
        return builder.build(fix_root_link=True)

    def _build_object(self):
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=OBJECT_HALF_SIZE, density=500.0)
        builder.add_box_visual(half_size=OBJECT_HALF_SIZE, material=(0.08, 0.35, 0.90))
        builder.initial_pose = sapien.Pose(p=[0.1, 0, 0.07])
        return builder.build("retrieve_object")

    def _build_target_tray(self):
        builder = self.scene.create_actor_builder()
        tray_color = (0.10, 0.72, 0.28)
        builder.add_box_collision(pose=sapien.Pose(p=[0, 0, 0.006]), half_size=(0.085, 0.085, 0.006))
        builder.add_box_visual(pose=sapien.Pose(p=[0, 0, 0.006]), half_size=(0.085, 0.085, 0.006), material=tray_color)
        for center, half_size in (
            ((0.080, 0, 0.022), (0.005, 0.085, 0.022)),
            ((-0.080, 0, 0.022), (0.005, 0.085, 0.022)),
            ((0, 0.080, 0.022), (0.085, 0.005, 0.022)),
            ((0, -0.080, 0.022), (0.085, 0.005, 0.022)),
        ):
            builder.add_box_collision(pose=sapien.Pose(p=center), half_size=half_size)
            builder.add_box_visual(pose=sapien.Pose(p=center), half_size=half_size, material=tray_color)
        builder.initial_pose = sapien.Pose(p=[0.03, -0.30, 0])
        return builder.build_kinematic("target_tray")

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        self.drawer = self._build_drawer()
        self.drawer_link = self.drawer.links_map["drawer"]
        self.obj = self._build_object()
        self.target_tray = self._build_target_tray()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            count = len(env_idx)
            self.table_scene.initialize(env_idx)
            params = sample_episode_parameters(self._batched_episode_rng, count, split=self.rlinf_split)

            drawer_qpos = self.drawer.get_qpos().clone()
            drawer_qpos[env_idx, 0] = torch.as_tensor(params.drawer_qpos, device=self.device, dtype=drawer_qpos.dtype)
            self.drawer.set_qpos(drawer_qpos)
            self.drawer.set_qvel(torch.zeros_like(self.drawer.get_qvel()))

            object_p = self.obj.pose.p.clone()
            object_q = self.obj.pose.q.clone()
            object_p[env_idx, :2] = torch.as_tensor(
                params.object_local_xy + DRAWER_ORIGIN[:2], device=self.device, dtype=object_p.dtype
            )
            object_p[env_idx, 2] = 0.058
            yaw = torch.as_tensor(params.object_yaw, device=self.device, dtype=object_q.dtype)
            object_q[env_idx] = 0
            object_q[env_idx, 0] = torch.cos(yaw / 2)
            object_q[env_idx, 3] = torch.sin(yaw / 2)
            self.obj.set_pose(Pose.create_from_pq(object_p, object_q))
            self.obj.set_linear_velocity(torch.zeros_like(self.obj.linear_velocity))
            self.obj.set_angular_velocity(torch.zeros_like(self.obj.angular_velocity))

            target_p = self.target_tray.pose.p.clone()
            target_p[env_idx, :2] = torch.as_tensor(params.goal_xy, device=self.device, dtype=target_p.dtype)
            target_p[env_idx, 2] = 0
            self.target_tray.set_pose(Pose.create_from_pq(target_p, self.target_tray.pose.q.clone()))

            if not hasattr(self, "_ever_drawer_opened"):
                self._ever_drawer_opened = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                self._ever_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                self._ever_lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            else:
                self._ever_drawer_opened[env_idx] = False
                self._ever_grasped[env_idx] = False
                self._ever_lifted[env_idx] = False

    @property
    def handle_world_position(self) -> torch.Tensor:
        local = torch.tensor(
            [HANDLE_LOCAL_X, self.handle_lateral_offset, HANDLE_LOCAL_Z, 1.0],
            device=self.device,
            dtype=self.drawer_link.pose.p.dtype,
        )
        transform = self.drawer_link.pose.to_transformation_matrix()
        return torch.matmul(transform, local)[:, :3]

    def evaluate(self):
        drawer_opened = self.drawer.get_qpos()[:, 0] <= -DRAWER_OPEN_THRESHOLD
        is_grasped = self.agent.is_grasping(self.obj)
        object_lifted = self.obj.pose.p[:, 2] >= 0.12
        target_delta = self.obj.pose.p[:, :2] - self.target_tray.pose.p[:, :2]
        object_in_target = (
            (torch.abs(target_delta[:, 0]) <= TARGET_INNER_HALF_SIZE[0])
            & (torch.abs(target_delta[:, 1]) <= TARGET_INNER_HALF_SIZE[1])
            & (self.obj.pose.p[:, 2] >= OBJECT_HALF_SIZE[2] - 0.005)
            & (self.obj.pose.p[:, 2] <= 0.085)
        )
        object_released = ~is_grasped
        is_robot_static = self.agent.is_static(0.2)
        self._ever_drawer_opened |= drawer_opened
        self._ever_grasped |= is_grasped
        self._ever_lifted |= object_lifted
        success = self._ever_drawer_opened & object_in_target & object_released & is_robot_static
        return {
            "success": success,
            "drawer_opened": drawer_opened,
            "ever_drawer_opened": self._ever_drawer_opened,
            "is_grasped": is_grasped,
            "ever_grasped": self._ever_grasped,
            "object_lifted": object_lifted,
            "ever_lifted": self._ever_lifted,
            "object_in_target": object_in_target,
            "object_released": object_released,
            "is_robot_static": is_robot_static,
        }

    def _get_obs_extra(self, info: dict):
        obs = {"tcp_pose": self.agent.tcp_pose.raw_pose}
        if "state" in self.obs_mode:
            obs.update(
                drawer_qpos=self.drawer.get_qpos(),
                object_pose=self.obj.pose.raw_pose,
                target_pose=self.target_tray.pose.raw_pose,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return (
            info["ever_drawer_opened"].float()
            + info["ever_grasped"].float()
            + info["ever_lifted"].float()
            + info["object_in_target"].float()
            + info["success"].float()
        )

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs, action, info) / 5.0


@register_env(ENV_IDS["id"], max_episode_steps=400)
class OpenDrawerRetrievePlaceIDEnv(OpenDrawerRetrievePlaceEnv):
    rlinf_split = "id"


@register_env(ENV_IDS["handle_ood"], max_episode_steps=400)
class OpenDrawerRetrievePlaceHandleOODEnv(OpenDrawerRetrievePlaceEnv):
    rlinf_split = "handle_ood"


@register_env(ENV_IDS["grasp_ood"], max_episode_steps=400)
class OpenDrawerRetrievePlaceGraspOODEnv(OpenDrawerRetrievePlaceEnv):
    rlinf_split = "grasp_ood"


@register_env(ENV_IDS["goal_ood"], max_episode_steps=400)
class OpenDrawerRetrievePlaceGoalOODEnv(OpenDrawerRetrievePlaceEnv):
    rlinf_split = "goal_ood"
