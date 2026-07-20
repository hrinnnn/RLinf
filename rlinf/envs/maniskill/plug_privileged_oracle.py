"""One-chunk PlugCharger expert using ManiSkill's official planner geometry."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from .peg_privileged_oracle import _as_numpy, _first_vector, _normalize_delta


@dataclass(frozen=True)
class PlugOraclePlan:
    actions: np.ndarray
    phase: str
    planning_succeeded: bool
    joint_targets: np.ndarray | None = None
    gripper: float = 1.0

    def action_at(self, qpos: Any, step_index: int) -> np.ndarray:
        if self.joint_targets is None:
            return self.actions[min(step_index, len(self.actions) - 1)]
        current = _as_numpy(qpos).reshape(-1, 9)[0].astype(np.float32)
        target = self.joint_targets[min(step_index, len(self.joint_targets) - 1)]
        arm = _normalize_delta(
            target[:7] - current[:7],
            np.full(7, -0.1, dtype=np.float32),
            np.full(7, 0.1, dtype=np.float32),
        )
        return np.concatenate([arm.astype(np.float32), [self.gripper]])


def _load_symbols():
    planner = importlib.import_module("mani_skill.examples.motionplanning.panda.motionplanner")
    utils = importlib.import_module("mani_skill.examples.motionplanning.base_motionplanner.utils")
    sapien = importlib.import_module("sapien")
    euler = importlib.import_module("transforms3d.euler")
    return planner.PandaArmMotionPlanningSolver, utils.compute_grasp_info_by_obb, sapien, euler.euler2quat


class PlugChargerPrivilegedChunkOracle:
    """Plan from current simulator state and execute no more than one chunk."""

    def __init__(self, *, chunk_size: int = 10):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = int(chunk_size)
        self._grasp_pose: Any | None = None
        self._phase = "reach"

    @staticmethod
    def _at_pose(tcp_pose: Any, target_pose: Any, tolerance: float = 0.02) -> bool:
        return float(np.linalg.norm(_first_vector(tcp_pose.p, 3) - _first_vector(target_pose.p, 3))) < tolerance

    def _initialize_grasp_pose(self, base_env: Any) -> None:
        if self._grasp_pose is not None:
            return
        _solver, compute_grasp_info_by_obb, sapien, euler2quat = _load_symbols()
        import trimesh

        base_size = np.asarray(base_env._base_size, dtype=np.float32) * 2
        obb = trimesh.primitives.Box(
            extents=base_size,
            transform=base_env.charger_base_pose.sp.to_transformation_matrix(),
        )
        approaching = np.array([0.0, 0.0, -1.0])
        closing = _as_numpy(base_env.agent.tcp.pose.sp.to_transformation_matrix()[:3, 1])
        grasp = compute_grasp_info_by_obb(obb, approaching=approaching, target_closing=closing, depth=0.025)
        pose = base_env.agent.build_grasp_pose(approaching, grasp["closing"], grasp["center"])
        self._grasp_pose = pose * sapien.Pose(q=euler2quat(0, np.deg2rad(15), 0))

    def _target(self, base_env: Any) -> tuple[Any | None, float, str]:
        _solver, _grasp, sapien, _euler = _load_symbols()
        self._initialize_grasp_pose(base_env)
        assert self._grasp_pose is not None
        reach = self._grasp_pose * sapien.Pose([0, 0, -0.05])
        if self._phase == "reach":
            if not self._at_pose(base_env.agent.tcp.pose, reach):
                return reach, 1.0, "reach"
            self._phase = "grasp"
        if self._phase == "grasp":
            if not self._at_pose(base_env.agent.tcp.pose, self._grasp_pose):
                return self._grasp_pose, 1.0, "grasp"
            self._phase = "close"
        if self._phase == "close":
            self._phase = "preinsert"
            return None, -1.0, "close"
        # These are the official solution's relative-pose equations.  The
        # current charger/TCP state makes them robust to policy interruptions.
        current_tcp = base_env.agent.tcp.pose.sp
        current_charger = base_env.charger.pose.sp
        insert = base_env.goal_pose.sp * current_charger.inv() * current_tcp
        preinsert = base_env.goal_pose.sp * sapien.Pose([-0.05, 0, 0]) * current_charger.inv() * current_tcp
        if self._phase == "preinsert":
            if not self._at_pose(base_env.agent.tcp.pose, preinsert):
                return preinsert, -1.0, "preinsert"
            self._phase = "insert"
        return insert, -1.0, "insert"

    def _plan_path(self, env: Any, target_pose: Any) -> np.ndarray | None:
        solver_cls, _grasp, _sapien, _euler = _load_symbols()
        base_env = env.unwrapped
        solver = solver_cls(
            env, debug=False, vis=False, base_pose=base_env.agent.robot.pose,
            visualize_target_grasp_pose=False, print_env_info=False,
            joint_vel_limits=0.5, joint_acc_limits=0.5,
        )
        pose = solver._transform_pose_for_planning(target_pose)
        target = np.concatenate([_first_vector(pose.p, 3), _first_vector(pose.q, 4)])
        qpos = _as_numpy(base_env.agent.robot.get_qpos()).reshape(-1, 9)[0]
        result = solver.planner.plan_screw(target, qpos, time_step=float(base_env.control_timestep), use_point_cloud=False)
        if result.get("status") != "Success":
            result = solver.planner.plan_qpos_to_pose(target, qpos, time_step=float(base_env.control_timestep), wrt_world=True)
        if result.get("status") != "Success":
            return None
        return np.asarray(result["position"], dtype=np.float32)

    def plan(self, env: Any) -> PlugOraclePlan:
        base_env = env.unwrapped
        if str(base_env.control_mode) != "pd_joint_delta_pos":
            raise ValueError("Plug privileged oracle requires pd_joint_delta_pos")
        target, gripper, phase = self._target(base_env)
        if target is None:
            actions = np.zeros((self.chunk_size, 8), dtype=np.float32)
            actions[:, -1] = gripper
            return PlugOraclePlan(actions, phase, True, gripper=gripper)
        path = self._plan_path(env, target)
        if path is None or len(path) == 0:
            actions = np.zeros((self.chunk_size, 8), dtype=np.float32)
            actions[:, -1] = gripper
            return PlugOraclePlan(actions, phase, False, gripper=gripper)
        targets = np.asarray([path[min(index, len(path) - 1), :7] for index in range(self.chunk_size)], dtype=np.float32)
        actions = np.zeros((self.chunk_size, 8), dtype=np.float32)
        actions[:, -1] = gripper
        return PlugOraclePlan(actions, phase, True, joint_targets=targets, gripper=gripper)
