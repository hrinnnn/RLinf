"""Current-state one-chunk StackCube oracle based on ManiSkill's solver."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from .peg_privileged_oracle import _as_numpy, _first_vector, _normalize_delta


@dataclass(frozen=True)
class StackCubeOraclePlan:
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
    planner = importlib.import_module(
        "mani_skill.examples.motionplanning.panda.motionplanner"
    )
    utils = importlib.import_module(
        "mani_skill.examples.motionplanning.base_motionplanner.utils"
    )
    sapien = importlib.import_module("sapien")
    euler = importlib.import_module("transforms3d.euler")
    return (
        planner.PandaArmMotionPlanningSolver,
        utils.compute_grasp_info_by_obb,
        utils.get_actor_obb,
        sapien,
        euler.euler2quat,
    )


class StackCubePrivilegedChunkOracle:
    """Replan one fixed-size intervention from the exact current state."""

    def __init__(self, *, chunk_size: int = 5):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = int(chunk_size)
        self._phase = "reach"
        self._grasp_pose: Any | None = None

    def initialize_from_state(self, env: Any) -> str:
        """Enter the earliest safe phase that matches the takeover state."""
        base = env.unwrapped
        grasped = bool(_as_numpy(base.agent.is_grasping(base.cubeA)).reshape(-1)[0])
        cube_z = float(_first_vector(base.cubeA.pose.p, 3)[2])
        if grasped:
            # Policy grasps may satisfy the predicate before the fingers have
            # fully settled. Reinforce closure before any lift or lateral move.
            self._phase = "close"
        else:
            self._phase = "reach"
            self._grasp_pose = None
        return self._phase

    @staticmethod
    def _at_pose(tcp_pose: Any, target_pose: Any, tolerance: float = 0.025) -> bool:
        return (
            float(
                np.linalg.norm(
                    _first_vector(tcp_pose.p, 3) - _first_vector(target_pose.p, 3)
                )
            )
            < tolerance
        )

    def _find_grasp_pose(self, env: Any) -> Any:
        solver_cls, compute_grasp, get_obb, sapien, euler2quat = _load_symbols()
        base = env.unwrapped
        obb = get_obb(base.cubeA)
        approaching = np.array([0.0, 0.0, -1.0])
        closing = _as_numpy(base.agent.tcp.pose.to_transformation_matrix()[0, :3, 1])
        grasp = compute_grasp(
            obb, approaching=approaching, target_closing=closing, depth=0.025
        )
        nominal = base.agent.build_grasp_pose(
            approaching, grasp["closing"], grasp["center"]
        )
        solver = solver_cls(
            env,
            debug=False,
            vis=False,
            base_pose=base.agent.robot.pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
        )
        for angle in np.repeat(np.arange(0, np.pi * 2 / 3, np.pi / 2), 2) * np.tile(
            [1, -1], 2
        ):
            candidate = nominal * sapien.Pose(q=euler2quat(0, 0, angle))
            if self._plan_path(env, candidate, solver=solver) is not None:
                return candidate
        return nominal

    def _target(self, env: Any) -> tuple[Any | None, float, str]:
        _solver, _compute, _obb, sapien, _euler = _load_symbols()
        base = env.unwrapped
        if self._grasp_pose is None:
            self._grasp_pose = self._find_grasp_pose(env)
        reach = self._grasp_pose * sapien.Pose([0, 0, -0.05])
        if self._phase == "reach":
            if not self._at_pose(base.agent.tcp.pose, reach):
                return reach, 1.0, "reach"
            self._phase = "grasp"
        if self._phase == "grasp":
            if not self._at_pose(base.agent.tcp.pose, self._grasp_pose):
                return self._grasp_pose, 1.0, "grasp"
            self._phase = "close"
        if self._phase == "close":
            self._phase = "lift"
            return None, -1.0, "close"
        if self._phase == "lift":
            if float(_first_vector(base.cubeA.pose.p, 3)[2]) < 0.07:
                tcp = base.agent.tcp.pose.sp
                return sapien.Pose(tcp.p + np.array([0, 0, 0.08]), tcp.q), -1.0, "lift"
            self._phase = "align"
        if self._phase == "align":
            tcp = base.agent.tcp.pose.sp
            goal = base.cubeB.pose.sp * sapien.Pose(
                [0, 0, float(base.cube_half_size[2].item()) * 2]
            )
            offset = goal.p - base.cubeA.pose.sp.p
            target = sapien.Pose(tcp.p + offset, tcp.q)
            if not self._at_pose(base.agent.tcp.pose, target):
                return target, -1.0, "align"
            self._phase = "release"
        return None, 1.0, "release"

    def _plan_path(
        self, env: Any, target_pose: Any, *, solver: Any | None = None
    ) -> np.ndarray | None:
        solver_cls, _compute, _obb, _sapien, _euler = _load_symbols()
        base = env.unwrapped
        solver = solver or solver_cls(
            env,
            debug=False,
            vis=False,
            base_pose=base.agent.robot.pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
        )
        pose = solver._transform_pose_for_planning(target_pose)
        target = np.concatenate([_first_vector(pose.p, 3), _first_vector(pose.q, 4)])
        qpos = _as_numpy(base.agent.robot.get_qpos()).reshape(-1, 9)[0]
        result = solver.planner.plan_screw(
            target,
            qpos,
            time_step=float(base.control_timestep),
            use_point_cloud=False,
        )
        if result.get("status") != "Success":
            result = solver.planner.plan_qpos_to_pose(
                target,
                qpos,
                time_step=float(base.control_timestep),
                wrt_world=True,
            )
        if result.get("status") != "Success":
            return None
        return np.asarray(result["position"], dtype=np.float32)

    def plan(self, env: Any) -> StackCubeOraclePlan:
        base = env.unwrapped
        if str(base.control_mode) != "pd_joint_delta_pos":
            raise ValueError("StackCube privileged oracle requires pd_joint_delta_pos")
        target, gripper, phase = self._target(env)
        if target is None:
            actions = np.zeros((self.chunk_size, 8), dtype=np.float32)
            actions[:, -1] = gripper
            return StackCubeOraclePlan(actions, phase, True, gripper=gripper)
        path = self._plan_path(env, target)
        if path is None or len(path) == 0:
            actions = np.zeros((self.chunk_size, 8), dtype=np.float32)
            actions[:, -1] = gripper
            return StackCubeOraclePlan(actions, phase, False, gripper=gripper)
        indices = np.linspace(0, len(path) - 1, num=self.chunk_size, dtype=np.int64)
        targets = np.asarray([path[index, :7] for index in indices], dtype=np.float32)
        actions = np.zeros((self.chunk_size, 8), dtype=np.float32)
        actions[:, -1] = gripper
        return StackCubeOraclePlan(
            actions, phase, True, joint_targets=targets, gripper=gripper
        )
