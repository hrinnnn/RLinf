"""A one-chunk privileged-state PegInsertionSide oracle.

This intentionally reuses ManiSkill's official Panda motion-planning solver,
but never calls its top-level ``solve`` helper because that helper resets the
environment.  The oracle plans from the simulator's current robot/object state
and returns at most one policy action chunk in ``pd_joint_delta_pos`` format.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np


_DEFAULT_ARM_DELTA_LOWER = -0.1
_DEFAULT_ARM_DELTA_UPPER = 0.1


@dataclass(frozen=True)
class PegOraclePlan:
    actions: np.ndarray
    phase: str
    planning_succeeded: bool


def _as_bool(value: Any) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value, dtype=bool).reshape(-1).any())


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _first_vector(value: Any, width: int) -> np.ndarray:
    array = _as_numpy(value).astype(np.float32)
    return array.reshape(-1, width)[0]


def _load_motion_planning_symbols():
    planner_module = importlib.import_module(
        "mani_skill.examples.motionplanning.panda.motionplanner"
    )
    utils_module = importlib.import_module(
        "mani_skill.examples.motionplanning.base_motionplanner.utils"
    )
    sapien = importlib.import_module("sapien")
    return (
        planner_module.PandaArmMotionPlanningSolver,
        utils_module.compute_grasp_info_by_obb,
        utils_module.get_actor_obb,
        sapien,
    )


def _normalize_delta(delta: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    clipped = np.clip(delta, lower, upper)
    return np.clip(2.0 * (clipped - lower) / (upper - lower) - 1.0, -1.0, 1.0)


class PegPrivilegedChunkOracle:
    """Plan one non-resetting privileged action chunk for a single ManiSkill env."""

    def __init__(self, *, chunk_size: int = 10):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = int(chunk_size)
        # Match ManiSkill's reference solution: grasp geometry is established
        # once at reset, then every one-chunk re-plan targets that same pose.
        self._peg_init_pose: Any | None = None
        self._grasp_pose: Any | None = None
        self._reach_pose: Any | None = None
        self._phase = "reach"

    @staticmethod
    def _at_pose(tcp_pose: Any, target_pose: Any, *, tolerance: float = 0.02) -> bool:
        tcp = _first_vector(tcp_pose.p, 3)
        target = _first_vector(target_pose.p, 3)
        return float(np.linalg.norm(tcp - target)) < tolerance

    def _initialize_reference_poses(self, base_env: Any) -> None:
        if self._grasp_pose is not None:
            return
        (
            _solver_cls,
            compute_grasp_info_by_obb,
            get_actor_obb,
            sapien,
        ) = _load_motion_planning_symbols()
        obb = get_actor_obb(base_env.peg)
        approaching = np.array([0.0, 0.0, -1.0])
        target_closing = _as_numpy(
            base_env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1]
        )
        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=0.025,
        )
        grasp_pose = base_env.agent.build_grasp_pose(
            approaching, grasp_info["closing"], grasp_info["center"]
        )
        offset = sapien.Pose(
            [-max(0.05, float(base_env.peg_half_sizes[0, 0]) / 2 + 0.01), 0, 0]
        )
        self._peg_init_pose = base_env.peg.pose
        self._grasp_pose = grasp_pose * offset
        self._reach_pose = self._grasp_pose * sapien.Pose([0, 0, -0.05])

    def _target(self, base_env: Any) -> tuple[Any, float, str]:
        _solver_cls, _grasp, _obb, sapien = _load_motion_planning_symbols()
        self._initialize_reference_poses(base_env)
        assert self._peg_init_pose is not None
        assert self._grasp_pose is not None
        assert self._reach_pose is not None
        if self._phase == "reach":
            if self._at_pose(base_env.agent.tcp.pose, self._reach_pose):
                self._phase = "grasp"
            else:
                return self._reach_pose, 1.0, "reach"

        if self._phase == "grasp":
            # The reference solver closes for six control steps.  A chunk can
            # be shorter than the remaining motion, so retain this waypoint
            # until the actual TCP reaches it.
            if self._at_pose(base_env.agent.tcp.pose, self._grasp_pose):
                self._phase = "preinsert"
            return self._grasp_pose, -1.0, "grasp"

        insert_pose = base_env.goal_pose * self._peg_init_pose.inv() * self._grasp_pose
        if self._phase == "preinsert":
            preinsert_pose = insert_pose * sapien.Pose(
                [-0.01 - float(base_env.peg_half_sizes[0, 0]), 0, 0]
            )
            if self._at_pose(base_env.agent.tcp.pose, preinsert_pose):
                self._phase = "insert"
            return preinsert_pose, -1.0, "preinsert"
        return insert_pose * sapien.Pose([0.05, 0, 0]), -1.0, "insert"

    def _plan_qpos_path(self, env: Any, target_pose: Any) -> np.ndarray | None:
        solver_cls, _grasp, _obb, _sapien = _load_motion_planning_symbols()
        base_env = env.unwrapped
        solver = solver_cls(
            env,
            debug=False,
            vis=False,
            base_pose=base_env.agent.robot.pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
            joint_vel_limits=0.5,
            joint_acc_limits=0.5,
        )
        pose = target_pose
        target = np.concatenate([_first_vector(pose.p, 3), _first_vector(pose.q, 4)])
        qpos = _as_numpy(base_env.agent.robot.get_qpos()).reshape(-1, 9)[0]
        result = solver.planner.plan_screw(
            target,
            qpos,
            time_step=float(base_env.control_timestep),
            use_point_cloud=False,
        )
        if result.get("status") != "Success":
            result = solver.planner.plan_qpos_to_pose(
                target,
                qpos,
                time_step=float(base_env.control_timestep),
                wrt_world=True,
            )
        if result.get("status") != "Success":
            return None
        return np.asarray(result["position"], dtype=np.float32)

    def plan(self, env: Any) -> PegOraclePlan:
        base_env = env.unwrapped
        if str(base_env.control_mode) != "pd_joint_delta_pos":
            raise ValueError("Peg privileged oracle requires pd_joint_delta_pos")
        target_pose, gripper, phase = self._target(base_env)
        path = self._plan_qpos_path(env, target_pose)
        current_qpos = _as_numpy(base_env.agent.robot.get_qpos()).reshape(-1, 9)[0].astype(np.float32)
        if path is None or len(path) == 0:
            hold = np.zeros((self.chunk_size, 8), dtype=np.float32)
            hold[:, -1] = gripper
            return PegOraclePlan(hold, phase, False)

        lower = np.full(7, _DEFAULT_ARM_DELTA_LOWER, dtype=np.float32)
        upper = np.full(7, _DEFAULT_ARM_DELTA_UPPER, dtype=np.float32)
        actions: list[np.ndarray] = []
        predicted_qpos = current_qpos.copy()
        for step in range(self.chunk_size):
            target_qpos = path[min(step, len(path) - 1), :7]
            arm = _normalize_delta(target_qpos - predicted_qpos[:7], lower, upper)
            actions.append(np.concatenate([arm.astype(np.float32), [gripper]]))
            predicted_qpos[:7] = target_qpos
        return PegOraclePlan(np.stack(actions), phase, True)
