"""Current-state Panda motion-planning oracle for UncoverSpherePlace."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from transforms3d.euler import euler2quat

from .peg_privileged_oracle import _as_numpy, _first_vector, _normalize_delta
from .uncover_sphere_place import BOWL_RADIUS, MUG_HALF_SIZE, PARKING_XY, SPHERE_RADIUS, TABLE_Z


@dataclass(frozen=True)
class UncoverSpherePlaceOraclePlan:
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
    return (
        planner.PandaArmMotionPlanningSolver,
        utils.compute_grasp_info_by_obb,
        utils.get_actor_obb,
        sapien,
    )


class UncoverSpherePlacePrivilegedChunkOracle:
    """Plan one fixed-size action chunk from the exact current simulator state."""

    def __init__(self, *, chunk_size: int = 10):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = int(chunk_size)
        self._phase = "cover_reach"
        self._cover_grasp_pose: Any | None = None
        self._sphere_grasp_pose: Any | None = None
        self._object_to_tcp: Any | None = None
        self._cover_attempts = 0
        self._sphere_attempts = 0
        self._sphere_close_chunks = 0
        self._sphere_stable_close_chunks = 0

    def resume_from_current_state(self, phase: str) -> None:
        """Resume planning from an already-created physical intermediate state.

        This is used by the oracle audit and by stage-localized data collection:
        the simulator state is preserved, while planner-local cached poses are
        deliberately discarded so the next plan is derived from live actors.
        """
        allowed = {"cover_reach", "sphere_reach", "sphere_lift"}
        if phase not in allowed:
            raise ValueError(f"unsupported resume phase: {phase}")
        self._phase = phase
        self._cover_grasp_pose = None
        self._sphere_grasp_pose = None
        self._object_to_tcp = None
        self._cover_attempts = 0
        self._sphere_attempts = 0
        self._sphere_close_chunks = 0
        self._sphere_stable_close_chunks = 0

    @staticmethod
    def _at_pose(tcp_pose: Any, target_pose: Any, tolerance: float = 0.025) -> bool:
        return float(
            np.linalg.norm(
                _first_vector(tcp_pose.p, 3) - _first_vector(target_pose.p, 3)
            )
        ) < tolerance

    @staticmethod
    def _pose_from_actor(actor: Any, sapien: Any) -> Any:
        pose = actor.pose.sp
        return sapien.Pose(p=np.asarray(pose.p), q=np.asarray(pose.q))

    def _planner(self, env: Any) -> Any:
        solver_cls, _compute, _obb, sapien = _load_symbols()
        base = env.unwrapped
        return solver_cls(
            env,
            debug=False,
            vis=False,
            base_pose=base.agent.robot.pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
            joint_vel_limits=0.5,
            joint_acc_limits=0.5,
        )

    def _find_grasp_pose(self, env: Any, actor: Any, *, attempt: int = 0) -> Any:
        solver_cls, compute_grasp, get_obb, sapien = _load_symbols()
        base = env.unwrapped
        obb = get_obb(actor)
        approaching = np.array([0.0, 0.0, -1.0])
        closing = _as_numpy(
            base.agent.tcp.pose.to_transformation_matrix()[0, :3, 1]
        )
        grasp = compute_grasp(
            obb,
            approaching=approaching,
            target_closing=closing,
            depth=0.025,
        )
        # The box cover needs its live geometric center because its OBB
        # surface offset is unstable under yaw. For the sphere, preserve the
        # official ManiSkill grasp center returned by the OBB helper.
        if "sphere" in str(getattr(actor, "name", "")):
            actor_center = grasp["center"]
        else:
            actor_center = _first_vector(actor.pose.p, 3)
        nominal = base.agent.build_grasp_pose(
            approaching, grasp["closing"], actor_center
        )
        # A cover is wider than a sphere and its stable grasp depends on the
        # yaw of the fingers. Select the first candidate for which the
        # official planner finds a collision-free path.
        solver = solver_cls(
            env,
            debug=False,
            vis=False,
            base_pose=base.agent.robot.pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
            joint_vel_limits=0.5,
            joint_acc_limits=0.5,
        )
        angles = (0.0, np.pi / 2, -np.pi / 2, np.pi)
        if "sphere" in str(getattr(actor, "name", "")):
            # A sphere has no preferred yaw.  Trying a different in-plane
            # finger orientation after a failed grasp avoids repeating the
            # same marginal contact caused by the preceding cover motion.
            if getattr(base, "rlinf_split", "id") == "handle_ood":
                # The rotated cover leaves a different sphere/tcp contact
                # geometry.  The axis-aligned candidate is the stable first
                # contact for this stage-local split.
                angles = (0.0, np.pi / 2, -np.pi / 2, np.pi)
            else:
                official_angles = np.arange(0.0, np.pi * 2 / 3, np.pi / 2) + np.pi / 4
                official_angles = np.repeat(official_angles, 2)
                official_angles[1::2] *= -1
                angles = tuple(np.roll(official_angles, -attempt % len(official_angles)))
        candidates = [
            nominal * sapien.Pose(q=euler2quat(0, 0, angle))
            for angle in angles
        ]
        for candidate in candidates:
            if self._plan_path(env, candidate, solver=solver) is not None:
                return candidate
        return nominal

    def _plan_path(
        self, env: Any, target_pose: Any, *, solver: Any | None = None
    ) -> np.ndarray | None:
        solver = solver or self._planner(env)
        base = env.unwrapped
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

    def _relative_tcp_pose(self, base: Any, actor: Any, sapien: Any) -> Any:
        return self._pose_from_actor(actor, sapien).inv() * base.agent.tcp.pose.sp

    def _object_target_tcp(self, base: Any, actor: Any, target: Any, sapien: Any) -> Any:
        if self._object_to_tcp is None:
            self._object_to_tcp = self._relative_tcp_pose(base, actor, sapien)
        return target * self._object_to_tcp

    def _target(self, env: Any) -> tuple[Any | None, float, str]:
        _solver, _compute, _obb, sapien = _load_symbols()
        base = env.unwrapped

        if self._phase == "cover_reach":
            # The dynamic cover can settle while the arm is approaching, so
            # refresh the target from the live pose until grasp is acquired.
            self._cover_grasp_pose = self._find_grasp_pose(env, base.mug)
            reach = self._cover_grasp_pose * sapien.Pose([0, 0, -0.05])
            if self._at_pose(base.agent.tcp.pose.sp, reach):
                self._phase = "cover_grasp"
            else:
                return reach, 1.0, "cover_reach"

        if self._phase == "cover_grasp":
            assert self._cover_grasp_pose is not None
            self._cover_grasp_pose = self._find_grasp_pose(env, base.mug)
            if not self._at_pose(base.agent.tcp.pose.sp, self._cover_grasp_pose):
                return self._cover_grasp_pose, 1.0, "cover_grasp"
            self._phase = "cover_close"

        if self._phase == "cover_close":
            # Check grasp only after this closing chunk has been executed.
            self._phase = "cover_settle"
            return None, -1.0, "cover_close"

        if self._phase == "cover_settle":
            if bool(np.asarray(base.agent.is_grasping(base.mug)).reshape(-1)[0]):
                self._phase = "cover_lift"
            else:
                self._cover_attempts += 1
                if self._cover_attempts > 3:
                    self._phase = "failed"
                else:
                    self._phase = "cover_reach"
                    self._cover_grasp_pose = None
                    self._object_to_tcp = None
                    return self._target(env)

        if self._phase in {"cover_lift", "cover_move", "cover_place"}:
            if not bool(np.asarray(base.agent.is_grasping(base.mug)).reshape(-1)[0]):
                self._phase = "cover_reach"
                self._cover_grasp_pose = None
                self._object_to_tcp = None
                return self._target(env)
            mug_pose = self._pose_from_actor(base.mug, sapien)
            if self._phase == "cover_lift":
                target_mug = sapien.Pose(
                    p=[mug_pose.p[0], mug_pose.p[1], 0.18], q=mug_pose.q
                )
                target = self._object_target_tcp(base, base.mug, target_mug, sapien)
                if _first_vector(base.mug.pose.p, 3)[2] >= 0.16:
                    self._phase = "cover_move"
                else:
                    return target, -1.0, "cover_lift"
            if self._phase == "cover_move":
                target_mug = sapien.Pose(
                    p=[PARKING_XY[0], PARKING_XY[1], 0.18], q=mug_pose.q
                )
                target = self._object_target_tcp(base, base.mug, target_mug, sapien)
                if np.linalg.norm(_first_vector(base.mug.pose.p, 3)[:2] - PARKING_XY) < 0.04:
                    self._phase = "cover_place"
                else:
                    return target, -1.0, "cover_move"
            if self._phase == "cover_place":
                target_mug = sapien.Pose(
                    p=[PARKING_XY[0], PARKING_XY[1], TABLE_Z + MUG_HALF_SIZE[2]],
                    q=mug_pose.q,
                )
                target = self._object_target_tcp(base, base.mug, target_mug, sapien)
                self._phase = "cover_open"
                return target, -1.0, "cover_place"

        if self._phase == "cover_open":
            self._phase = "sphere_reach"
            self._object_to_tcp = None
            return None, 1.0, "cover_open"

        if self._phase == "sphere_reach":
            # The sphere is dynamic and can roll slightly when the cover is
            # parked. Recompute from its live pose at every chunk instead of
            # pursuing a stale grasp target.
            self._sphere_grasp_pose = self._find_grasp_pose(
                env, base.sphere, attempt=self._sphere_attempts
            )
            reach = self._sphere_grasp_pose * sapien.Pose([0, 0, -0.04])
            if self._at_pose(base.agent.tcp.pose.sp, reach):
                self._phase = "sphere_grasp"
            else:
                return reach, 1.0, "sphere_reach"

        if self._phase == "sphere_grasp":
            self._sphere_grasp_pose = self._find_grasp_pose(
                env, base.sphere, attempt=self._sphere_attempts
            )
            if not self._at_pose(base.agent.tcp.pose.sp, self._sphere_grasp_pose):
                return self._sphere_grasp_pose, 1.0, "sphere_grasp"
            self._phase = "sphere_close"
            self._sphere_close_chunks = 0
            self._sphere_stable_close_chunks = 0

        if self._phase == "sphere_close":
            self._sphere_close_chunks += 1
            grasped = bool(np.asarray(base.agent.is_grasping(base.sphere)).reshape(-1)[0])
            if grasped:
                self._sphere_stable_close_chunks += 1
            else:
                self._sphere_stable_close_chunks = 0
            if self._sphere_stable_close_chunks >= 2:
                self._phase = "sphere_lift"
            elif self._sphere_close_chunks > 3:
                self._sphere_attempts += 1
                self._phase = "sphere_reach"
                self._sphere_grasp_pose = None
                self._object_to_tcp = None
                self._sphere_close_chunks = 0
                self._sphere_stable_close_chunks = 0
                return self._target(env)
            else:
                return None, -1.0, "sphere_close"

        if self._phase in {"sphere_lift", "sphere_move", "sphere_place"}:
            if not bool(np.asarray(base.agent.is_grasping(base.sphere)).reshape(-1)[0]):
                self._sphere_attempts += 1
                self._phase = "sphere_reach"
                self._sphere_grasp_pose = None
                self._object_to_tcp = None
                return self._target(env)
            sphere_pose = self._pose_from_actor(base.sphere, sapien)
            if self._phase == "sphere_lift":
                target_sphere = sapien.Pose(
                    p=[sphere_pose.p[0], sphere_pose.p[1], 0.18], q=sphere_pose.q
                )
                target = self._object_target_tcp(base, base.sphere, target_sphere, sapien)
                if _first_vector(base.sphere.pose.p, 3)[2] >= 0.16:
                    self._phase = "sphere_move"
                else:
                    return target, -1.0, "sphere_lift"
            if self._phase == "sphere_move":
                bowl_p = _first_vector(base.bowl.pose.p, 3)
                target_sphere = sapien.Pose(
                    p=[bowl_p[0], bowl_p[1], 0.18], q=sphere_pose.q
                )
                target = self._object_target_tcp(base, base.sphere, target_sphere, sapien)
                if np.linalg.norm(_first_vector(base.sphere.pose.p, 3)[:2] - bowl_p[:2]) < 0.04:
                    self._phase = "sphere_place"
                else:
                    return target, -1.0, "sphere_move"
            if self._phase == "sphere_place":
                bowl_p = _first_vector(base.bowl.pose.p, 3)
                target_sphere = sapien.Pose(
                    p=[bowl_p[0], bowl_p[1], TABLE_Z + SPHERE_RADIUS], q=sphere_pose.q
                )
                target = self._object_target_tcp(base, base.sphere, target_sphere, sapien)
                # Keep the gripper closed until the live sphere pose reaches
                # the bowl.  Switching phases when the path is merely
                # planned releases the sphere several chunks too early.
                if np.linalg.norm(
                    _first_vector(base.sphere.pose.p, 3) - _first_vector(target_sphere.p, 3)
                ) < 0.035:
                    self._phase = "sphere_open"
                else:
                    return target, -1.0, "sphere_place"

        if self._phase == "sphere_open":
            self._phase = "done"
            return None, 1.0, "sphere_open"

        hold = np.zeros((self.chunk_size, 8), dtype=np.float32)
        hold[:, -1] = 1.0
        return None, 1.0, "done"

    def plan(self, env: Any) -> UncoverSpherePlaceOraclePlan:
        base = env.unwrapped
        if str(base.control_mode) != "pd_joint_delta_pos":
            raise ValueError("UncoverSpherePlace oracle requires pd_joint_delta_pos")
        target, gripper, phase = self._target(env)
        if target is None:
            actions = np.zeros((self.chunk_size, 8), dtype=np.float32)
            actions[:, -1] = gripper
            return UncoverSpherePlaceOraclePlan(actions, phase, True, gripper=gripper)
        path = self._plan_path(env, target)
        if path is None or len(path) == 0:
            actions = np.zeros((self.chunk_size, 8), dtype=np.float32)
            actions[:, -1] = gripper
            return UncoverSpherePlaceOraclePlan(actions, phase, False, gripper=gripper)
        # Preserve fine-grained waypoints for approach and grasp phases, where
        # contact precision matters. Once an object is held, cover the full
        # transport path in one chunk; replaying only its prefix can make a
        # long transport take hundreds of replanning chunks.
        transport_phase = phase in {
            "cover_lift",
            "cover_move",
            "cover_place",
            "sphere_lift",
            "sphere_move",
            "sphere_place",
        }
        if transport_phase:
            indices = np.linspace(0, len(path) - 1, num=self.chunk_size, dtype=np.int64)
        else:
            indices = np.minimum(np.arange(self.chunk_size), len(path) - 1)
        targets = np.asarray([path[index, :7] for index in indices], dtype=np.float32)
        actions = np.zeros((self.chunk_size, 8), dtype=np.float32)
        actions[:, -1] = gripper
        return UncoverSpherePlaceOraclePlan(
            actions, phase, True, joint_targets=targets, gripper=gripper
        )
