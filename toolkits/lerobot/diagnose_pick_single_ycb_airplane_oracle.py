#!/usr/bin/env python3
"""Validate a physically grounded top-down oracle grasp for the YCB airplane.

This is intentionally separate from dataset collection: a candidate must keep
the object grasped through a real lift before it may be used as an expert.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RLINF_ROOT = Path(__file__).resolve().parents[2]
if str(RLINF_ROOT) not in sys.path:
    sys.path.insert(0, str(RLINF_ROOT))

from rlinf.envs.maniskill.pick_single_ycb_airplane_variants import (
    PICK_SINGLE_YCB_AIRPLANE_ID_ENV_ID,
    PICK_SINGLE_YCB_AIRPLANE_OOD_ENV_ID,
    register_controlled_pick_single_ycb_airplane_variants,
)


# The main wing is near local y=0 and the tail is near y=-0.10 m.  These
# points span the narrow fuselage between them; they are not wing grasps.
NECK_GRASP_CANDIDATES = (
    ("neck_y_minus_030", np.array([0.0, -0.030, 0.0], dtype=np.float64)),
    ("neck_y_minus_040", np.array([0.0, -0.040, 0.0], dtype=np.float64)),
    ("neck_y_minus_050", np.array([0.0, -0.050, 0.0], dtype=np.float64)),
    ("neck_y_minus_060", np.array([0.0, -0.060, 0.0], dtype=np.float64)),
)

# A local refinement around the first viable neck point.  The small vertical
# offsets change where the fingers meet the fuselage; they do not move toward
# either wing.
NECK_REFINEMENT_CANDIDATES = (
    # Collision-mesh component 7 is the narrow fuselage neck.  Its local
    # bounds are approximately x=[-0.046, 0.019], y=[-0.077, -0.014],
    # z=[0.007, 0.053], so the grasp centre must sit above and slightly left
    # of the object origin.  The older x=0,z=0 candidates touched the lower
    # edge and could report a transient grasp before slipping during lift.
    ("neck_center_x_minus_014_y_minus_046_z_plus_030", np.array([-0.014, -0.046, 0.030], dtype=np.float64)),
    ("neck_center_x_minus_014_y_minus_050_z_plus_030", np.array([-0.014, -0.050, 0.030], dtype=np.float64)),
    ("neck_center_x_minus_014_y_minus_042_z_plus_030", np.array([-0.014, -0.042, 0.030], dtype=np.float64)),
    ("neck_center_x_minus_020_y_minus_046_z_plus_026", np.array([-0.020, -0.046, 0.026], dtype=np.float64)),
    ("neck_center_x_minus_008_y_minus_046_z_plus_026", np.array([-0.008, -0.046, 0.026], dtype=np.float64)),
    ("neck_y_minus_046_z_minus_010", np.array([0.0, -0.046, -0.010], dtype=np.float64)),
    ("neck_y_minus_046_z_zero", np.array([0.0, -0.046, 0.0], dtype=np.float64)),
    ("neck_y_minus_050_z_minus_010", np.array([0.0, -0.050, -0.010], dtype=np.float64)),
    ("neck_y_minus_050_z_zero", np.array([0.0, -0.050, 0.0], dtype=np.float64)),
    ("neck_y_minus_054_z_minus_010", np.array([0.0, -0.054, -0.010], dtype=np.float64)),
    ("neck_y_minus_054_z_zero", np.array([0.0, -0.054, 0.0], dtype=np.float64)),
    ("neck_y_minus_046_z_zero_flip", np.array([0.0, -0.046, 0.0], dtype=np.float64)),
    ("neck_y_minus_042_z_zero", np.array([0.0, -0.042, 0.0], dtype=np.float64)),
    ("neck_y_minus_042_z_zero_flip", np.array([0.0, -0.042, 0.0], dtype=np.float64)),
    ("neck_y_minus_038_z_zero", np.array([0.0, -0.038, 0.0], dtype=np.float64)),
    ("neck_y_minus_038_z_zero_flip", np.array([0.0, -0.038, 0.0], dtype=np.float64)),
)

# Fixed order for the task oracle.  Every option remains in the same narrow
# fuselage region.  A failed attempt is reset to the identical seeded state,
# so it never contaminates the accepted expert trajectory.
ORACLE_NECK_CANDIDATES = (
    # Keep the empirically strongest nominal pose first.  The mesh-centred
    # poses below are geometric fallbacks, not replacements for the validated
    # contact point.
    NECK_REFINEMENT_CANDIDATES[6],
    *NECK_REFINEMENT_CANDIDATES[:5],
    NECK_REFINEMENT_CANDIDATES[11],
    NECK_REFINEMENT_CANDIDATES[12],
    NECK_REFINEMENT_CANDIDATES[13],
    NECK_REFINEMENT_CANDIDATES[14],
    NECK_REFINEMENT_CANDIDATES[15],
    NECK_REFINEMENT_CANDIDATES[9],
    NECK_REFINEMENT_CANDIDATES[5],
    NECK_REFINEMENT_CANDIDATES[7],
)


def _scalar(value) -> bool:
    array = np.asarray(value)
    return bool(array.reshape(-1)[0])


def _pose_result_ok(result) -> bool:
    return result != -1


def _move_with_planning_fallback(planner, pose) -> bool:
    """Prefer a short Cartesian screw path, then use official RRTConnect."""

    if _pose_result_ok(planner.move_to_pose_with_screw(pose)):
        return True
    return _pose_result_ok(planner.move_to_pose_with_RRTConnect(pose))


def _is_grasping(unwrapped) -> bool:
    """Use the ManiSkill Panda grasp predicate across supported versions."""

    predicate = getattr(unwrapped.agent, "is_grasping", None)
    if predicate is None:
        predicate = getattr(unwrapped.agent, "is_grasped", None)
    if predicate is None:
        raise AttributeError("Panda agent exposes neither is_grasping nor is_grasped")
    return _scalar(predicate(unwrapped.obj))


def _close_gripper_until_stable_grasp(
    planner,
    unwrapped,
    *,
    max_steps: int,
    stable_steps: int,
) -> tuple[bool, int]:
    """Close only until the grasp predicate remains true for a short window."""

    if max_steps < 1:
        raise ValueError("max_steps must be at least one")
    if stable_steps < 1 or stable_steps > max_steps:
        raise ValueError("stable_steps must be in [1, max_steps]")
    consecutive_grasp_steps = 0
    for executed_steps in range(1, max_steps + 1):
        planner.close_gripper(t=1)
        if _is_grasping(unwrapped):
            consecutive_grasp_steps += 1
            if consecutive_grasp_steps >= stable_steps:
                return True, executed_steps
        else:
            consecutive_grasp_steps = 0
    return False, max_steps


def _build_top_down_neck_pose(unwrapped, local_point: np.ndarray, *, closing_sign: float = 1.0):
    """Build a top-down grasp centred on the narrow fuselage, not a wing."""

    import sapien

    object_matrix = unwrapped.obj.pose.to_transformation_matrix()[0].cpu().numpy()
    world_point = object_matrix[:3, :3] @ local_point + object_matrix[:3, 3]
    approach = np.array([0.0, 0.0, -1.0])
    closing = object_matrix[:3, :3] @ np.array([closing_sign, 0.0, 0.0])
    # A policy may leave the airplane tilted before asking for help.  Preserve
    # a top-down approach while projecting the fuselage-relative closing axis
    # into its orthogonal plane, as required by Panda.build_grasp_pose.
    closing = closing - approach * float(approach @ closing)
    norm = float(np.linalg.norm(closing))
    closing = closing / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
    return unwrapped.agent.build_grasp_pose(approach, closing, world_point)


def try_candidate(
    env,
    *,
    seed: int,
    name: str,
    local_point: np.ndarray,
    close_steps: int,
    complete_task: bool,
    reset_before_attempt: bool = True,
    force_planner_pd_joint_pos: bool = False,
    closing_sign: float = 1.0,
    stable_grasp_steps: int = 4,
) -> dict[str, object]:
    """Run contact, lift, and optionally transport to the official goal."""

    import sapien
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver

    if reset_before_attempt:
        env.reset(seed=seed)
    unwrapped = env.unwrapped
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )
    if force_planner_pd_joint_pos:
        # A caller may adapt each absolute joint target into another control
        # mode inside env.step.  Keep planner outputs in the standard 8-D
        # [joint_target, gripper] form in that case.
        planner.control_mode = "pd_joint_pos"
    try:
        grasp_pose = _build_top_down_neck_pose(unwrapped, local_point, closing_sign=closing_sign)
        initial_z = float(unwrapped.obj.pose.p[0, 2].cpu())
        reached_pregrasp = _move_with_planning_fallback(planner, grasp_pose * sapien.Pose([0.0, 0.0, -0.065]))
        reached_grasp = reached_pregrasp and _move_with_planning_fallback(planner, grasp_pose)
        grasped_after_close = False
        close_executed_steps = 0
        if reached_grasp:
            grasped_after_close, close_executed_steps = _close_gripper_until_stable_grasp(
                planner,
                unwrapped,
                max_steps=close_steps,
                stable_steps=stable_grasp_steps,
            )
        lifted = False
        moved_to_goal = False
        success = False
        goal_distance = float("inf")
        final_z = float(unwrapped.obj.pose.p[0, 2].cpu())
        if grasped_after_close:
            object_in_tcp = unwrapped.agent.tcp.pose.sp.inv() * unwrapped.obj.pose.sp
            held_object = unwrapped.obj.pose.sp
            lifted_object = sapien.Pose(held_object.p + np.array([0.0, 0.0, 0.12]), held_object.q)
            lifted_tcp = lifted_object * object_in_tcp.inv()
            lifted = _move_with_planning_fallback(planner, lifted_tcp)
            final_z = float(unwrapped.obj.pose.p[0, 2].cpu())
        still_grasped = _is_grasping(unwrapped) if grasped_after_close else False
        accepted_lift = bool(lifted and still_grasped and final_z - initial_z >= 0.06)
        if complete_task and accepted_lift:
            # Keep the grasp transform that physics produced, rather than
            # assuming the object remains perfectly at the nominal TCP pose.
            object_in_tcp = unwrapped.agent.tcp.pose.sp.inv() * unwrapped.obj.pose.sp
            target_object = sapien.Pose(unwrapped.goal_site.pose.sp.p, unwrapped.obj.pose.sp.q)
            moved_to_goal = _move_with_planning_fallback(planner, target_object * object_in_tcp.inv())
            if moved_to_goal:
                planner.close_gripper(t=20)
            evaluation = unwrapped.evaluate()
            success = _scalar(evaluation["success"])
            goal_distance = float(np.linalg.norm(unwrapped.goal_site.pose.p[0].cpu().numpy() - unwrapped.obj.pose.p[0].cpu().numpy()))
        return {
            "seed": seed,
            "candidate": name,
            "local_point": local_point.tolist(),
            "close_max_steps": close_steps,
            "close_executed_steps": close_executed_steps,
            "stable_grasp_steps": stable_grasp_steps,
            "reached_pregrasp": reached_pregrasp,
            "reached_grasp": reached_grasp,
            "grasped_after_close": grasped_after_close,
            "lift_command_completed": lifted,
            "still_grasped_after_lift": still_grasped,
            "initial_z": initial_z,
            "final_z": final_z,
            "lift_delta_z": final_z - initial_z,
            "accepted_lift": accepted_lift,
            "moved_to_goal": moved_to_goal,
            "goal_distance": goal_distance,
            "success": success,
            "accepted": success if complete_task else accepted_lift,
        }
    finally:
        planner.close()


def run_oracle_with_fallback(env, *, seed: int, close_steps: int, complete_task: bool) -> dict[str, object]:
    """Retry only deterministic narrow-neck poses from the identical reset."""

    attempts: list[dict[str, object]] = []
    for name, local_point in ORACLE_NECK_CANDIDATES:
        attempt = try_candidate(
            env,
            seed=seed,
            name=name,
            local_point=local_point,
            close_steps=close_steps,
            complete_task=complete_task,
            closing_sign=-1.0 if name.endswith("_flip") else 1.0,
        )
        attempts.append(attempt)
        if bool(attempt["accepted"]):
            return {
                "seed": seed,
                "accepted": True,
                "selected_candidate": name,
                "attempts": attempts,
            }
    return {"seed": seed, "accepted": False, "selected_candidate": None, "attempts": attempts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--split", choices=("id", "ood"), default="id")
    parser.add_argument("--profile", choices=("baseline", "refinement"), default="baseline")
    parser.add_argument("--candidate-name", default=None)
    parser.add_argument("--close-steps", type=int, default=45)
    parser.add_argument("--complete-task", action="store_true")
    parser.add_argument("--oracle-fallback", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    register_controlled_pick_single_ycb_airplane_variants()
    env = gym.make(
        PICK_SINGLE_YCB_AIRPLANE_ID_ENV_ID if args.split == "id" else PICK_SINGLE_YCB_AIRPLANE_OOD_ENV_ID,
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        reward_mode="sparse",
        sim_backend="physx_cpu",
        sim_config={"sim_freq": 100, "control_freq": 10},
        max_episode_steps=80,
    )
    try:
        if args.oracle_fallback:
            rows = [{"split": args.split, **run_oracle_with_fallback(
                env, seed=args.seed, close_steps=args.close_steps, complete_task=args.complete_task
            )}]
        else:
            candidates = NECK_GRASP_CANDIDATES if args.profile == "baseline" else NECK_REFINEMENT_CANDIDATES
            if args.candidate_name is not None:
                candidates = tuple(candidate for candidate in candidates if candidate[0] == args.candidate_name)
                if not candidates:
                    raise ValueError(f"candidate {args.candidate_name!r} is not part of profile {args.profile!r}")
            rows = [
                {
                    "split": args.split,
                    **try_candidate(
                        env,
                        seed=args.seed,
                        name=name,
                        local_point=point,
                        close_steps=args.close_steps,
                        complete_task=args.complete_task,
                        closing_sign=-1.0 if name.endswith("_flip") else 1.0,
                    ),
                }
                for name, point in candidates
            ]
    finally:
        env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
