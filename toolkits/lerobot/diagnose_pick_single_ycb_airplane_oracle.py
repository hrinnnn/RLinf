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


def _scalar(value) -> bool:
    array = np.asarray(value)
    return bool(array.reshape(-1)[0])


def _pose_result_ok(result) -> bool:
    return result != -1


def _build_top_down_neck_pose(unwrapped, local_point: np.ndarray):
    """Build a top-down grasp centred on the narrow fuselage, not a wing."""

    import sapien

    object_matrix = unwrapped.obj.pose.to_transformation_matrix()[0].cpu().numpy()
    world_point = object_matrix[:3, :3] @ local_point + object_matrix[:3, 3]
    approach = np.array([0.0, 0.0, -1.0])
    closing = object_matrix[:3, :3] @ np.array([1.0, 0.0, 0.0])
    return unwrapped.agent.build_grasp_pose(approach, closing, world_point)


def try_candidate(env, *, seed: int, name: str, local_point: np.ndarray) -> dict[str, object]:
    """Run contact, close, and lift. A contact-only grasp is explicitly rejected."""

    import sapien
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver

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
    try:
        grasp_pose = _build_top_down_neck_pose(unwrapped, local_point)
        initial_z = float(unwrapped.obj.pose.p[0, 2].cpu())
        reached_pregrasp = _pose_result_ok(planner.move_to_pose_with_screw(grasp_pose * sapien.Pose([0.0, 0.0, -0.065])))
        reached_grasp = reached_pregrasp and _pose_result_ok(planner.move_to_pose_with_screw(grasp_pose))
        if reached_grasp:
            planner.close_gripper(t=30)
        grasped_after_close = reached_grasp and _scalar(unwrapped.agent.is_grasped(unwrapped.obj))
        lifted = False
        final_z = float(unwrapped.obj.pose.p[0, 2].cpu())
        if grasped_after_close:
            object_in_tcp = unwrapped.agent.tcp.pose.sp.inv() * unwrapped.obj.pose.sp
            held_object = unwrapped.obj.pose.sp
            lifted_object = sapien.Pose(held_object.p + np.array([0.0, 0.0, 0.12]), held_object.q)
            lifted_tcp = lifted_object * object_in_tcp.inv()
            lifted = _pose_result_ok(planner.move_to_pose_with_screw(lifted_tcp))
            final_z = float(unwrapped.obj.pose.p[0, 2].cpu())
        still_grasped = _scalar(unwrapped.agent.is_grasped(unwrapped.obj)) if grasped_after_close else False
        return {
            "seed": seed,
            "candidate": name,
            "local_point": local_point.tolist(),
            "reached_pregrasp": reached_pregrasp,
            "reached_grasp": reached_grasp,
            "grasped_after_close": grasped_after_close,
            "lift_command_completed": lifted,
            "still_grasped_after_lift": still_grasped,
            "initial_z": initial_z,
            "final_z": final_z,
            "lift_delta_z": final_z - initial_z,
            "accepted": bool(lifted and still_grasped and final_z - initial_z >= 0.06),
        }
    finally:
        planner.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    register_controlled_pick_single_ycb_airplane_variants()
    env = gym.make(
        PICK_SINGLE_YCB_AIRPLANE_ID_ENV_ID,
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        reward_mode="sparse",
        sim_backend="physx_cpu",
        sim_config={"sim_freq": 100, "control_freq": 10},
        max_episode_steps=80,
    )
    try:
        rows = [try_candidate(env, seed=args.seed, name=name, local_point=point) for name, point in NECK_GRASP_CANDIDATES]
    finally:
        env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
