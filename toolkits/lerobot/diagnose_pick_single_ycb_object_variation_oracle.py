#!/usr/bin/env python3
"""Run the object-variation PickSingleYCB oracle and write evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RLINF_ROOT = Path(__file__).resolve().parents[2]
if str(RLINF_ROOT) not in sys.path:
    sys.path.insert(0, str(RLINF_ROOT))

from rlinf.envs.maniskill.pick_single_ycb_object_variation import (
    PICK_SINGLE_YCB_OBJECT_ID_ENV_ID,
    PICK_SINGLE_YCB_OBJECT_OOD_ENV_ID,
    register_controlled_pick_single_ycb_object_variants,
    reset_metadata,
)


def _scalar(value: object) -> bool:
    return bool(np.asarray(value).reshape(-1)[0])


def _move(planner, pose) -> bool:
    result = planner.move_to_pose_with_screw(pose)
    if result != -1:
        return True
    return planner.move_to_pose_with_RRTConnect(pose) != -1


def run_oracle(env, *, seed: int) -> dict[str, object]:
    import sapien
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver

    env.reset(seed=seed)
    base = env.unwrapped
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=base.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )
    try:
        approaching = np.array([0.0, 0.0, -1.0])
        target_closing = base.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        # The installed ManiSkill wheel includes the official Panda planner but
        # not the optional example OBB helper. Both selected YCB objects are
        # fixed upright variants, so a top-down grasp with the planner's TCP
        # closing axis is the deterministic equivalent for this task.
        closing = target_closing - (approaching @ target_closing) * approaching
        closing_norm = float(np.linalg.norm(closing))
        closing = closing / closing_norm if closing_norm > 1e-6 else np.array([1.0, 0.0, 0.0])
        grasp_pose = base.agent.build_grasp_pose(approaching, closing, base.obj.pose.sp.p)
        reached_pregrasp = _move(planner, grasp_pose * sapien.Pose([0.0, 0.0, -0.06]))
        reached_grasp = reached_pregrasp and _move(planner, grasp_pose)
        grasp_steps = 0
        stable_grasp = False
        consecutive = 0
        for step in range(1, 41):
            planner.close_gripper(t=1)
            grasp_steps = step
            if _scalar(base.agent.is_grasping(base.obj)):
                consecutive += 1
                if consecutive >= 4:
                    stable_grasp = True
                    break
            else:
                consecutive = 0

        initial_z = float(base.obj.pose.p[0, 2].cpu())
        lifted = False
        if stable_grasp:
            object_in_tcp = base.agent.tcp.pose.sp.inv() * base.obj.pose.sp
            held = base.obj.pose.sp
            lift_object = sapien.Pose(held.p + np.array([0.0, 0.0, 0.10]), held.q)
            lift_tcp = lift_object * object_in_tcp.inv()
            lifted = _move(planner, lift_tcp)

        still_grasped = _scalar(base.agent.is_grasping(base.obj)) if stable_grasp else False
        final_z = float(base.obj.pose.p[0, 2].cpu())
        accepted_lift = bool(lifted and still_grasped and final_z - initial_z >= 0.05)
        placed = False
        if accepted_lift:
            object_in_tcp = base.agent.tcp.pose.sp.inv() * base.obj.pose.sp
            target = sapien.Pose(base.goal_site.pose.sp.p, base.obj.pose.sp.q)
            placed = _move(planner, target * object_in_tcp.inv())
            if placed:
                planner.close_gripper(t=15)

        evaluation = base.evaluate()
        success = _scalar(evaluation["success"])
        metadata = reset_metadata(base, split=base.rlinf_split)
        return {
            "seed": seed,
            "split": base.rlinf_split,
            "accepted": success,
            "success": success,
            "reached_pregrasp": reached_pregrasp,
            "reached_grasp": reached_grasp,
            "stable_grasp": stable_grasp,
            "grasp_steps": grasp_steps,
            "lift_command_completed": lifted,
            "still_grasped_after_lift": still_grasped,
            "lift_delta_z": final_z - initial_z,
            "placed": placed,
            "reset_metadata": metadata,
        }
    finally:
        planner.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("id", "ood"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    register_controlled_pick_single_ycb_object_variants()
    env_id = PICK_SINGLE_YCB_OBJECT_ID_ENV_ID if args.split == "id" else PICK_SINGLE_YCB_OBJECT_OOD_ENV_ID
    env = gym.make(
        env_id,
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        reward_mode="sparse",
        sim_backend="physx_cpu",
        sim_config={"sim_freq": 100, "control_freq": 10},
        render_mode="rgb_array",
        max_episode_steps=200,
    )
    try:
        row = run_oracle(env, seed=args.seed)
    finally:
        env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
