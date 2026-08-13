#!/usr/bin/env python3
"""Validate the privileged OpenDrawerRetrievePlace oracle on fixed seeds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

RLINF_ROOT = Path(__file__).resolve().parents[2]
if str(RLINF_ROOT) not in sys.path:
    sys.path.insert(0, str(RLINF_ROOT))


def _scalar(value: Any) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value).reshape(-1)[0])


def _vector(value: Any, length: int | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    return result if length is None else result[:length]


def _top_down_grasp(base, center: np.ndarray, closing: np.ndarray):
    closing = np.asarray(closing, dtype=np.float64)
    closing[2] = 0.0
    closing /= max(float(np.linalg.norm(closing)), 1e-8)
    return base.agent.build_grasp_pose(
        approaching=np.array([0.0, 0.0, -1.0]),
        closing=closing,
        center=np.asarray(center, dtype=np.float64),
    )


def _move_to_pose(
    env,
    planner,
    target_world,
    *,
    gripper: float,
    position_tolerance: float = 0.012,
) -> tuple[bool, int]:
    base = env.unwrapped
    planning_pose = planner._transform_pose_for_planning(target_world)
    target = np.concatenate(
        [_vector(planning_pose.p, 3), _vector(planning_pose.q, 4)]
    )
    qpos = _vector(base.agent.robot.get_qpos(), 9)
    result = planner.planner.plan_screw(
        target,
        qpos,
        time_step=float(base.control_timestep),
        use_point_cloud=False,
    )
    if result.get("status") != "Success":
        result = planner.planner.plan_qpos_to_pose(
            target,
            qpos,
            time_step=float(base.control_timestep),
            wrt_world=True,
        )
    if result.get("status") != "Success":
        return False, 0
    positions = np.asarray(result["position"], dtype=np.float32)
    for position in positions:
        env.step(np.concatenate([position[:7], [gripper]]).astype(np.float32))
    position_error = np.linalg.norm(
        _vector(base.agent.tcp.pose.p, 3) - np.asarray(target_world.p)
    )
    return position_error <= position_tolerance, len(positions)


def _hold_gripper(env, *, gripper: float, steps: int) -> None:
    base = env.unwrapped
    for _ in range(steps):
        qpos = _vector(base.agent.robot.get_qpos(), 9)
        env.step(np.concatenate([qpos[:7], [gripper]]).astype(np.float32))


def _close_until_object_grasped(env, base, *, max_steps: int = 24, stable_steps: int = 3):
    consecutive = 0
    for step in range(1, max_steps + 1):
        _hold_gripper(env, gripper=-1.0, steps=1)
        if _scalar(base.agent.is_grasping(base.obj)):
            consecutive += 1
            if consecutive >= stable_steps:
                return True, step
        else:
            consecutive = 0
    return False, max_steps


def solve_episode(env, seed: int) -> dict[str, Any]:
    import gymnasium as gym
    import sapien
    from mani_skill.examples.motionplanning.panda.motionplanner import (
        PandaArmMotionPlanningSolver,
    )

    env.reset(seed=seed)
    base = env.unwrapped
    proxy_env = gym.make(
        "PickCube-v1",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode=None,
        sim_backend="cpu",
    )
    proxy_env.reset(seed=0)
    planner = PandaArmMotionPlanningSolver(
        proxy_env,
        debug=False,
        vis=False,
        base_pose=base.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )
    stages: dict[str, Any] = {"seed": int(seed), "split": base.rlinf_split}
    handle_center = _vector(base.handle_world_position, 3)
    handle_grasp = _top_down_grasp(base, handle_center, np.array([1.0, 0.0, 0.0]))
    reached, steps = _move_to_pose(
        env,
        planner,
        handle_grasp * sapien.Pose([0.0, 0.0, -0.075]),
        gripper=1.0,
    )
    stages["reached_handle_pregrasp"] = reached
    stages["handle_pregrasp_steps"] = steps
    reached, steps = _move_to_pose(env, planner, handle_grasp, gripper=1.0)
    stages["reached_handle"] = stages["reached_handle_pregrasp"] and reached
    stages["handle_reach_steps"] = steps
    if stages["reached_handle"]:
        _hold_gripper(env, gripper=-1.0, steps=14)
        tcp = base.agent.tcp.pose.sp
        pull_target = sapien.Pose(tcp.p + np.array([-0.19, 0.0, 0.0]), tcp.q)
        moved, steps = _move_to_pose(
            env,
            planner,
            pull_target,
            gripper=-1.0,
            position_tolerance=0.018,
        )
        stages["pull_motion_completed"] = moved
        stages["pull_steps"] = steps
    else:
        stages["pull_motion_completed"] = False
        stages["pull_steps"] = 0
    stages["drawer_qpos_after_pull"] = float(_vector(base.drawer.get_qpos(), 1)[0])
    stages["drawer_opened"] = stages["drawer_qpos_after_pull"] <= -0.16
    _hold_gripper(env, gripper=1.0, steps=10)

    object_matrix = base.obj.pose.to_transformation_matrix()[0].cpu().numpy()
    object_center = object_matrix[:3, 3]
    object_closing = object_matrix[:3, 1]
    object_grasp = _top_down_grasp(base, object_center, object_closing)
    reached, steps = _move_to_pose(
        env,
        planner,
        object_grasp * sapien.Pose([0.0, 0.0, -0.075]),
        gripper=1.0,
    )
    stages["reached_object_pregrasp"] = reached
    stages["object_pregrasp_steps"] = steps
    reached, steps = _move_to_pose(env, planner, object_grasp, gripper=1.0)
    stages["reached_object"] = stages["reached_object_pregrasp"] and reached
    stages["object_reach_steps"] = steps
    if stages["reached_object"]:
        grasped, close_steps = _close_until_object_grasped(env, base)
    else:
        grasped, close_steps = False, 0
    stages["object_grasped"] = grasped
    stages["object_close_steps"] = close_steps

    initial_object_z = float(_vector(base.obj.pose.p, 3)[2])
    stages["initial_object_z"] = initial_object_z
    if grasped:
        object_in_tcp = base.agent.tcp.pose.sp.inv() * base.obj.pose.sp
        held = base.obj.pose.sp
        lifted_object = sapien.Pose(held.p + np.array([0.0, 0.0, 0.13]), held.q)
        moved, steps = _move_to_pose(
            env, planner, lifted_object * object_in_tcp.inv(), gripper=-1.0
        )
        stages["lift_motion_completed"] = moved
        stages["lift_steps"] = steps
    else:
        stages["lift_motion_completed"] = False
        stages["lift_steps"] = 0
    lifted_z = float(_vector(base.obj.pose.p, 3)[2])
    stages["lifted_object_z"] = lifted_z
    stages["object_lifted"] = (
        stages["lift_motion_completed"]
        and _scalar(base.agent.is_grasping(base.obj))
        and lifted_z - initial_object_z >= 0.08
    )

    if stages["object_lifted"]:
        object_in_tcp = base.agent.tcp.pose.sp.inv() * base.obj.pose.sp
        target_xy = _vector(base.target_tray.pose.p, 3)[:2]
        current_object = base.obj.pose.sp
        above_target = sapien.Pose([target_xy[0], target_xy[1], 0.15], current_object.q)
        moved, steps = _move_to_pose(
            env, planner, above_target * object_in_tcp.inv(), gripper=-1.0
        )
        stages["transport_completed"] = moved
        stages["transport_steps"] = steps
        place_target = sapien.Pose([target_xy[0], target_xy[1], 0.043], base.obj.pose.sp.q)
        moved, steps = _move_to_pose(
            env, planner, place_target * object_in_tcp.inv(), gripper=-1.0
        )
        stages["place_motion_completed"] = stages["transport_completed"] and moved
        stages["place_steps"] = steps
    else:
        stages["transport_completed"] = False
        stages["transport_steps"] = 0
        stages["place_motion_completed"] = False
        stages["place_steps"] = 0

    _hold_gripper(env, gripper=1.0, steps=14)
    tcp = base.agent.tcp.pose.sp
    moved, steps = _move_to_pose(
        env,
        planner,
        sapien.Pose(tcp.p + np.array([0.0, 0.0, 0.08]), tcp.q),
        gripper=1.0,
    )
    stages["retreat_completed"] = moved
    stages["retreat_steps"] = steps
    _hold_gripper(env, gripper=1.0, steps=8)
    evaluation = base.evaluate()
    for name in (
        "success",
        "ever_drawer_opened",
        "ever_grasped",
        "ever_lifted",
        "object_in_target",
        "object_released",
        "is_robot_static",
    ):
        stages[name] = _scalar(evaluation[name])
    stages["final_object_position"] = _vector(base.obj.pose.p, 3).tolist()
    stages["target_position"] = _vector(base.target_tray.pose.p, 3).tolist()
    planner.close()
    proxy_env.close()
    return stages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=("id", "handle_ood", "grasp_ood", "goal_ood", "all"),
        default="all",
    )
    parser.add_argument("--start-seed", type=int, default=70000)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-video", action="store_true")
    return parser.parse_args()


def main() -> None:
    import gymnasium as gym
    from mani_skill.utils.wrappers.record import RecordEpisode

    import rlinf.envs.maniskill.open_drawer_retrieve_place  # noqa: F401
    from rlinf.envs.maniskill.open_drawer_retrieve_place_spec import ENV_IDS

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    splits = tuple(ENV_IDS) if args.split == "all" else (args.split,)
    combined: dict[str, Any] = {}
    for split_index, split in enumerate(splits):
        split_dir = args.output_dir / split
        split_dir.mkdir()
        env = gym.make(
            ENV_IDS[split],
            obs_mode="none",
            control_mode="pd_joint_pos",
            render_mode="rgb_array",
            sim_backend="cpu",
        )
        env = RecordEpisode(
            env,
            output_dir=str(split_dir / "videos"),
            save_video=args.save_video,
            save_trajectory=False,
            save_on_reset=False,
            video_fps=30,
        )
        records = []
        try:
            for offset in range(args.num_seeds):
                seed = args.start_seed + split_index * 10000 + offset
                record = solve_episode(env, seed)
                records.append(record)
                with (split_dir / "episodes.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                if args.save_video:
                    env.flush_video(save=True)
        finally:
            env.close()
        successes = sum(bool(record["success"]) for record in records)
        summary = {
            "split": split,
            "attempts": len(records),
            "successes": successes,
            "success_rate": successes / len(records),
            "all_stage_rates": {
                name: float(np.mean([bool(record[name]) for record in records]))
                for name in (
                    "drawer_opened",
                    "object_grasped",
                    "object_lifted",
                    "transport_completed",
                    "place_motion_completed",
                    "object_in_target",
                    "success",
                )
            },
        }
        (split_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        combined[split] = summary
    (args.output_dir / "summary.json").write_text(
        json.dumps(combined, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
