#!/usr/bin/env python3
"""Validate the privileged OpenDrawerRetrievePlace oracle on fixed seeds."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

RLINF_ROOT = Path(__file__).resolve().parents[2]
if str(RLINF_ROOT) not in sys.path:
    sys.path.insert(0, str(RLINF_ROOT))

from rlinf.envs.maniskill.open_drawer_retrieve_place_spec import (
    DRAWER_OPEN_THRESHOLD,
    ENV_IDS,
)


def _scalar(value: Any) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value).reshape(-1)[0])


def _vector(value: Any, length: int | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    return result if length is None else result[:length]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _top_down_grasp(base, center: np.ndarray, closing: np.ndarray):
    closing = np.asarray(closing, dtype=np.float64)
    closing[2] = 0.0
    closing /= max(float(np.linalg.norm(closing)), 1e-8)
    return base.agent.build_grasp_pose(
        approaching=np.array([0.0, 0.0, -1.0]),
        closing=closing,
        center=np.asarray(center, dtype=np.float64),
    )


class PandaPosePlannerClient:
    def __init__(self):
        server = Path(__file__).with_name("panda_pose_planner_server.py")
        planner_python = os.environ.get("PANDA_PLANNER_PYTHON", sys.executable)
        self.process = subprocess.Popen(
            [planner_python, "-u", str(server)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        if self.process.stdout is None or self.process.stdin is None:
            raise RuntimeError("failed to create planner pipes")
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("planner exited before becoming ready")
            if line.strip() == "READY":
                break

    def plan(self, target_world, qpos: np.ndarray, time_step: float) -> np.ndarray | None:
        assert self.process.stdin is not None and self.process.stdout is not None
        request = {
            "target_p": np.asarray(target_world.p).tolist(),
            "target_q": np.asarray(target_world.q).tolist(),
            "qpos": np.asarray(qpos).tolist(),
            "time_step": float(time_step),
        }
        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("planner exited while handling a request")
            if line.startswith("RESULT "):
                response = json.loads(line[len("RESULT ") :])
                if response["status"] != "Success":
                    return None
                return np.asarray(response["positions"], dtype=np.float32)

    def close(self) -> None:
        if self.process.poll() is None and self.process.stdin is not None:
            self.process.stdin.write('{"command":"close"}\n')
            self.process.stdin.flush()
        self.process.wait(timeout=30)


def _move_to_pose(
    env,
    planner,
    target_world,
    *,
    gripper: float,
    position_tolerance: float = 0.012,
) -> tuple[bool, int]:
    base = env.unwrapped
    qpos = _vector(base.agent.robot.get_qpos(), 9)
    positions = planner.plan(target_world, qpos, float(base.control_timestep))
    if positions is None:
        return False, 0
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


def solve_episode(env, seed: int, planner) -> dict[str, Any]:
    import sapien

    env.reset(seed=seed)
    base = env.unwrapped
    stages: dict[str, Any] = {"seed": int(seed), "split": base.rlinf_split}
    handle_center = _vector(base.handle_world_position, 3)
    handle_grasp = _top_down_grasp(base, handle_center, np.array([0.0, 1.0, 0.0]))
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
        pull_target = sapien.Pose(tcp.p + np.array([-0.37, 0.0, 0.0]), tcp.q)
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
    stages["drawer_opened"] = (
        stages["drawer_qpos_after_pull"] <= -DRAWER_OPEN_THRESHOLD
    )
    _hold_gripper(env, gripper=1.0, steps=10)
    tcp = base.agent.tcp.pose.sp
    moved, steps = _move_to_pose(
        env,
        planner,
        sapien.Pose(tcp.p + np.array([0.0, 0.0, 0.15]), tcp.q),
        gripper=1.0,
    )
    stages["handle_retreat_completed"] = moved
    stages["handle_retreat_steps"] = steps

    object_matrix = base.obj.pose.to_transformation_matrix()[0].cpu().numpy()
    object_center = object_matrix[:3, 3]
    object_closing = object_matrix[:3, 1]
    object_grasp = _top_down_grasp(base, object_center, object_closing)
    reached, steps = _move_to_pose(
        env,
        planner,
        object_grasp * sapien.Pose([0.0, 0.0, -0.140]),
        gripper=1.0,
        position_tolerance=0.025,
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
            env,
            planner,
            above_target * object_in_tcp.inv(),
            gripper=-1.0,
            position_tolerance=0.035,
        )
        stages["transport_completed"] = moved
        stages["transport_steps"] = steps
        place_target = sapien.Pose([target_xy[0], target_xy[1], 0.043], base.obj.pose.sp.q)
        moved, steps = _move_to_pose(
            env,
            planner,
            place_target * object_in_tcp.inv(),
            gripper=-1.0,
            position_tolerance=0.035,
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
    return stages


def continue_episode(env, planner, *, seed: int | None = None) -> dict[str, Any]:
    """Finish an episode from its current state after an expert takeover.

    This deliberately does not reset the environment.  The next action is
    selected from observable task predicates, so a takeover after drawer
    opening or after object grasping skips work that has already happened.
    """

    import sapien

    base = env.unwrapped
    stages: dict[str, Any] = {
        "seed": None if seed is None else int(seed),
        "split": base.rlinf_split,
        "takeover_from_current_state": True,
    }
    drawer_opened = _scalar(
        _vector(base.drawer.get_qpos(), 1)[0] <= DRAWER_OPEN_THRESHOLD * -1.0
    )
    stages["drawer_opened_before_takeover"] = drawer_opened

    if not drawer_opened:
        handle_center = _vector(base.handle_world_position, 3)
        handle_grasp = _top_down_grasp(base, handle_center, np.array([0.0, 1.0, 0.0]))
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
            pull_target = sapien.Pose(tcp.p + np.array([-0.37, 0.0, 0.0]), tcp.q)
            moved, steps = _move_to_pose(
                env, planner, pull_target, gripper=-1.0, position_tolerance=0.018
            )
            stages["pull_motion_completed"] = moved
            stages["pull_steps"] = steps
        else:
            stages["pull_motion_completed"] = False
            stages["pull_steps"] = 0

    drawer_opened = bool(
        _vector(base.drawer.get_qpos(), 1)[0] <= -DRAWER_OPEN_THRESHOLD
    )
    stages["drawer_opened_after_takeover"] = drawer_opened
    if not drawer_opened:
        stages["success"] = False
        return stages

    grasped = _scalar(base.agent.is_grasping(base.obj))
    stages["object_grasped_before_takeover"] = grasped
    if not grasped:
        _hold_gripper(env, gripper=1.0, steps=10)
        tcp = base.agent.tcp.pose.sp
        moved, steps = _move_to_pose(
            env,
            planner,
            sapien.Pose(tcp.p + np.array([0.0, 0.0, 0.15]), tcp.q),
            gripper=1.0,
        )
        stages["handle_retreat_completed"] = moved
        stages["handle_retreat_steps"] = steps

        object_matrix = base.obj.pose.to_transformation_matrix()[0].cpu().numpy()
        object_center = object_matrix[:3, 3]
        object_closing = object_matrix[:3, 1]
        object_grasp = _top_down_grasp(base, object_center, object_closing)
        reached, steps = _move_to_pose(
            env,
            planner,
            object_grasp * sapien.Pose([0.0, 0.0, -0.140]),
            gripper=1.0,
            position_tolerance=0.025,
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
        stages["object_close_steps"] = close_steps

    stages["object_grasped_after_takeover"] = bool(grasped)
    initial_object_z = float(_vector(base.obj.pose.p, 3)[2])
    already_lifted = grasped and initial_object_z >= 0.12
    if grasped and not already_lifted:
        object_in_tcp = base.agent.tcp.pose.sp.inv() * base.obj.pose.sp
        lifted_object = sapien.Pose(
            base.obj.pose.sp.p + np.array([0.0, 0.0, 0.13]), base.obj.pose.sp.q
        )
        moved, steps = _move_to_pose(
            env, planner, lifted_object * object_in_tcp.inv(), gripper=-1.0
        )
        stages["lift_motion_completed"] = moved
        stages["lift_steps"] = steps
    lifted = already_lifted or (
        _scalar(base.agent.is_grasping(base.obj))
        and float(_vector(base.obj.pose.p, 3)[2]) - initial_object_z >= 0.08
    )
    stages["object_lifted_after_takeover"] = lifted
    if not lifted:
        stages["success"] = False
        return stages

    object_in_tcp = base.agent.tcp.pose.sp.inv() * base.obj.pose.sp
    target_xy = _vector(base.target_tray.pose.p, 3)[:2]
    current_object = base.obj.pose.sp
    above_target = sapien.Pose([target_xy[0], target_xy[1], 0.15], current_object.q)
    moved, steps = _move_to_pose(
        env,
        planner,
        above_target * object_in_tcp.inv(),
        gripper=-1.0,
        position_tolerance=0.035,
    )
    stages["transport_completed"] = moved
    stages["transport_steps"] = steps
    place_target = sapien.Pose([target_xy[0], target_xy[1], 0.043], base.obj.pose.sp.q)
    moved, steps = _move_to_pose(
        env,
        planner,
        place_target * object_in_tcp.inv(),
        gripper=-1.0,
        position_tolerance=0.035,
    )
    stages["place_motion_completed"] = stages["transport_completed"] and moved
    stages["place_steps"] = steps
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
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    splits = tuple(ENV_IDS) if args.split == "all" else (args.split,)
    combined: dict[str, Any] = {}
    for split in splits:
        split_dir = args.output_dir / split
        split_dir.mkdir()
        planner = PandaPosePlannerClient()
        env = gym.make(
            ENV_IDS[split],
            obs_mode="none",
            control_mode="pd_joint_pos",
            render_mode="rgb_array" if args.save_video else None,
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
                seed = args.start_seed + offset
                record = _jsonable(solve_episode(env, seed, planner))
                records.append(record)
                with (split_dir / "episodes.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                if args.save_video:
                    env.flush_video(save=True)
        finally:
            env.close()
            planner.close()
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
