#!/usr/bin/env python3
"""Collect fixed-airplane PickSingleYCB expert demonstrations for pi0.5.

The official Panda motion planner is run in absolute joint-position mode and
replayed from the identical reset using the pi0.5-compatible joint-delta
control.  Only successful replay trajectories are admitted to the dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

RLINF_ROOT = Path(__file__).resolve().parents[2]
if str(RLINF_ROOT) not in sys.path:
    sys.path.insert(0, str(RLINF_ROOT))

from rlinf.envs.maniskill.pick_single_ycb_airplane_variants import (
    PICK_SINGLE_YCB_AIRPLANE_ID_ENV_ID,
    PICK_SINGLE_YCB_AIRPLANE_OOD_ENV_ID,
    PICK_SINGLE_YCB_AIRPLANE_TASK,
    register_controlled_pick_single_ycb_airplane_variants,
    reset_metadata,
)
from toolkits.lerobot.collect_maniskill_peg_lerobot_joint import (
    MAIN_CAMERA_CANDIDATES,
    WRIST_CAMERA_CANDIDATES,
    _bool_scalar,
    _build_frames,
    _convert_solver_action_to_joint_delta,
    _create_dataset,
    _extract_record,
    _joint_delta_arm_bounds,
    _resolve_output_path,
    _select_camera,
    _to_numpy,
    _video_output_dir,
    _write_episode_video,
    validate_visual_motion,
)
from toolkits.lerobot.diagnose_pick_single_ycb_airplane_oracle import run_oracle_with_fallback

LOG = logging.getLogger("collect_maniskill_pick_single_ycb_airplane_lerobot")


def controlled_env_id(split: str) -> str:
    if split == "id":
        return PICK_SINGLE_YCB_AIRPLANE_ID_ENV_ID
    if split == "ood":
        return PICK_SINGLE_YCB_AIRPLANE_OOD_ENV_ID
    raise ValueError(f"split must be id or ood, got {split!r}")


def write_episode_video_durably(frames: list[dict[str, Any]], *, video_dir: Path, episode_index: int, seed: int, fps: int) -> Path:
    """Encode locally before copying a complete file to an OSSFS destination."""

    name = f"episode_{episode_index:06d}_seed_{seed:06d}.mp4"
    video_dir.mkdir(parents=True, exist_ok=True)
    destination = video_dir / name
    with tempfile.TemporaryDirectory(prefix="pick-airplane-video-") as temp:
        _write_episode_video(frames, video_dir=Path(temp), episode_index=episode_index, seed=seed, fps=fps)
        encoded = Path(temp) / name
        if not encoded.is_file() or encoded.stat().st_size == 0:
            raise RuntimeError("local video encoding did not produce a non-empty mp4")
        shutil.copy2(encoded, destination)
    return destination


def _build_env(args: argparse.Namespace, *, control_mode: str):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    register_controlled_pick_single_ycb_airplane_variants()
    return gym.make(
        controlled_env_id(args.split), num_envs=1, obs_mode="rgb", control_mode=control_mode,
        reward_mode="sparse", render_mode="rgb_array", sim_backend=args.sim_backend,
        sim_config={"sim_freq": 100, "control_freq": args.control_freq},
        sensor_configs={"width": args.image_size, "height": args.image_size}, max_episode_steps=args.max_episode_steps,
    )


def _solve(env: Any, *, seed: int) -> dict[str, Any] | None:
    """Run the validated top-down narrow-fuselage oracle to true task success."""

    result = run_oracle_with_fallback(env, seed=seed, close_steps=45, complete_task=True)
    return result if bool(result["accepted"]) else None


def _run_reference(env: Any, seed: int) -> tuple[list[Any], list[Any], dict[str, Any]] | None:
    records: list[Any] = []
    actions: list[Any] = []
    metadata: dict[str, Any] = {}
    original_reset, original_step = env.reset, env.step

    def reset_hook(*args, **kwargs):
        obs, info = original_reset(*args, **kwargs)
        records.clear(); actions.clear(); records.append(_extract_record(obs)); metadata.clear()
        metadata.update(reset_metadata(env, split=env.unwrapped.rlinf_split))
        return obs, info

    def step_hook(action, *args, **kwargs):
        actions.append(_to_numpy(action).astype("float32").reshape(-1))
        obs, reward, terminated, truncated, info = original_step(action, *args, **kwargs)
        records.append(_extract_record(obs))
        return obs, reward, terminated, truncated, info

    env.reset, env.step = reset_hook, step_hook  # type: ignore[method-assign]
    try:
        oracle = _solve(env, seed=seed)
        if oracle is None or len(records) != len(actions) + 1 or not actions:
            return None
        metadata["oracle"] = {
            "type": "top_down_airplane_narrow_fuselage",
            "selected_candidate": oracle["selected_candidate"],
            "attempt_count": len(oracle["attempts"]),
        }
        return records, actions, dict(metadata)
    finally:
        env.reset, env.step = original_reset, original_step  # type: ignore[method-assign]


def _replay(env: Any, seed: int, solver_actions: list[Any], lower: Any, upper: Any):
    obs, _info = env.reset(seed=seed)
    records, actions = [_extract_record(obs)], []
    success = False
    for solver_action in solver_actions:
        action = _convert_solver_action_to_joint_delta(records[-1].qpos, solver_action, lower, upper)
        obs, _reward, terminated, truncated, info = env.step(action)
        records.append(_extract_record(obs)); actions.append(action)
        success = _bool_scalar(info.get("success"))
        if success or _bool_scalar(terminated) or _bool_scalar(truncated):
            break
    return (records, actions) if success and len(records) == len(actions) + 1 else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("id", "ood"), default="id")
    parser.add_argument("--num-episodes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--control-freq", type=int, default=10)
    # The validated top-down grasp, lift, and goal transport uses roughly
    # 125 low-level joint commands.  Every split must share this horizon.
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--min-visual-change", type=float, default=1.0)
    parser.add_argument("--sim-backend", choices=("physx_cpu", "gpu"), default="physx_cpu")
    parser.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    if args.num_episodes < 1 or args.max_attempts < args.num_episodes:
        raise ValueError("max-attempts must cover the requested positive episode count")
    dataset_path = _resolve_output_path(args.repo_id)
    if dataset_path.exists() or args.output_dir.exists():
        raise FileExistsError("refusing to overwrite an existing dataset or output directory")
    args.output_dir.mkdir(parents=True)
    solver_env, replay_env = _build_env(args, control_mode="pd_joint_pos"), _build_env(args, control_mode="pd_joint_delta_pos")
    lower, upper = _joint_delta_arm_bounds(replay_env)
    dataset = None; rows: list[dict[str, Any]] = []; saved = attempts = 0; main_camera = wrist_camera = ""
    try:
        while saved < args.num_episodes and attempts < args.max_attempts:
            seed = args.seed + attempts; attempts += 1
            reference = _run_reference(solver_env, seed)
            if reference is None: continue
            _reference_records, solver_actions, metadata = reference
            replay = _replay(replay_env, seed, solver_actions, lower, upper)
            if replay is None: continue
            records, actions = replay
            if not main_camera:
                main_camera = _select_camera(records[0].obs, "", ("base_camera",) + MAIN_CAMERA_CANDIDATES, "main")
                wrist_camera = _select_camera(records[0].obs, "", ("hand_camera",) + WRIST_CAMERA_CANDIDATES, "wrist")
            frames = _build_frames(records=records, actions=actions, task=PICK_SINGLE_YCB_AIRPLANE_TASK,
                                   main_camera=main_camera, wrist_camera=wrist_camera)
            validate_visual_motion(frames, min_peak_mean_abs_delta=args.min_visual_change)
            if dataset is None:
                dataset = _create_dataset(repo_id=args.repo_id, image_shape=tuple(frames[0]["image"].shape),
                    wrist_image_shape=tuple(frames[0]["wrist_image"].shape), fps=args.control_freq,
                    image_writer_threads=4, image_writer_processes=4)
            for frame in frames: dataset.add_frame(frame)
            dataset.save_episode()
            rows.append({"episode_index": saved, "seed": seed, "source": "official_panda_motion_planning_top_down_airplane_oracle", **metadata,
                         "actions": len(actions), "frames": len(records)})
            if args.save_videos:
                write_episode_video_durably(frames, video_dir=_video_output_dir(args.repo_id, ""), episode_index=saved, seed=seed, fps=args.control_freq)
            saved += 1; LOG.info("accepted %d/%d seed=%d", saved, args.num_episodes, seed)
    finally:
        if dataset is not None and getattr(dataset, "image_writer", None) is not None: dataset.image_writer.wait_until_done()
        solver_env.close(); replay_env.close()
    if saved != args.num_episodes: raise RuntimeError(f"collected {saved}/{args.num_episodes} successful replays after {attempts} attempts")
    (args.output_dir / "episodes.jsonl").write_text("".join(json.dumps(row) + "\\n" for row in rows), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps({"dataset": str(dataset_path), "split": args.split, "episodes": saved, "attempts": attempts}, indent=2) + "\\n")


if __name__ == "__main__": main()
