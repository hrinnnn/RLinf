#!/usr/bin/env python3
"""Collect replayable PickSingleYCB object-variation demonstrations."""

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

from rlinf.envs.maniskill.pick_single_ycb_object_variation import (  # noqa: E402
    PICK_SINGLE_YCB_OBJECT_ID_ENV_ID,
    PICK_SINGLE_YCB_OBJECT_OOD_ENV_ID,
    PICK_SINGLE_YCB_OBJECT_TASK,
    register_controlled_pick_single_ycb_object_variants,
    reset_metadata,
)
from rlinf.toolkits.lerobot.collect_maniskill_peg_lerobot_joint import (  # noqa: E402
    _build_frames,
    _camera_image,
    _convert_solver_action_to_joint_delta,
    _create_dataset,
    _extract_record,
    _joint_delta_arm_bounds,
    _select_camera,
    _to_numpy,
    validate_visual_motion,
)
from toolkits.lerobot.diagnose_pick_single_ycb_object_variation_oracle import (  # noqa: E402
    run_oracle,
)

LOG = logging.getLogger("collect_maniskill_pick_single_ycb_object_variation_lerobot")


def _build_env(split: str, *, control_mode: str, max_episode_steps: int):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    register_controlled_pick_single_ycb_object_variants()
    env_id = PICK_SINGLE_YCB_OBJECT_ID_ENV_ID if split == "id" else PICK_SINGLE_YCB_OBJECT_OOD_ENV_ID
    return gym.make(
        env_id,
        num_envs=1,
        robot_uids="panda_wristcam",
        obs_mode="rgb",
        control_mode=control_mode,
        reward_mode="sparse",
        sim_backend="physx_cpu",
        sim_config={"sim_freq": 100, "control_freq": 10},
        render_mode="rgb_array",
        max_episode_steps=max_episode_steps,
    )


def _reference(env: Any, *, seed: int):
    records: list[Any] = []
    solver_actions: list[np.ndarray] = []
    original_reset, original_step = env.reset, env.step

    def reset_hook(*args, **kwargs):
        obs, info = original_reset(*args, **kwargs)
        records.clear()
        solver_actions.clear()
        records.append(_extract_record(obs))
        return obs, info

    def step_hook(action, *args, **kwargs):
        solver_actions.append(_to_numpy(action).astype(np.float32).reshape(-1))
        obs, reward, terminated, truncated, info = original_step(action, *args, **kwargs)
        records.append(_extract_record(obs))
        return obs, reward, terminated, truncated, info

    env.reset, env.step = reset_hook, step_hook  # type: ignore[method-assign]
    try:
        result = run_oracle(env, seed=seed)
        if not bool(result["accepted"]) or len(records) != len(solver_actions) + 1:
            return None, result
        return (records, solver_actions), result
    finally:
        env.reset, env.step = original_reset, original_step  # type: ignore[method-assign]


def _replay(env: Any, *, seed: int, solver_actions: list[np.ndarray], lower: np.ndarray, upper: np.ndarray):
    obs, _ = env.reset(seed=seed)
    records = [_extract_record(obs)]
    actions: list[np.ndarray] = []
    last_info: dict[str, Any] = {}
    for solver_action in solver_actions:
        action = _convert_solver_action_to_joint_delta(records[-1].qpos, solver_action, lower, upper)
        obs, _reward, terminated, truncated, info = env.step(action)
        actions.append(action.astype(np.float32))
        records.append(_extract_record(obs))
        last_info = info
        if bool(np.asarray(info.get("success", False)).reshape(-1)[0]):
            break
        if bool(np.asarray(terminated).reshape(-1)[0]) or bool(np.asarray(truncated).reshape(-1)[0]):
            break
    if len(records) != len(actions) + 1:
        return None
    if not bool(np.asarray(last_info.get("success", False)).reshape(-1)[0]):
        return None
    return records, actions


def _write_video(frames: list[dict[str, Any]], destination: Path, fps: int) -> None:
    from rlinf.toolkits.lerobot.collect_maniskill_peg_lerobot_joint import _make_video_frame
    import imageio.v2 as imageio

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="object-variation-video-") as temp:
        encoded = Path(temp) / destination.name
        writer = imageio.get_writer(encoded, format="FFMPEG", fps=fps, codec="libx264", pixelformat="yuv420p")
        try:
            for frame in frames:
                writer.append_data(_make_video_frame(frame))
        finally:
            writer.close()
        if not encoded.is_file() or encoded.stat().st_size == 0:
            raise RuntimeError("video encoder produced no file")
        shutil.copy2(encoded, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("id", "ood"), required=True)
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--max-attempts", type=int, required=True)
    parser.add_argument("--repo-id", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    args = parser.parse_args()
    if args.output_dir.exists() or args.repo_id.exists():
        raise FileExistsError("refusing to overwrite existing collection output")

    args.output_dir.mkdir(parents=True)
    (args.output_dir / "videos").mkdir()
    solver_env = _build_env(args.split, control_mode="pd_joint_pos", max_episode_steps=args.max_episode_steps)
    replay_env = _build_env(args.split, control_mode="pd_joint_delta_pos", max_episode_steps=args.max_episode_steps)
    lower, upper = _joint_delta_arm_bounds(replay_env)
    dataset = None
    rows: list[dict[str, Any]] = []
    accepted = 0
    attempts = 0
    main_camera = wrist_camera = ""
    try:
        while accepted < args.num_episodes and attempts < args.max_attempts:
            seed = args.seed_start + attempts
            attempts += 1
            reference, oracle = _reference(solver_env, seed=seed)
            if reference is None:
                rows.append({"attempt": attempts - 1, "seed": seed, "accepted": False, "oracle": oracle})
                continue
            records, solver_actions = reference
            replay = _replay(replay_env, seed=seed, solver_actions=solver_actions, lower=lower, upper=upper)
            if replay is None:
                rows.append({"attempt": attempts - 1, "seed": seed, "accepted": False, "replay_success": False, "oracle": oracle})
                continue
            replay_records, actions = replay
            if not main_camera:
                main_camera = _select_camera(replay_records[0].obs, "base_camera", ("base_camera",), "main")
                wrist_camera = _select_camera(replay_records[0].obs, "hand_camera", ("hand_camera",), "wrist")
            frames = _build_frames(
                records=replay_records,
                actions=actions,
                task=PICK_SINGLE_YCB_OBJECT_TASK,
                main_camera=main_camera,
                wrist_camera=wrist_camera,
            )
            validate_visual_motion(frames, min_peak_mean_abs_delta=1.0)
            if dataset is None:
                dataset = _create_dataset(
                    repo_id=str(args.repo_id),
                    image_shape=tuple(frames[0]["image"].shape),
                    wrist_image_shape=tuple(frames[0]["wrist_image"].shape),
                    fps=10,
                    image_writer_threads=4,
                    image_writer_processes=2,
                )
            for frame in frames:
                dataset.add_frame(frame)
            dataset.save_episode()
            video = args.output_dir / "videos" / f"episode_{accepted:06d}_seed_{seed:06d}.mp4"
            _write_video(frames, video, fps=10)
            rows.append(
                {
                    "attempt": attempts - 1,
                    "seed": seed,
                    "accepted": True,
                    "frames": len(replay_records),
                    "actions": len(actions),
                    "video": str(video),
                    "oracle": oracle,
                    "reset_metadata": oracle["reset_metadata"],
                }
            )
            accepted += 1
            LOG.info("accepted %d/%d seed=%d", accepted, args.num_episodes, seed)
    finally:
        if dataset is not None and getattr(dataset, "image_writer", None) is not None:
            dataset.image_writer.wait_until_done()
        solver_env.close()
        replay_env.close()

    summary = {
        "format": "pick_single_ycb_object_variation_lerobot_collection_v1",
        "split": args.split,
        "episodes": accepted,
        "target_episodes": args.num_episodes,
        "attempts": attempts,
        "raw_attempts": len(rows),
        "video_count": len(list((args.output_dir / "videos").glob("*.mp4"))),
        "dataset": str(args.repo_id),
        "object_variation_only": True,
    }
    (args.output_dir / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if accepted != args.num_episodes:
        (args.output_dir / "COLLECTION_FAILED").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(2)
    (args.output_dir / "COLLECTION_COMPLETE").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()

