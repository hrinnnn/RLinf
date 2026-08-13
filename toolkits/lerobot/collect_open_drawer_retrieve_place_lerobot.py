#!/usr/bin/env python3
"""Collect successful OpenDrawerRetrievePlace ID demonstrations for X-VLA.

The privileged oracle runs with absolute joint-position targets. Each accepted
trajectory is replayed from the same seed with normalized joint-delta actions,
which is the action representation consumed by the X-VLA training pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

RLINF_ROOT = Path(__file__).resolve().parents[2]
if str(RLINF_ROOT) not in sys.path:
    sys.path.insert(0, str(RLINF_ROOT))

from rlinf.envs.maniskill.open_drawer_retrieve_place_spec import (  # noqa: E402
    ENV_IDS,
    TASK_INSTRUCTION,
    reset_metadata,
)
from toolkits.lerobot.collect_maniskill_peg_lerobot_joint import (  # noqa: E402
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
    validate_visual_motion,
)
from toolkits.lerobot.collect_maniskill_pick_single_ycb_airplane_lerobot import (  # noqa: E402
    write_episode_video_durably,
)
from toolkits.lerobot.validate_open_drawer_retrieve_place_oracle import (  # noqa: E402
    PandaPosePlannerClient,
    solve_episode,
)


LOG = logging.getLogger("collect_open_drawer_retrieve_place_lerobot")


def _build_env(args: argparse.Namespace, *, control_mode: str):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import rlinf.envs.maniskill.open_drawer_retrieve_place  # noqa: F401

    return gym.make(
        ENV_IDS["id"],
        robot_uids="panda_wristcam",
        num_envs=1,
        obs_mode="rgb",
        control_mode=control_mode,
        reward_mode="sparse",
        render_mode="rgb_array",
        sim_backend=args.sim_backend,
        sim_config={"sim_freq": 100, "control_freq": args.control_freq},
        sensor_configs={"width": args.image_size, "height": args.image_size},
        max_episode_steps=args.max_episode_steps,
    )


def _run_reference(env: Any, seed: int, planner: PandaPosePlannerClient):
    records: list[Any] = []
    actions: list[Any] = []
    metadata: dict[str, Any] = {}
    original_reset, original_step = env.reset, env.step

    def reset_hook(*args, **kwargs):
        observation, info = original_reset(*args, **kwargs)
        records.clear()
        actions.clear()
        records.append(_extract_record(observation))
        metadata.clear()
        metadata.update(reset_metadata(env, split="id"))
        return observation, info

    def step_hook(action, *args, **kwargs):
        actions.append(_to_numpy(action).astype("float32").reshape(-1))
        observation, reward, terminated, truncated, info = original_step(
            action, *args, **kwargs
        )
        records.append(_extract_record(observation))
        return observation, reward, terminated, truncated, info

    env.reset = reset_hook  # type: ignore[method-assign]
    env.step = step_hook  # type: ignore[method-assign]
    try:
        stages = solve_episode(env, seed, planner)
        if not bool(stages["success"]):
            return None
        if not actions or len(records) != len(actions) + 1:
            return None
        metadata["oracle"] = {
            "type": "privileged_panda_motion_planning",
            "stages": stages,
        }
        return records, actions, dict(metadata)
    finally:
        env.reset = original_reset  # type: ignore[method-assign]
        env.step = original_step  # type: ignore[method-assign]


def _replay(env: Any, seed: int, solver_actions: list[Any], lower, upper):
    observation, _info = env.reset(seed=seed)
    records = [_extract_record(observation)]
    actions = []
    success = False
    for solver_action in solver_actions:
        action = _convert_solver_action_to_joint_delta(
            records[-1].qpos, solver_action, lower, upper
        )
        observation, _reward, terminated, truncated, info = env.step(action)
        records.append(_extract_record(observation))
        actions.append(action)
        success = _bool_scalar(info.get("success", False))
        if success or _bool_scalar(terminated) or _bool_scalar(truncated):
            break
    if not success or len(records) != len(actions) + 1:
        return None
    return records, actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--num-episodes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=73000)
    parser.add_argument("--max-attempts", type=int, default=192)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--control-freq", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=400)
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
    video_dir = args.video_dir or args.output_dir / "videos"
    for path in (dataset_path, args.output_dir, video_dir):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing path: {path}")
    args.output_dir.mkdir(parents=True)

    solver_env = _build_env(args, control_mode="pd_joint_pos")
    replay_env = _build_env(args, control_mode="pd_joint_delta_pos")
    lower, upper = _joint_delta_arm_bounds(replay_env)
    planner = PandaPosePlannerClient()
    dataset = None
    rows: list[dict[str, Any]] = []
    saved = attempts = 0
    total_actions = 0
    main_camera = wrist_camera = ""
    manifest_path = args.output_dir / "episodes.jsonl"

    try:
        while saved < args.num_episodes and attempts < args.max_attempts:
            seed = args.seed + attempts
            attempts += 1
            reference = _run_reference(solver_env, seed, planner)
            if reference is None:
                LOG.warning("Rejected seed %d: absolute-position oracle failed", seed)
                continue
            _reference_records, solver_actions, metadata = reference
            replay = _replay(replay_env, seed, solver_actions, lower, upper)
            if replay is None:
                LOG.warning("Rejected seed %d: joint-delta replay failed", seed)
                continue
            records, actions = replay

            if not main_camera:
                main_camera = _select_camera(
                    records[0].obs,
                    "",
                    ("base_camera",) + MAIN_CAMERA_CANDIDATES,
                    "main",
                )
                wrist_camera = _select_camera(
                    records[0].obs,
                    "",
                    ("hand_camera",) + WRIST_CAMERA_CANDIDATES,
                    "wrist",
                )
            frames = _build_frames(
                records=records,
                actions=actions,
                task=TASK_INSTRUCTION,
                main_camera=main_camera,
                wrist_camera=wrist_camera,
            )
            visual_motion = validate_visual_motion(
                frames, min_peak_mean_abs_delta=args.min_visual_change
            )

            if dataset is None:
                dataset = _create_dataset(
                    repo_id=args.repo_id,
                    image_shape=tuple(frames[0]["image"].shape),
                    wrist_image_shape=tuple(frames[0]["wrist_image"].shape),
                    fps=args.control_freq,
                    image_writer_threads=4,
                    image_writer_processes=4,
                )
            for frame in frames:
                dataset.add_frame(frame)
            dataset.save_episode()
            if args.save_videos:
                write_episode_video_durably(
                    frames,
                    video_dir=video_dir,
                    episode_index=saved,
                    seed=seed,
                    fps=args.control_freq,
                )

            row = {
                "episode_index": saved,
                "seed": seed,
                "source": "oracle",
                "success": True,
                "control_mode": "pd_joint_delta_pos",
                "num_actions": len(actions),
                "main_camera": main_camera,
                "wrist_camera": wrist_camera,
                "visual_motion": visual_motion,
                **metadata,
            }
            rows.append(row)
            saved += 1
            total_actions += len(actions)
            manifest_path.write_text(
                "".join(json.dumps(item, sort_keys=True) + "\n" for item in rows),
                encoding="utf-8",
            )
            LOG.info(
                "Saved %d/%d seed=%d actions=%d visual=%s",
                saved,
                args.num_episodes,
                seed,
                len(actions),
                visual_motion,
            )
    finally:
        if dataset is not None and getattr(dataset, "image_writer", None) is not None:
            dataset.image_writer.wait_until_done()
        planner.close()
        solver_env.close()
        replay_env.close()

    if saved != args.num_episodes:
        raise RuntimeError(
            f"collected {saved}/{args.num_episodes} successful ID trajectories "
            f"after {attempts} attempts"
        )
    action_counts = [int(row["num_actions"]) for row in rows]
    summary = {
        "dataset": str(dataset_path),
        "videos": str(video_dir) if args.save_videos else None,
        "split": "id",
        "episodes": saved,
        "attempts": attempts,
        "total_actions": total_actions,
        "min_actions": min(action_counts),
        "max_actions": max(action_counts),
        "main_camera": main_camera,
        "wrist_camera": wrist_camera,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
