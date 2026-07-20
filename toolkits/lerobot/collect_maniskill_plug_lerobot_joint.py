#!/usr/bin/env python3
"""Collect controlled PlugCharger expert demonstrations for pi0.5.

The official PlugCharger solver only supports absolute joint-position control.
We record that solver, replay it from the same reset using joint deltas, and
save the replayed successful trajectory as the training dataset.  Reset
metadata is emitted beside the LeRobot dataset so ID/OOD provenance cannot be
lost during later online updates.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

RLINF_ROOT = Path(__file__).resolve().parents[2]
if str(RLINF_ROOT) not in sys.path:
    sys.path.insert(0, str(RLINF_ROOT))

from rlinf.envs.maniskill.plug_charger_variants import (
    PLUG_CHARGER_ID_ENV_ID,
    PLUG_CHARGER_OOD_ENV_ID,
    PLUG_CHARGER_TASK,
    register_controlled_plug_charger_variants,
    reset_metadata,
)
from toolkits.lerobot.collect_maniskill_peg_lerobot_joint import (
    MAIN_CAMERA_CANDIDATES,
    WRIST_CAMERA_CANDIDATES,
    _bool_scalar,
    _build_frames,
    _create_dataset,
    _extract_record,
    _joint_delta_arm_bounds,
    _resolve_output_path,
    _select_camera,
    _solver_success,
    _to_numpy,
    _video_output_dir,
    _write_episode_video,
    _convert_solver_action_to_joint_delta,
)


LOG = logging.getLogger("collect_maniskill_plug_lerobot_joint")


def controlled_env_id(split: str) -> str:
    if split == "id":
        return PLUG_CHARGER_ID_ENV_ID
    if split == "ood":
        return PLUG_CHARGER_OOD_ENV_ID
    raise ValueError(f"split must be id or ood, got {split!r}")


def build_episode_manifest_row(*, episode_index: int, seed: int, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "episode_index": int(episode_index),
        "seed": int(seed),
        "source": "expert",
        **metadata,
    }


def _build_env(args: argparse.Namespace, *, control_mode: str):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    register_controlled_plug_charger_variants()
    return gym.make(
        controlled_env_id(args.split),
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


def _official_solver():
    module = importlib.import_module("mani_skill.examples.motionplanning.panda.solutions.plug_charger")
    return module.solve


def _run_reference(env: Any, seed: int) -> tuple[list[Any], list[Any], dict[str, Any]] | None:
    records: list[Any] = []
    actions: list[Any] = []
    metadata: dict[str, Any] = {}
    reset = env.reset
    step = env.step

    def reset_hook(*args, **kwargs):
        observation, info = reset(*args, **kwargs)
        records.clear()
        actions.clear()
        records.append(_extract_record(observation))
        metadata.clear()
        metadata.update(reset_metadata(env))
        return observation, info

    def step_hook(action, *args, **kwargs):
        actions.append(_to_numpy(action).astype("float32").reshape(-1))
        observation, reward, terminated, truncated, info = step(action, *args, **kwargs)
        records.append(_extract_record(observation))
        return observation, reward, terminated, truncated, info

    env.reset = reset_hook  # type: ignore[method-assign]
    env.step = step_hook  # type: ignore[method-assign]
    try:
        result = _official_solver()(env, seed=seed, debug=False, vis=False)
        if not _solver_success(result) or len(records) < 2 or not actions:
            return None
        return records, actions, dict(metadata)
    finally:
        env.reset = reset  # type: ignore[method-assign]
        env.step = step  # type: ignore[method-assign]


def _replay(env: Any, seed: int, solver_actions: list[Any], lower, upper):
    observation, _ = env.reset(seed=seed)
    records = [_extract_record(observation)]
    actions = []
    success = False
    for solver_action in solver_actions:
        action = _convert_solver_action_to_joint_delta(records[-1].qpos, solver_action, lower, upper)
        observation, _reward, terminated, truncated, info = env.step(action)
        records.append(_extract_record(observation))
        actions.append(action)
        success = _bool_scalar(info.get("success"))
        if success or _bool_scalar(terminated) or _bool_scalar(truncated):
            break
    if not success or len(records) != len(actions) + 1:
        return None
    return records, actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("id", "ood"), default="id")
    parser.add_argument("--num-episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--control-freq", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--sim-backend", choices=("physx_cpu", "gpu"), default="physx_cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    if args.num_episodes <= 0 or args.max_attempts < args.num_episodes:
        raise ValueError("num-episodes must be positive and max-attempts must cover it")
    dataset_path = _resolve_output_path(args.repo_id)
    if dataset_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"Dataset exists at {dataset_path}; pass --overwrite")
        shutil.rmtree(dataset_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "episodes.jsonl"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Manifest exists at {manifest_path}; pass --overwrite")

    solver_env = _build_env(args, control_mode="pd_joint_pos")
    replay_env = _build_env(args, control_mode="pd_joint_delta_pos")
    lower, upper = _joint_delta_arm_bounds(replay_env)
    dataset = None
    manifest_rows = []
    main_camera = wrist_camera = ""
    saved = attempts = 0
    try:
        while saved < args.num_episodes and attempts < args.max_attempts:
            seed = args.seed + attempts
            attempts += 1
            reference = _run_reference(solver_env, seed)
            if reference is None:
                continue
            reference_records, solver_actions, metadata = reference
            replay = _replay(replay_env, seed, solver_actions, lower, upper)
            if replay is None:
                continue
            records, actions = replay
            if not main_camera:
                main_camera = _select_camera(records[0].obs, "", ("base_camera",) + MAIN_CAMERA_CANDIDATES, "main")
                wrist_camera = _select_camera(records[0].obs, "", ("hand_camera",) + WRIST_CAMERA_CANDIDATES, "wrist")
            frames = _build_frames(records=records, actions=actions, task=PLUG_CHARGER_TASK, main_camera=main_camera, wrist_camera=wrist_camera)
            if dataset is None:
                dataset = _create_dataset(repo_id=args.repo_id, image_shape=tuple(frames[0]["image"].shape), wrist_image_shape=tuple(frames[0]["wrist_image"].shape), fps=args.control_freq, image_writer_threads=4, image_writer_processes=4)
            for frame in frames:
                dataset.add_frame(frame)
            dataset.save_episode()
            manifest_rows.append(build_episode_manifest_row(episode_index=saved, seed=seed, metadata=metadata))
            if args.save_videos:
                _write_episode_video(frames, video_dir=_video_output_dir(args.repo_id, ""), episode_index=saved, seed=seed, fps=args.control_freq)
            saved += 1
    finally:
        if dataset is not None and getattr(dataset, "image_writer", None) is not None:
            dataset.image_writer.wait_until_done()
        solver_env.close()
        replay_env.close()
    if saved != args.num_episodes:
        raise RuntimeError(f"collected {saved}/{args.num_episodes} successful {args.split} trajectories after {attempts} attempts")
    manifest_path.write_text("".join(json.dumps(row) + "\n" for row in manifest_rows), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps({"dataset": str(dataset_path), "split": args.split, "episodes": saved, "attempts": attempts}, indent=2) + "\n")


if __name__ == "__main__":
    main()
