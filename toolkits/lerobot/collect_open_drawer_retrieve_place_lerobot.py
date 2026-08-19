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

import numpy as np

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
    _jsonable,
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
        metadata["oracle"] = {
            "type": "privileged_panda_motion_planning",
            "stages": _jsonable(stages),
        }
        valid = bool(stages["success"]) and bool(actions) and len(records) == len(actions) + 1
        return records, actions, dict(metadata), valid
    except Exception as exc:
        metadata["oracle_error"] = repr(exc)
        return records, actions, dict(metadata), False
    finally:
        env.reset = original_reset  # type: ignore[method-assign]
        env.step = original_step  # type: ignore[method-assign]


def _replay(env: Any, seed: int, solver_actions: list[Any], lower, upper):
    observation, _info = env.reset(seed=seed)
    metadata = reset_metadata(env, split="id")
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
    return records, actions, metadata, bool(success and len(records) == len(actions) + 1)


def _save_attempt_evidence(
    attempt_dir: Path,
    *,
    seed: int,
    attempt: int,
    metadata: dict[str, Any],
    records: list[Any],
    actions: list[Any],
    accepted: bool,
    reference_success: bool,
    replay_success: bool,
    main_camera: str,
    wrist_camera: str,
    control_freq: int,
) -> dict[str, Any]:
    """Persist raw attempt evidence before an accepted episode is committed."""

    attempt_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "attempt": attempt,
        "seed": seed,
        "accepted": accepted,
        "reference_success": reference_success,
        "replay_success": replay_success,
        "steps": len(actions),
        "reset_metadata": metadata,
    }
    if records and len(records) == len(actions) + 1:
        action_array = _to_numpy(actions).astype("float32") if actions else _to_numpy([]).astype("float32")
        state_array = _to_numpy([record.state for record in records]).astype("float32")
        npy_actions = attempt_dir / "actions.npy"
        npy_states = attempt_dir / "states.npy"
        np.save(npy_actions, action_array)
        np.save(npy_states, state_array)
        payload["actions"] = str(npy_actions)
        payload["states"] = str(npy_states)
        payload["state_shape"] = list(state_array.shape)
        payload["action_shape"] = list(action_array.shape)
        (attempt_dir / "reset_metadata.json").write_text(
            json.dumps(_jsonable(metadata), indent=2) + "\n", encoding="utf-8"
        )
        payload["reset_metadata_path"] = str(attempt_dir / "reset_metadata.json")
        if main_camera and wrist_camera and actions:
            frames = _build_frames(
                records=records,
                actions=actions,
                task=TASK_INSTRUCTION,
                main_camera=main_camera,
                wrist_camera=wrist_camera,
            )
            video_path = write_episode_video_durably(
                frames,
                video_dir=attempt_dir,
                episode_index=0,
                seed=seed,
                fps=control_freq,
            )
            payload["video"] = str(video_path)
    (attempt_dir / "attempt.json").write_text(
        json.dumps(_jsonable(payload), indent=2) + "\n", encoding="utf-8"
    )
    return payload


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
    raw_attempt_dir = args.output_dir / "raw_attempts"
    accepted_evidence_dir = args.output_dir / "episodes"
    for path in (dataset_path, args.output_dir, video_dir, raw_attempt_dir):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing path: {path}")
    args.output_dir.mkdir(parents=True)
    raw_attempt_dir.mkdir()
    accepted_evidence_dir.mkdir()

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
            attempt_dir = raw_attempt_dir / f"attempt_{attempts:06d}_seed_{seed}"
            reference = _run_reference(solver_env, seed, planner)
            reference_records, solver_actions, reference_metadata, reference_success = reference
            if not reference_success:
                _save_attempt_evidence(
                    attempt_dir,
                    seed=seed,
                    attempt=attempts,
                    metadata=reference_metadata,
                    records=reference_records,
                    actions=solver_actions,
                    accepted=False,
                    reference_success=False,
                    replay_success=False,
                    main_camera="",
                    wrist_camera="",
                    control_freq=args.control_freq,
                )
                LOG.warning("Rejected seed %d: absolute-position oracle failed", seed)
                continue
            replay = _replay(replay_env, seed, solver_actions, lower, upper)
            records, actions, replay_metadata, replay_success = replay
            metadata = {**reference_metadata, **replay_metadata}
            metadata["oracle"] = reference_metadata.get("oracle", {})
            if not replay_success:
                _save_attempt_evidence(
                    attempt_dir,
                    seed=seed,
                    attempt=attempts,
                    metadata=metadata,
                    records=records,
                    actions=actions,
                    accepted=False,
                    reference_success=True,
                    replay_success=False,
                    main_camera="",
                    wrist_camera="",
                    control_freq=args.control_freq,
                )
                LOG.warning("Rejected seed %d: joint-delta replay failed", seed)
                continue

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
            metadata["camera"] = {
                "main": main_camera,
                "wrist": wrist_camera,
                "main_shape": list(frames[0]["image"].shape),
                "wrist_shape": list(frames[0]["wrist_image"].shape),
                "requested_size": [args.image_size, args.image_size],
            }
            action_array = np.asarray(actions, dtype=np.float32)
            state_array = np.asarray([record.state for record in records], dtype=np.float32)
            if action_array.ndim != 2 or action_array.shape[1] != 8:
                _save_attempt_evidence(
                    attempt_dir,
                    seed=seed,
                    attempt=attempts,
                    metadata=metadata,
                    records=records,
                    actions=actions,
                    accepted=False,
                    reference_success=True,
                    replay_success=True,
                    main_camera=main_camera,
                    wrist_camera=wrist_camera,
                    control_freq=args.control_freq,
                )
                LOG.warning("Rejected seed %d: invalid action shape %s", seed, action_array.shape)
                continue
            if state_array.shape != (len(actions) + 1, 9) or not visual_motion:
                _save_attempt_evidence(
                    attempt_dir,
                    seed=seed,
                    attempt=attempts,
                    metadata=metadata,
                    records=records,
                    actions=actions,
                    accepted=False,
                    reference_success=True,
                    replay_success=True,
                    main_camera=main_camera,
                    wrist_camera=wrist_camera,
                    control_freq=args.control_freq,
                )
                LOG.warning("Rejected seed %d: state shape %s or visual_motion=%s", seed, state_array.shape, visual_motion)
                continue

            _save_attempt_evidence(
                attempt_dir,
                seed=seed,
                attempt=attempts,
                metadata=metadata,
                records=records,
                actions=actions,
                accepted=True,
                reference_success=True,
                replay_success=True,
                main_camera=main_camera,
                wrist_camera=wrist_camera,
                control_freq=args.control_freq,
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
            video_path = None
            if args.save_videos:
                video_path = write_episode_video_durably(
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
                "video": str(video_path) if video_path else None,
                **metadata,
            }
            accepted_dir = accepted_evidence_dir / f"episode_{saved:06d}"
            accepted_dir.mkdir()
            np.save(accepted_dir / "actions.npy", action_array)
            np.save(accepted_dir / "states.npy", state_array)
            (accepted_dir / "reset_metadata.json").write_text(
                json.dumps(_jsonable(metadata), indent=2) + "\n", encoding="utf-8"
            )
            (accepted_dir / "oracle_stages.json").write_text(
                json.dumps(_jsonable(metadata.get("oracle", {}).get("stages", {})), indent=2) + "\n",
                encoding="utf-8",
            )
            row["accepted_evidence"] = str(accepted_dir)
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
        "raw_attempts": str(raw_attempt_dir),
        "accepted_evidence": str(accepted_evidence_dir),
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
