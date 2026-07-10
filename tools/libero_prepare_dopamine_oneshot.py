#!/usr/bin/env python3
"""Export one successful LIBERO demo into Robo-Dopamine's official SFT layout.

The exported annotation segments are uniformly spaced bootstrap keyframes. They are
valid input for the official data-generation pipeline, but are deliberately marked in
the manifest because they are not the paper's human semantic keyframe annotations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from libero_extract_grm_goal_bank import (
    MAIN_IMAGE_KEYS,
    WRIST_IMAGE_KEYS,
    _build_task_index,
    _find_hdf5_dataset,
    _select_hdf5_demo,
    _task_id_for_demo,
    _to_uint8_image,
)


def _find_demo(demo_root: Path, task_suite: str, task_id: int) -> Path:
    import h5py

    task_index = _build_task_index(task_suite)
    candidates = sorted(demo_root.rglob("*.hdf5")) + sorted(demo_root.rglob("*.h5"))
    for fallback_id, path in enumerate(candidates):
        try:
            with h5py.File(path, "r") as file:
                data = file["data"]
                problem_info = data.attrs.get("problem_info", "{}")
                if isinstance(problem_info, bytes):
                    problem_info = problem_info.decode("utf-8")
                info = json.loads(problem_info) if isinstance(problem_info, str) else {}
                candidate_id = _task_id_for_demo(
                    path,
                    fallback_id,
                    task_index,
                    task_description=str(info.get("language_instruction", "")),
                    bddl_file=Path(str(data.attrs.get("bddl_file_name", ""))).name,
                )
        except Exception:
            continue
        if candidate_id == task_id:
            return path
    raise FileNotFoundError(f"No demo for {task_suite} task_id={task_id} under {demo_root}")


def _write_video(dataset: Any, path: Path, frame_count: int, fps: int) -> None:
    import cv2

    first = np.asarray(_to_uint8_image(dataset[0]))
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video: {path}")
    try:
        for frame_idx in range(frame_count):
            rgb = np.asarray(_to_uint8_image(dataset[frame_idx]))
            if rgb.shape[:2] != (height, width):
                rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _segments(frame_count: int, segment_count: int) -> list[dict[str, Any]]:
    boundaries = np.linspace(0, frame_count - 1, segment_count + 1).round().astype(int)
    return [
        {
            "annotation": f"uniform_segment_{index:02d}",
            "start_frame_id": int(boundaries[index]),
            "end_frame_id": int(boundaries[index + 1]),
        }
        for index in range(segment_count)
        if boundaries[index + 1] > boundaries[index]
    ]


def export_oneshot_demo(
    demo_root: Path,
    output_dir: Path,
    task_suite: str,
    task_id: int,
    segment_count: int,
    fps: int,
) -> Path:
    import h5py

    source = _find_demo(demo_root, task_suite, task_id)
    episode_dir = output_dir / "episode_001"
    episode_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(source, "r") as file:
        data = file["data"]
        demo = _select_hdf5_demo(data)
        obs = demo["obs"] if "obs" in demo else demo
        main = _find_hdf5_dataset(obs, MAIN_IMAGE_KEYS)
        wrist = _find_hdf5_dataset(obs, WRIST_IMAGE_KEYS)
        if main is None:
            raise ValueError(f"No main camera images in {source}:{demo.name}")
        frame_count = len(main)
        if frame_count < 3:
            raise ValueError(f"Demo has too few frames: {frame_count}")
        _write_video(main, episode_dir / "cam_high.mp4", frame_count, fps)
        # LIBERO exposes one wrist camera. Duplicate it to preserve the official 8-image schema.
        if wrist is None:
            wrist = main
        _write_video(wrist, episode_dir / "cam_left_wrist.mp4", frame_count, fps)
        _write_video(wrist, episode_dir / "cam_right_wrist.mp4", frame_count, fps)

        problem_info = data.attrs.get("problem_info", "{}")
        if isinstance(problem_info, bytes):
            problem_info = problem_info.decode("utf-8")
        info = json.loads(problem_info) if isinstance(problem_info, str) else {}
        task_description = str(info.get("language_instruction", f"LIBERO task {task_id}"))
        manifest = {
            "task_suite": task_suite,
            "task_id": task_id,
            "task_description": task_description,
            "source_demo": f"{source}:{demo.name}",
            "frame_count": frame_count,
            "fps": fps,
            "annotation_strategy": "uniform_temporal_segments",
            "paper_equivalence": "one-shot bootstrap; replace with human semantic keyframes for strict paper reproduction",
            "views": ["main", "wrist", "wrist_duplicate"],
        }

    with open(output_dir / "task_instruction.json", "w", encoding="utf-8") as file:
        json.dump([task_description], file, ensure_ascii=False, indent=2)
    with open(episode_dir / "annotated_keyframes.json", "w", encoding="utf-8") as file:
        json.dump(_segments(frame_count, segment_count), file, ensure_ascii=False, indent=2)
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a single LIBERO success demo for official Robo-Dopamine one-shot SFT."
    )
    parser.add_argument("--demo-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--task-suite", default="libero_goal")
    parser.add_argument("--task-id", required=True, type=int)
    parser.add_argument("--segment-count", type=int, default=8)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    output = export_oneshot_demo(
        args.demo_root,
        args.output_dir,
        args.task_suite,
        args.task_id,
        args.segment_count,
        args.fps,
    )
    print(f"Exported official one-shot raw layout to {output}")


if __name__ == "__main__":
    main()
