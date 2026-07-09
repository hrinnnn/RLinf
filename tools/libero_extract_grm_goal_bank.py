#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

MAIN_IMAGE_KEYS = (
    "agentview_rgb",
    "agentview_image",
    "image",
    "main_image",
    "main_images",
    "full_image",
    "rgb",
)
WRIST_IMAGE_KEYS = (
    "eye_in_hand_rgb",
    "robot0_eye_in_hand_image",
    "wrist_image",
    "wrist_images",
)


def _to_uint8_image(array: Any) -> Image.Image:
    arr = np.asarray(array)
    while arr.ndim > 3:
        arr = arr[-1]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.dtype != np.uint8:
        if np.max(arr) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim != 3:
        raise ValueError(f"Expected image array, got shape {arr.shape}")
    return Image.fromarray(arr[..., :3]).convert("RGB")


def _walk_dict(obj: Any, prefix: str = ""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_path = f"{prefix}/{key}" if prefix else str(key)
            yield key_path, value
            yield from _walk_dict(value, key_path)


def _find_image(obj: Any, candidates: tuple[str, ...]) -> Image.Image | None:
    for key_path, value in _walk_dict(obj):
        key_name = key_path.split("/")[-1]
        if key_name not in candidates:
            continue
        try:
            return _to_uint8_image(value)
        except Exception:
            continue
    return None


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as file:
        return pickle.load(file)


def _load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _load_hdf5(path: Path) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to read LIBERO hdf5 demos") from exc

    def read_group(group):
        result = {}
        for key, value in group.items():
            if isinstance(value, h5py.Dataset):
                result[key] = value[()]
            else:
                result[key] = read_group(value)
        return result

    with h5py.File(path, "r") as file:
        return read_group(file)


def _read_hdf5_attr_json(value: Any) -> dict[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def _select_hdf5_demo(data_group: Any) -> Any:
    demo_names = sorted(
        (name for name in data_group.keys() if name.startswith("demo_")),
        key=lambda name: int(name.split("_")[-1]),
    )
    if not demo_names:
        raise ValueError("No demo_* groups found")

    first_demo = data_group[demo_names[0]]
    selected = first_demo
    for name in demo_names:
        demo = data_group[name]
        rewards = demo.get("rewards")
        if rewards is not None and np.asarray(rewards).max(initial=0) > 0:
            return demo
        selected = demo
    return selected


def _find_hdf5_dataset(group: Any, candidates: tuple[str, ...]) -> Any | None:
    for candidate in candidates:
        parts = candidate.split("/")
        node = group
        for part in parts:
            if part not in node:
                node = None
                break
            node = node[part]
        if node is not None:
            return node
    return None


def _extract_hdf5_goal(path: Path) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to read LIBERO hdf5 demos") from exc

    with h5py.File(path, "r") as file:
        data_group = file["data"]
        demo = _select_hdf5_demo(data_group)
        rewards = demo.get("rewards")
        if rewards is not None:
            rewards_array = np.asarray(rewards)
            success_indices = np.where(rewards_array > 0)[0]
            frame_idx = int(success_indices[-1]) if len(success_indices) else -1
        else:
            frame_idx = -1

        obs_group = demo["obs"] if "obs" in demo else demo
        main_dataset = _find_hdf5_dataset(obs_group, MAIN_IMAGE_KEYS)
        if main_dataset is None:
            raise ValueError(f"No main image dataset found in {path}")
        wrist_dataset = _find_hdf5_dataset(obs_group, WRIST_IMAGE_KEYS)

        problem_info = _read_hdf5_attr_json(data_group.attrs.get("problem_info", ""))
        return {
            "main": _to_uint8_image(main_dataset[frame_idx]),
            "wrist": _to_uint8_image(wrist_dataset[frame_idx])
            if wrist_dataset is not None
            else None,
            "task_description": str(problem_info.get("language_instruction", "")),
            "bddl_file": Path(str(data_group.attrs.get("bddl_file_name", ""))).name,
            "demo_name": demo.name,
            "frame_index": frame_idx,
        }


def _load_demo(path: Path) -> Any:
    if path.suffix in {".pkl", ".pickle"}:
        return _load_pickle(path)
    if path.suffix == ".npz":
        return _load_npz(path)
    if path.suffix in {".hdf5", ".h5"}:
        return _load_hdf5(path)
    raise ValueError(f"Unsupported demo file: {path}")


def _task_id_from_path(path: Path, fallback: int) -> int:
    for part in reversed(path.parts):
        digits = "".join(ch for ch in part if ch.isdigit())
        if digits:
            return int(digits)
    return fallback


def _normalize_task_text(text: str) -> str:
    return "_".join(text.lower().replace("-", "_").split())


def _build_task_index(task_suite_name: str) -> dict[str, int]:
    try:
        from libero.libero import benchmark
    except Exception:
        return {}

    benchmark_dict = benchmark.get_benchmark_dict()
    if task_suite_name not in benchmark_dict:
        return {}
    suite = benchmark_dict[task_suite_name]()
    task_index: dict[str, int] = {}
    for task_id in range(suite.get_num_tasks()):
        task = suite.get_task(task_id)
        language = str(getattr(task, "language", ""))
        bddl_file = str(getattr(task, "bddl_file", ""))
        if language:
            task_index[f"lang:{_normalize_task_text(language)}"] = task_id
        if bddl_file:
            task_index[f"bddl:{Path(bddl_file).name}"] = task_id
            task_index[f"stem:{Path(bddl_file).stem}"] = task_id
    return task_index


def _task_id_for_demo(
    demo_path: Path,
    fallback: int,
    task_index: dict[str, int],
    task_description: str = "",
    bddl_file: str = "",
) -> int:
    if bddl_file:
        task_id = task_index.get(f"bddl:{Path(bddl_file).name}")
        if task_id is not None:
            return task_id
    if task_description:
        task_id = task_index.get(f"lang:{_normalize_task_text(task_description)}")
        if task_id is not None:
            return task_id
    task_id = task_index.get(f"stem:{demo_path.name.removesuffix('_demo.hdf5')}")
    if task_id is not None:
        return task_id
    return _task_id_from_path(demo_path, fallback)


def _task_description(obj: Any, task_id: int) -> str:
    for key_path, value in _walk_dict(obj):
        key_name = key_path.split("/")[-1]
        if key_name in {"task_description", "language_instruction", "instruction"}:
            if isinstance(value, bytes):
                return value.decode("utf-8")
            if isinstance(value, np.ndarray):
                value = value.reshape(-1)[0]
                if isinstance(value, bytes):
                    return value.decode("utf-8")
            return str(value)
    return f"libero task {task_id}"


def extract_goal_bank(
    demo_root: Path, output_dir: Path, task_suite_name: str = "libero_spatial"
) -> int:
    demo_files = sorted(
        path
        for path in demo_root.rglob("*")
        if path.suffix in {".pkl", ".pickle", ".npz", ".hdf5", ".h5"}
    )
    task_index = _build_task_index(task_suite_name)
    written_task_ids: set[int] = set()
    for fallback_id, demo_path in enumerate(demo_files):
        try:
            if demo_path.suffix in {".hdf5", ".h5"}:
                extracted = _extract_hdf5_goal(demo_path)
                main_image = extracted["main"]
                wrist_image = extracted["wrist"]
                task_description = extracted["task_description"]
                bddl_file = extracted["bddl_file"]
                source_demo = f"{demo_path}:{extracted['demo_name']}:{extracted['frame_index']}"
            else:
                demo = _load_demo(demo_path)
                main_image = _find_image(demo, MAIN_IMAGE_KEYS)
                if main_image is None:
                    continue
                wrist_image = _find_image(demo, WRIST_IMAGE_KEYS)
                task_description = _task_description(demo, fallback_id)
                bddl_file = ""
                source_demo = str(demo_path)
        except Exception as exc:
            print(f"[skip] {demo_path}: {exc}")
            continue

        task_id = _task_id_for_demo(
            demo_path,
            fallback_id,
            task_index,
            task_description=task_description,
            bddl_file=bddl_file,
        )
        if task_id in written_task_ids:
            continue
        task_dir = output_dir / f"task_{task_id:03d}"
        task_dir.mkdir(parents=True, exist_ok=True)
        main_image.save(task_dir / "goal_main.png")
        views = {"main": "goal_main.png"}
        if wrist_image is not None:
            wrist_image.save(task_dir / "goal_wrist.png")
            views["wrist"] = "goal_wrist.png"

        meta = {
            "task_id": task_id,
            "task_description": task_description or f"libero task {task_id}",
            "source_demo": source_demo,
            "views": views,
        }
        with open(task_dir / "meta.json", "w", encoding="utf-8") as file:
            json.dump(meta, file, ensure_ascii=False, indent=2)
        written_task_ids.add(task_id)
        print(f"[write] task_{task_id:03d} <- {demo_path}")
    return len(written_task_ids)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Robo-Dopamine goal reference images from LIBERO demos."
    )
    parser.add_argument("--demo-root", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        default=Path("assets/grm_goal_bank/libero_spatial"),
        type=Path,
    )
    parser.add_argument("--task-suite-name", default="libero_spatial")
    args = parser.parse_args()
    count = extract_goal_bank(args.demo_root, args.output_dir, args.task_suite_name)
    print(f"Extracted {count} task goal reference(s) into {args.output_dir}")


if __name__ == "__main__":
    main()
