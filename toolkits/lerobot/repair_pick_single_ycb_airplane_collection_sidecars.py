#!/usr/bin/env python3
"""Repair legacy airplane collection sidecars and validate every rollout video.

The first collector version wrote the two JSON sidecars with literal ``\\n``
characters.  This utility leaves those originals untouched and writes new,
parseable files alongside them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v3 as iio


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    args = parser.parse_args()

    raw_episodes = (args.collection_dir / "episodes.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(part) for part in raw_episodes.split("\\n") if part]
    repaired_episodes = args.collection_dir / "episodes_repaired.jsonl"
    repaired_episodes.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )

    raw_summary = (args.collection_dir / "summary.json").read_text(encoding="utf-8")
    summary = json.loads(raw_summary.removesuffix("\\n"))
    (args.collection_dir / "summary_repaired.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    videos = sorted(args.videos_dir.glob("*.mp4"))
    video_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for path in videos:
        try:
            frame = iio.imread(path, index=0)
            video_rows.append(
                {
                    "filename": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "first_frame_shape": list(frame.shape),
                }
            )
        except Exception as error:  # Validation must report every undecodable video.
            failures.append({"filename": path.name, "error": repr(error)})

    manifest_path = args.collection_dir / "videos_manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in video_rows), encoding="utf-8"
    )
    validation = {
        "episode_sidecars": len(rows),
        "video_files": len(videos),
        "decoded_videos": len(video_rows),
        "failures": failures,
        "videos_dir": str(args.videos_dir),
    }
    (args.collection_dir / "video_decode_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if len(rows) != len(videos) or failures:
        raise RuntimeError(json.dumps(validation, sort_keys=True))


if __name__ == "__main__":
    main()
