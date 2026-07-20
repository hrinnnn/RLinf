#!/usr/bin/env python3
"""Merge per-dataset AWBC manifests in OpenPI ``ConcatDataset`` order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = list(range(len(rows)))
    actual = [int(row["dataset_index"]) for row in rows]
    if actual != expected:
        raise ValueError(f"{path} must have contiguous dataset_index values starting at zero")
    return rows


def merge_progress_rows(sources: Sequence[tuple[str, Sequence[dict[str, Any]]]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    dataset_offset = 0
    episode_offset = 0
    for source_name, rows in sources:
        if [int(row["dataset_index"]) for row in rows] != list(range(len(rows))):
            raise ValueError(f"{source_name} has non-contiguous dataset indices")
        source_episodes = [int(row["episode_index"]) for row in rows]
        for row in rows:
            item = dict(row)
            item["dataset_index"] = int(item["dataset_index"]) + dataset_offset
            item["episode_index"] = int(item["episode_index"]) + episode_offset
            item["source_dataset"] = source_name
            merged.append(item)
        dataset_offset += len(rows)
        episode_offset += (max(source_episodes) + 1) if source_episodes else 0
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs=2, metavar=("NAME", "MANIFEST"), action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = [(name, read_jsonl(Path(path))) for name, path in args.input]
    rows = merge_progress_rows(sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
