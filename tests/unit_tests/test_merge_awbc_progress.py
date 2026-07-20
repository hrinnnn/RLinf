from __future__ import annotations

from toolkits.lerobot.merge_awbc_progress import merge_progress_rows


def _row(index: int, episode: int, source: str) -> dict:
    return {
        "dataset_index": index,
        "episode_index": episode,
        "frame_index": index,
        "next_frame_index": index + 1,
        "phi": 0.0,
        "phi_next": 0.25,
        "delta_phi": 0.25,
        "valid": True,
        "confidence": 1.0,
        "episode_length_chunks": 1,
        "source": source,
        "success": False,
    }


def test_merge_offsets_dataset_and_episode_indices_in_concat_order():
    merged = merge_progress_rows(
        [
            ("expert", [_row(0, 0, "expert"), _row(1, 0, "expert")]),
            ("round_1", [_row(0, 0, "policy")]),
        ]
    )
    assert [row["dataset_index"] for row in merged] == [0, 1, 2]
    assert [row["episode_index"] for row in merged] == [0, 0, 1]
    assert merged[-1]["source_dataset"] == "round_1"
