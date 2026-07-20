from __future__ import annotations

import pytest

from toolkits.lerobot.collect_maniskill_plug_lerobot_joint import (
    build_episode_manifest_row,
    controlled_env_id,
    write_episode_video_durably,
)


def test_controlled_env_id_rejects_unknown_split():
    assert controlled_env_id("id") == "RLinfPlugChargerID-v1"
    assert controlled_env_id("ood") == "RLinfPlugChargerOOD-v1"
    with pytest.raises(ValueError):
        controlled_env_id("mixed")


def test_expert_manifest_row_preserves_reset_distribution_metadata():
    row = build_episode_manifest_row(
        episode_index=3,
        seed=17,
        metadata={"split": "id", "relative_yaw": 0.1, "charger_pose": {"p": [1, 2, 3]}},
    )
    assert row["source"] == "expert"
    assert row["episode_index"] == 3
    assert row["split"] == "id"
    assert row["relative_yaw"] == 0.1


def test_video_is_encoded_locally_before_copying_to_requested_directory(tmp_path, monkeypatch):
    calls = []

    def fake_write(frames, *, video_dir, episode_index, seed, fps):
        calls.append(video_dir)
        (video_dir / f"episode_{episode_index:06d}_seed_{seed:06d}.mp4").write_bytes(b"video")

    monkeypatch.setattr(
        "toolkits.lerobot.collect_maniskill_plug_lerobot_joint._write_episode_video",
        fake_write,
    )
    destination = write_episode_video_durably(
        [{"image": "unused"}],
        video_dir=tmp_path / "oss-output",
        episode_index=3,
        seed=7,
        fps=10,
    )

    assert calls and calls[0] != tmp_path / "oss-output"
    assert destination.read_bytes() == b"video"
