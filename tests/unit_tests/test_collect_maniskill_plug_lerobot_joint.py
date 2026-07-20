from __future__ import annotations

import pytest

from toolkits.lerobot.collect_maniskill_plug_lerobot_joint import (
    build_episode_manifest_row,
    controlled_env_id,
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
