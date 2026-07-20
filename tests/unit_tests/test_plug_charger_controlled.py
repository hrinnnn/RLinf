from __future__ import annotations

import numpy as np

from rlinf.envs.maniskill.plug_charger_variants import (
    PLUG_CHARGER_ID_YAW_RANGE,
    PLUG_CHARGER_OOD_YAW_RANGE,
    split_for_env_id,
    wrap_yaw,
    yaw_from_quaternion_wxyz,
)


def test_controlled_yaw_ranges_are_disjoint_with_a_large_gap():
    assert PLUG_CHARGER_ID_YAW_RANGE[1] < PLUG_CHARGER_OOD_YAW_RANGE[0]
    # The closest ID/OOD orientations are 15° and 165° apart.
    assert np.isclose(
        PLUG_CHARGER_OOD_YAW_RANGE[0] - PLUG_CHARGER_ID_YAW_RANGE[1],
        5 * np.pi / 6,
    )


def test_quaternion_yaw_and_wrapping_follow_wxyz_convention():
    quarter_turn = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    assert np.isclose(yaw_from_quaternion_wxyz(quarter_turn), np.pi / 2)
    assert np.isclose(wrap_yaw(3 * np.pi), -np.pi)


def test_split_lookup_only_accepts_registered_controlled_ids():
    assert split_for_env_id("RLinfPlugChargerID-v1") == "id"
    assert split_for_env_id("RLinfPlugChargerOOD-v1") == "ood"
