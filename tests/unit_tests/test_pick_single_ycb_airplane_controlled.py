from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_SOURCE = Path(__file__).parents[2] / "rlinf/envs/maniskill/pick_single_ycb_airplane_variants.py"
_SPEC = importlib.util.spec_from_file_location("pick_single_ycb_airplane_variants_test", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

PICK_SINGLE_YCB_AIRPLANE_ID_YAW_RANGE = _MODULE.PICK_SINGLE_YCB_AIRPLANE_ID_YAW_RANGE
PICK_SINGLE_YCB_AIRPLANE_OOD_YAW_RANGES = _MODULE.PICK_SINGLE_YCB_AIRPLANE_OOD_YAW_RANGES
sample_airplane_yaw = _MODULE.sample_airplane_yaw
yaw_in_ranges = _MODULE.yaw_in_ranges


def test_yaw_splits_are_disjoint():
    assert PICK_SINGLE_YCB_AIRPLANE_ID_YAW_RANGE[1] < PICK_SINGLE_YCB_AIRPLANE_OOD_YAW_RANGES[0][0]
    assert PICK_SINGLE_YCB_AIRPLANE_OOD_YAW_RANGES[1][1] < PICK_SINGLE_YCB_AIRPLANE_ID_YAW_RANGE[0]


def test_yaw_sampling_is_reproducible_and_respects_splits():
    id_a = sample_airplane_yaw(np.random.default_rng(7), 64, split="id")
    id_b = sample_airplane_yaw(np.random.default_rng(7), 64, split="id")
    ood = sample_airplane_yaw(np.random.default_rng(7), 64, split="ood")
    assert np.array_equal(id_a, id_b)
    assert np.all((id_a >= PICK_SINGLE_YCB_AIRPLANE_ID_YAW_RANGE[0]) & (id_a <= PICK_SINGLE_YCB_AIRPLANE_ID_YAW_RANGE[1]))
    assert np.all(yaw_in_ranges(ood, PICK_SINGLE_YCB_AIRPLANE_OOD_YAW_RANGES))
    assert not np.any(yaw_in_ranges(id_a, PICK_SINGLE_YCB_AIRPLANE_OOD_YAW_RANGES))
