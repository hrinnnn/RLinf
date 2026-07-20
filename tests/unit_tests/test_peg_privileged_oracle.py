from __future__ import annotations

import numpy as np

from rlinf.envs.maniskill.peg_privileged_oracle import _normalize_delta


def test_joint_delta_normalization_maps_controller_bounds_to_unit_interval():
    values = _normalize_delta(
        np.asarray([-0.1, 0.0, 0.1], dtype=np.float32),
        np.full(3, -0.1, dtype=np.float32),
        np.full(3, 0.1, dtype=np.float32),
    )
    assert np.allclose(values, [-1.0, 0.0, 1.0])
