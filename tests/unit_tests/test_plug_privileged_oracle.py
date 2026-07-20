from __future__ import annotations

import numpy as np

from rlinf.envs.maniskill.plug_privileged_oracle import PlugOraclePlan, PlugChargerPrivilegedChunkOracle


def test_plug_oracle_plan_converts_targets_against_live_qpos():
    plan = PlugOraclePlan(
        actions=np.zeros((1, 8), dtype=np.float32), phase="reach", planning_succeeded=True,
        joint_targets=np.full((1, 7), 0.05, dtype=np.float32), gripper=-1.0,
    )
    qpos = np.concatenate([np.full(7, 0.03, dtype=np.float32), [0.04, 0.04]])
    action = plan.action_at(qpos, 0)
    assert np.allclose(action[:7], 0.2)
    assert action[-1] == -1.0


def test_plug_oracle_rejects_nonpositive_chunk_size():
    try:
        PlugChargerPrivilegedChunkOracle(chunk_size=0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("expected a ValueError")
