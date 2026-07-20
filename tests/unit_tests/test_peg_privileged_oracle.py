from __future__ import annotations

import numpy as np

from rlinf.envs.maniskill import peg_privileged_oracle
from rlinf.envs.maniskill.peg_privileged_oracle import (
    PegOraclePlan,
    PegPrivilegedChunkOracle,
    _normalize_delta,
)


def test_joint_delta_normalization_maps_controller_bounds_to_unit_interval():
    values = _normalize_delta(
        np.asarray([-0.1, 0.0, 0.1], dtype=np.float32),
        np.full(3, -0.1, dtype=np.float32),
        np.full(3, 0.1, dtype=np.float32),
    )
    assert np.allclose(values, [-1.0, 0.0, 1.0])


def test_oracle_uses_cached_reach_pose_before_switching_to_grasp(monkeypatch):
    class Pose:
        def __init__(self, point):
            self.p = np.asarray([point], dtype=np.float32)

    class Agent:
        tcp = type("Tcp", (), {"pose": Pose([0.01, 0.0, 0.0])})()

        @staticmethod
        def is_grasping(*_args, **_kwargs):
            return False

    env = type("Env", (), {"agent": Agent(), "peg": object()})()
    oracle = PegPrivilegedChunkOracle(chunk_size=10)
    reach_pose = Pose([0.0, 0.0, 0.0])
    grasp_pose = Pose([0.0, 0.0, 0.05])
    oracle._reach_pose = reach_pose
    oracle._grasp_pose = grasp_pose
    oracle._peg_init_pose = object()
    monkeypatch.setattr(oracle, "_initialize_reference_poses", lambda _env: None)
    monkeypatch.setattr(
        peg_privileged_oracle,
        "_load_motion_planning_symbols",
        lambda: (None, None, None, None),
    )

    target, gripper, phase = oracle._target(env)

    assert target is grasp_pose
    assert gripper == -1.0
    assert phase == "grasp"
    assert oracle._phase == "grasp"


def test_oracle_converts_joint_target_against_live_qpos():
    plan = PegOraclePlan(
        actions=np.zeros((1, 8), dtype=np.float32),
        phase="reach",
        planning_succeeded=True,
        joint_targets=np.full((1, 7), 0.05, dtype=np.float32),
        gripper=-1.0,
    )
    qpos = np.concatenate([np.full(7, 0.03, dtype=np.float32), [0.04, 0.04]])

    action = plan.action_at(qpos, 0)

    assert np.allclose(action[:7], 0.2)
    assert action[-1] == -1.0
