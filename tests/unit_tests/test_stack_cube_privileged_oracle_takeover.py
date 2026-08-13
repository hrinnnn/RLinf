from types import SimpleNamespace

import numpy as np

from rlinf.envs.maniskill.stack_cube_privileged_oracle import StackCubePrivilegedChunkOracle


class TensorLike:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def env(grasped: bool, cube_z: float):
    agent = SimpleNamespace(is_grasping=lambda _cube: TensorLike([grasped]))
    cube = SimpleNamespace(pose=SimpleNamespace(p=TensorLike([[0.0, 0.0, cube_z]])))
    return SimpleNamespace(unwrapped=SimpleNamespace(agent=agent, cubeA=cube))


def test_takeover_initializes_from_current_grasp_stage() -> None:
    oracle = StackCubePrivilegedChunkOracle()
    assert oracle.initialize_from_state(env(False, 0.02)) == "reach"
    assert oracle.initialize_from_state(env(True, 0.05)) == "close"
    assert oracle.initialize_from_state(env(True, 0.08)) == "close"


def test_stable_grasp_hint_overrides_transient_grasp_predicate() -> None:
    oracle = StackCubePrivilegedChunkOracle()
    assert oracle.initialize_from_state(env(False, 0.08), grasped_hint=True) == "close"
