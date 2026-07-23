"""Privileged StackCube milestones used only for diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np


def _bool(value: Any) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value, dtype=bool).reshape(-1).any())


class StackCubeProgressState:
    def __init__(self) -> None:
        self.phi = 0.0

    def update(self, env: Any, info: dict[str, Any]) -> float:
        base = env.unwrapped
        cube_z = float(base.cubeA.pose.p.reshape(-1, 3)[0, 2].item())
        candidate = 0.0
        if _bool(info.get("is_cubeA_grasped", False)):
            candidate = 0.25
        if cube_z >= 0.07:
            candidate = max(candidate, 0.5)
        if _bool(info.get("is_cubeA_on_cubeB", False)):
            candidate = 0.75
        if _bool(info.get("success", False)):
            candidate = 1.0
        self.phi = max(self.phi, candidate)
        return self.phi
