"""Privileged, monotonic progress labels for controlled PlugCharger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


PLUG_PROGRESS_LEVELS = {
    "ungrasped": 0.0,
    "grasped": 0.25,
    "lifted": 0.50,
    "prealigned": 0.75,
    "success": 1.0,
}


def _as_bool(value: Any) -> bool:
    if value is None:
        return False
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value, dtype=bool).reshape(-1).any())


def _scalar(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])


@dataclass
class PlugProgressState:
    """Episode-local milestone state. Create a new object after every reset."""

    grasped_once: bool = False
    lifted_once: bool = False
    prealigned_once: bool = False
    success_once: bool = False
    initial_base_z: float | None = None

    def update(self, env: Any, info: dict[str, Any] | None) -> float:
        base_env = env.unwrapped
        if self.initial_base_z is None:
            self.initial_base_z = _scalar(base_env.charger_base_pose.p[..., 2])
        self.grasped_once = self.grasped_once or _as_bool(base_env.agent.is_grasping(base_env.charger))
        base_z = _scalar(base_env.charger_base_pose.p[..., 2])
        self.lifted_once = self.lifted_once or (self.grasped_once and base_z >= self.initial_base_z + 0.03)
        distance, angle = base_env._compute_distance()
        self.prealigned_once = self.prealigned_once or (
            self.grasped_once and _scalar(distance) <= 0.06 and _scalar(angle) <= 0.35
        )
        self.success_once = self.success_once or _as_bool((info or {}).get("success"))
        if self.success_once:
            return PLUG_PROGRESS_LEVELS["success"]
        if self.prealigned_once:
            return PLUG_PROGRESS_LEVELS["prealigned"]
        if self.lifted_once:
            return PLUG_PROGRESS_LEVELS["lifted"]
        if self.grasped_once:
            return PLUG_PROGRESS_LEVELS["grasped"]
        return PLUG_PROGRESS_LEVELS["ungrasped"]
