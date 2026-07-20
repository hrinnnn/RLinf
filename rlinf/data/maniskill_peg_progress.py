"""Convert PegInsertionSide privileged milestones into AWBC progress labels."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


PEG_PROGRESS_LEVELS = {
    "ungrasped": 0.0,
    "grasped": 0.25,
    "prealigned": 0.50,
    "partially_inserted": 0.75,
    "success": 1.0,
}


def _as_bool(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value, dtype=bool).reshape(-1).any())


def peg_privileged_phi(info: dict[str, Any] | None) -> float:
    """Return the highest monotonic Peg milestone observed in ``info``.

    The environment's ``*_once`` keys are preferred so transient contact loss
    does not turn a completed grasp or alignment into artificial regress.
    """

    info = info or {}
    if _as_bool(info.get("success_once", info.get("success_current", info.get("success")))):
        return PEG_PROGRESS_LEVELS["success"]
    if _as_bool(info.get("partial_insert_once", info.get("partial_insert_current"))):
        return PEG_PROGRESS_LEVELS["partially_inserted"]
    if _as_bool(info.get("prealign_once", info.get("prealigned_current"))):
        return PEG_PROGRESS_LEVELS["prealigned"]
    if _as_bool(
        info.get("consecutive_grasp_once", info.get("consecutive_grasp_current"))
    ):
        return PEG_PROGRESS_LEVELS["grasped"]
    return PEG_PROGRESS_LEVELS["ungrasped"]
