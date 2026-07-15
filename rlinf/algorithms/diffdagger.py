# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DiffDAgger uncertainty calibration and intervention utilities.

The original DiffDAgger policy calibrates diffusion reconstruction loss on
in-distribution demonstrations, queries an expert above an empirical quantile,
and requires repeated threshold crossings before intervention.  This module
keeps those policy-independent mechanics separate from the OpenPI model and
RLinf worker plumbing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_calibration_scores(path: str | Path) -> np.ndarray:
    """Load finite scalar uncertainty scores from JSON, JSONL, NPY, or NPZ."""

    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"DiffDAgger calibration file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        values = np.load(path)
    elif suffix == ".npz":
        archive = np.load(path)
        if "scores" not in archive:
            raise ValueError(f"DiffDAgger NPZ must contain a 'scores' array: {path}")
        values = archive["scores"]
    elif suffix == ".json":
        payload = json.loads(path.read_text())
        values = payload.get("scores") if isinstance(payload, dict) else payload
    elif suffix == ".jsonl":
        records = [json.loads(line) for line in path.read_text().splitlines() if line]
        values = [
            record["score"] if isinstance(record, dict) else record
            for record in records
        ]
    else:
        raise ValueError(
            f"DiffDAgger calibration must be .json, .jsonl, .npy, or .npz, got: {path}"
        )

    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        raise ValueError("DiffDAgger calibration scores cannot be empty.")
    if not np.isfinite(scores).all():
        raise ValueError("DiffDAgger calibration scores must all be finite.")
    return scores


class EmpiricalCDF:
    """Small dependency-free empirical CDF matching DiffDAgger semantics."""

    def __init__(self, scores: np.ndarray | list[float]):
        values = np.asarray(scores, dtype=np.float64).reshape(-1)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError("EmpiricalCDF requires non-empty finite scores.")
        self.sorted_scores = np.sort(values)

    def probability(self, values: np.ndarray | torch.Tensor) -> np.ndarray:
        array = (
            values.detach().cpu().numpy()
            if isinstance(values, torch.Tensor)
            else np.asarray(values)
        )
        return np.searchsorted(self.sorted_scores, array, side="right") / len(
            self.sorted_scores
        )

    def quantile(self, q: float) -> float:
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"Quantile must be in [0, 1], got {q}.")
        index = min(int(len(self.sorted_scores) * q), len(self.sorted_scores) - 1)
        return float(self.sorted_scores[index])


@dataclass(frozen=True)
class DiffDAggerDecision:
    scores: torch.Tensor
    cdf_values: torch.Tensor
    threshold: float
    exceedances: torch.Tensor
    query_mask: torch.Tensor


class DiffDAggerQueryGate:
    """Apply a calibrated threshold and per-environment patience windows."""

    def __init__(
        self,
        calibration_scores: np.ndarray | list[float],
        *,
        alpha: float = 0.99,
        patience: int = 2,
        patience_window: int | None = None,
    ):
        if patience < 1:
            raise ValueError("DiffDAgger patience must be at least 1.")
        if patience_window is None:
            patience_window = patience
        if patience_window < patience:
            raise ValueError(
                "patience_window must be greater than or equal to patience."
            )

        self.cdf = EmpiricalCDF(calibration_scores)
        self.alpha = float(alpha)
        self.threshold = self.cdf.quantile(self.alpha)
        self.patience = int(patience)
        self.patience_window = int(patience_window)
        self._history: torch.Tensor | None = None

    def _ensure_history(self, batch_size: int, device: torch.device) -> None:
        if self._history is None or self._history.shape[0] != batch_size:
            self._history = torch.zeros(
                batch_size,
                self.patience_window,
                dtype=torch.bool,
                device=device,
            )
        elif self._history.device != device:
            self._history = self._history.to(device)

    def reset(self, reset_mask: torch.Tensor | None = None) -> None:
        if self._history is None:
            return
        if reset_mask is None:
            self._history.zero_()
            return
        mask = torch.as_tensor(
            reset_mask, dtype=torch.bool, device=self._history.device
        )
        if mask.ndim > 1:
            mask = mask.reshape(mask.shape[0], -1).any(dim=-1)
        if mask.numel() != self._history.shape[0]:
            raise ValueError(
                f"Reset mask batch {mask.numel()} does not match gate batch "
                f"{self._history.shape[0]}."
            )
        self._history[mask] = False

    def decide(
        self,
        scores: torch.Tensor,
        *,
        reset_mask: torch.Tensor | None = None,
    ) -> DiffDAggerDecision:
        scores = torch.as_tensor(scores, dtype=torch.float32).reshape(-1)
        if not torch.isfinite(scores).all():
            raise ValueError("DiffDAgger uncertainty scores must be finite.")
        self._ensure_history(scores.shape[0], scores.device)
        if reset_mask is not None:
            self.reset(reset_mask)

        exceedances = scores > self.threshold
        assert self._history is not None
        self._history = torch.roll(self._history, shifts=-1, dims=1)
        self._history[:, -1] = exceedances
        query_mask = self._history.sum(dim=1) >= self.patience
        cdf_values = torch.as_tensor(
            self.cdf.probability(scores), dtype=torch.float32, device=scores.device
        )
        return DiffDAggerDecision(
            scores=scores,
            cdf_values=cdf_values,
            threshold=self.threshold,
            exceedances=exceedances,
            query_mask=query_mask,
        )


def intervention_keep_mask(
    intervene_flags: torch.Tensor,
    *,
    reward_type: str,
) -> torch.Tensor:
    """Return the PPO mask that excludes expert-executed actions.

    ``intervene_flags`` may be action-dimension flags or per-action flags.  For
    chunk-level PPO, any intervention in a chunk excludes the macro transition.
    """

    flags = torch.as_tensor(intervene_flags, dtype=torch.bool)
    if reward_type == "chunk_level":
        leading_dims = flags.shape[:2] if flags.ndim >= 3 else flags.shape[:1]
        return ~flags.reshape(*leading_dims, -1).any(dim=-1, keepdim=True)
    return ~flags


def flow_matching_noisy_action_and_target(
    actions: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct the pi0/pi0.5 flow-matching input and velocity target."""

    if actions.shape != noise.shape:
        raise ValueError(
            f"Action/noise shapes differ: {actions.shape} vs {noise.shape}."
        )
    t = timesteps.to(device=actions.device, dtype=actions.dtype)
    while t.ndim < actions.ndim:
        t = t.unsqueeze(-1)
    return t * noise + (1.0 - t) * actions, noise - actions


def flow_matching_reconstruction_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    action_chunk: int,
    action_dim: int,
) -> torch.Tensor:
    """Return per-sample flow reconstruction MSE on environment actions."""

    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction/target shapes differ: {prediction.shape} vs {target.shape}."
        )
    error = (
        prediction[:, :action_chunk, :action_dim]
        - target[:, :action_chunk, :action_dim]
    ).float()
    return error.square().flatten(start_dim=1).mean(dim=1)


def combine_loss_masks(
    base_mask: torch.Tensor | None,
    additional_mask: torch.Tensor,
) -> torch.Tensor:
    additional_mask = additional_mask.to(torch.bool)
    if base_mask is None:
        return additional_mask
    return base_mask.to(torch.bool) & additional_mask


def _select_rows(
    student: torch.Tensor,
    expert: torch.Tensor,
    query_mask: torch.Tensor,
) -> torch.Tensor:
    if student.shape != expert.shape:
        raise ValueError(
            f"Student/expert tensor shapes differ: {student.shape} vs {expert.shape}."
        )
    mask = query_mask.to(device=student.device, dtype=torch.bool)
    mask = mask.reshape(mask.shape[0], *([1] * (student.ndim - 1)))
    return torch.where(mask, expert.to(student.device), student)


def merge_expert_interventions(
    student_actions: torch.Tensor,
    student_result: dict[str, Any],
    expert_actions: torch.Tensor,
    expert_result: dict[str, Any],
    query_mask: torch.Tensor,
    *,
    num_action_chunks: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Replace queried rows and attach expert labels for hybrid PPO/DAgger."""

    query_mask = torch.as_tensor(query_mask, dtype=torch.bool).reshape(-1)
    if query_mask.shape[0] != student_actions.shape[0]:
        raise ValueError("Query mask batch does not match student action batch.")

    merged_result = dict(student_result)
    merged_forward = dict(student_result.get("forward_inputs", {}))
    expert_forward = expert_result.get("forward_inputs", {})
    for key in ("action", "model_action"):
        if key in merged_forward and key in expert_forward:
            merged_forward[key] = _select_rows(
                merged_forward[key], expert_forward[key], query_mask
            )
    merged_result["forward_inputs"] = merged_forward
    merged_result["intervene_flags"] = query_mask[:, None].expand(-1, num_action_chunks)
    return (
        _select_rows(student_actions, expert_actions, query_mask),
        merged_result,
    )
