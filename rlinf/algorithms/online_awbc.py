"""Small, model-agnostic building blocks for online VFD + AWBC rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class FixedVFDThreshold:
    """A versioned but intentionally non-updating global VFD threshold."""

    threshold: float
    quantile: float
    calibration_count: int

    @classmethod
    def calibrate(
        cls, scores: Iterable[float], *, quantile: float = 0.95
    ) -> "FixedVFDThreshold":
        if not 0.0 < quantile < 1.0:
            raise ValueError("quantile must be strictly between zero and one")
        values = np.asarray(list(scores), dtype=np.float64).reshape(-1)
        if values.size == 0:
            raise ValueError("at least one VFD calibration score is required")
        if not np.isfinite(values).all():
            raise ValueError("VFD calibration scores must be finite")
        return cls(
            threshold=float(np.quantile(values, quantile)),
            quantile=float(quantile),
            calibration_count=int(values.size),
        )

    def decide(self, scores: torch.Tensor | Sequence[float]) -> torch.Tensor:
        """Return ``True`` exactly where the oracle owns the next chunk."""

        values = torch.as_tensor(scores, dtype=torch.float32).reshape(-1)
        if not torch.isfinite(values).all():
            raise ValueError("VFD scores must be finite")
        return values > self.threshold


@dataclass(frozen=True)
class ChunkControlDecision:
    """Diagnostics for one policy/expert arbitration point."""

    vfd_scores: torch.Tensor
    expert_mask: torch.Tensor
    threshold: float

    @property
    def controllers(self) -> tuple[str, ...]:
        return tuple("expert" if value else "policy" for value in self.expert_mask.tolist())


class FixedThresholdChunkController:
    """Choose the controller independently for every action chunk.

    Unlike sticky takeover gates, this object is deliberately stateless: a low
    VFD score immediately returns control to the policy on the next chunk.
    """

    def __init__(self, threshold: FixedVFDThreshold):
        self.threshold = threshold

    def decide(self, scores: torch.Tensor | Sequence[float]) -> ChunkControlDecision:
        values = torch.as_tensor(scores, dtype=torch.float32).reshape(-1)
        return ChunkControlDecision(
            vfd_scores=values,
            expert_mask=self.threshold.decide(values),
            threshold=self.threshold.threshold,
        )


class HysteresisChunkController:
    """Keep expert control until uncertainty is stably below a lower boundary."""

    def __init__(
        self,
        threshold: FixedVFDThreshold,
        *,
        return_ratio: float = 0.9,
        policy_release_streak: int = 2,
    ):
        if not 0.0 < return_ratio <= 1.0:
            raise ValueError("return_ratio must lie in (0, 1]")
        if policy_release_streak < 1:
            raise ValueError("policy_release_streak must be positive")
        self.threshold = threshold
        self.return_threshold = threshold.threshold * return_ratio
        self.policy_release_streak = policy_release_streak
        self._expert_active = False
        self._low_streak = 0

    def decide(self, scores: torch.Tensor | Sequence[float]) -> ChunkControlDecision:
        values = torch.as_tensor(scores, dtype=torch.float32).reshape(-1)
        if values.numel() != 1:
            raise ValueError("hysteresis control accepts exactly one VFD score")
        score = float(values.item())
        if not np.isfinite(score):
            raise ValueError("VFD score must be finite")
        if score > self.threshold.threshold:
            self._expert_active = True
            self._low_streak = 0
        elif self._expert_active and score < self.return_threshold:
            self._low_streak += 1
            if self._low_streak >= self.policy_release_streak:
                self._expert_active = False
                self._low_streak = 0
        elif self._expert_active:
            self._low_streak = 0
        return ChunkControlDecision(
            vfd_scores=values,
            expert_mask=torch.tensor([self._expert_active]),
            threshold=self.threshold.threshold,
        )


def uniformly_spaced_chunk_indices(
    num_chunks: int, *, samples_per_episode: int
) -> tuple[int, ...]:
    """Select equal-time calibration anchors without over-weighting long episodes."""

    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")
    if samples_per_episode <= 0:
        raise ValueError("samples_per_episode must be positive")
    count = min(num_chunks, samples_per_episode)
    indices = np.linspace(0, num_chunks - 1, num=count, dtype=np.int64)
    return tuple(int(index) for index in np.unique(indices))


def first_vfd_action_candidate(candidates: torch.Tensor) -> torch.Tensor:
    """Select the exact VFD reference sample used as the policy action."""

    if candidates.ndim != 4 or candidates.shape[1] < 1:
        raise ValueError("VFD candidates must have shape [B,C,H,A] with C >= 1")
    return candidates[:, 0]
