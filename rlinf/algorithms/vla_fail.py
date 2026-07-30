"""Core, training-free failure scores from the VLA-FAIL protocol.

The implementation deliberately keeps statistics, calibration, and temporal
action consistency separate from model/runtime code.  This lets an evaluator
use exactly the same scores for a simulated rollout and for a robot log.

The terminology follows VLA-FAIL: LLMD is a token-wise squared Mahalanobis
distance on the Action Expert feature immediately before the action head;
ACC is action-chunk consistency on predicted *absolute end-effector* points.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Iterable, Mapping, Sequence

import torch


def resolve_feature_probe_indices(
    num_layers: int, fractions: Sequence[float]
) -> tuple[int, ...]:
    """Map human-readable layer fractions to stable zero-based block indices."""

    if num_layers < 1:
        raise ValueError("feature probe needs at least one transformer block")
    indices = []
    for fraction in fractions:
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"feature probe fraction must be in (0, 1], got {fraction}")
        index = min(num_layers - 1, max(0, ceil(num_layers * fraction) - 1))
        if index not in indices:
            indices.append(index)
    return tuple(indices)


def pool_valid_prefix_tokens(hidden_states: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool only valid VLM prefix tokens into one fixed-shape feature."""

    if hidden_states.ndim != 3:
        raise ValueError(f"VLM hidden states must be [B,T,D], got {tuple(hidden_states.shape)}")
    if pad_mask.ndim != 2 or tuple(pad_mask.shape) != tuple(hidden_states.shape[:2]):
        raise ValueError(
            "VLM prefix mask must match hidden-state batch and token dimensions: "
            f"hidden={tuple(hidden_states.shape)}, mask={tuple(pad_mask.shape)}"
        )
    weights = pad_mask.to(device=hidden_states.device, dtype=hidden_states.dtype).unsqueeze(-1)
    counts = weights.sum(dim=1, keepdim=True)
    if torch.any(counts <= 0):
        raise ValueError("every VLM prefix must contain at least one valid token")
    return (hidden_states * weights).sum(dim=1, keepdim=True) / counts


@dataclass(frozen=True)
class LLMDStatistics:
    """One Gaussian feature model per action token."""

    mean: torch.Tensor
    precision: torch.Tensor
    ridge: float
    num_observations: int

    def validate(self) -> None:
        if self.mean.ndim != 2:
            raise ValueError("LLMD mean must have shape [tokens, features]")
        tokens, features = self.mean.shape
        if self.precision.shape != (tokens, features, features):
            raise ValueError("LLMD precision has incompatible token/feature shape")
        if self.ridge <= 0:
            raise ValueError("LLMD ridge must be positive")
        if self.num_observations < 2:
            raise ValueError("LLMD needs at least two feature observations")

    def state_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "mean": self.mean.cpu(),
            "precision": self.precision.cpu(),
            "ridge": self.ridge,
            "num_observations": self.num_observations,
        }

    @classmethod
    def from_state_dict(cls, payload: dict[str, object]) -> "LLMDStatistics":
        result = cls(
            mean=torch.as_tensor(payload["mean"], dtype=torch.float64),
            precision=torch.as_tensor(payload["precision"], dtype=torch.float64),
            ridge=float(payload["ridge"]),
            num_observations=int(payload["num_observations"]),
        )
        result.validate()
        return result


def fit_llmd_statistics(
    features: torch.Tensor, *, ridge: float = 1e-6
) -> LLMDStatistics:
    """Fit the paper's token-wise Gaussian models.

    ``features`` has shape ``[observations, action_tokens, hidden_dim]``.  The
    covariance uses the population convention in the paper's empirical
    distribution; ``ridge * I`` makes its inverse well defined.
    """

    if features.ndim != 3:
        raise ValueError("LLMD features must have shape [observations, tokens, features]")
    observations, tokens, hidden_dim = features.shape
    if observations < 2:
        raise ValueError("LLMD needs at least two feature observations")
    if tokens < 1 or hidden_dim < 1:
        raise ValueError("LLMD features must include at least one token and dimension")
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    if not torch.isfinite(features).all():
        raise ValueError("LLMD features must be finite")

    values = features.detach().to(device="cpu", dtype=torch.float64)
    mean = values.mean(dim=0)
    centered = values - mean.unsqueeze(0)
    covariance = torch.einsum("nth,ntk->thk", centered, centered) / observations
    covariance = covariance + ridge * torch.eye(hidden_dim, dtype=torch.float64).expand(
        tokens, -1, -1
    )
    precision = torch.linalg.inv(covariance)
    result = LLMDStatistics(mean, precision, float(ridge), int(observations))
    result.validate()
    return result


def llmd_token_scores(features: torch.Tensor, statistics: LLMDStatistics) -> torch.Tensor:
    """Return squared Mahalanobis distance for every action token."""

    statistics.validate()
    if features.ndim != 3:
        raise ValueError("LLMD features must have shape [batch, tokens, features]")
    batch, tokens, hidden_dim = features.shape
    if (tokens, hidden_dim) != tuple(statistics.mean.shape):
        raise ValueError(
            "LLMD feature shape does not match fitted statistics: "
            f"got {(tokens, hidden_dim)}, expected {tuple(statistics.mean.shape)}"
        )
    values = features.to(dtype=torch.float64)
    mean = statistics.mean.to(values.device)
    precision = statistics.precision.to(values.device)
    delta = values - mean.unsqueeze(0)
    scores = torch.einsum("bth,thk,btk->bt", delta, precision, delta)
    if scores.shape != (batch, tokens) or not torch.isfinite(scores).all():
        raise RuntimeError("LLMD produced invalid token scores")
    return scores.to(dtype=torch.float32)


def llmd_score(features: torch.Tensor, statistics: LLMDStatistics) -> torch.Tensor:
    """VLA-FAIL's conservative aggregation: maximum score over action tokens."""

    return llmd_token_scores(features, statistics).amax(dim=-1)


def fixed_gaussian_prior(
    *, action_horizon: int, action_dim: int, seed: int, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Create one fixed Gaussian action prior shared by every LLMD query."""

    if action_horizon < 1 or action_dim < 1:
        raise ValueError("action horizon and action dimension must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn((1, action_horizon, action_dim), generator=generator)
    return noise.to(device=device)


def constant_split_conformal_threshold(
    successful_trajectory_scores: Iterable[Iterable[float]], *, delta: float = 0.05
) -> float:
    """VLA-FAIL/FIPER constant band from successful trajectory maxima.

    This is the finite-sample split-conformal order statistic, not NumPy's
    interpolated percentile.  With 20 rollouts and ``delta=.05``, it is the
    largest successful trajectory maximum, exactly as the paper protocol.
    """

    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie strictly between zero and one")
    maxima = []
    for scores in successful_trajectory_scores:
        trace = [float(value) for value in scores]
        if not trace:
            raise ValueError("every calibration trajectory must contain a score")
        if not torch.isfinite(torch.tensor(trace)).all().item():
            raise ValueError("calibration scores must be finite")
        maxima.append(max(trace))
    if not maxima:
        raise ValueError("need at least one successful calibration trajectory")
    maxima.sort()
    rank = min(len(maxima), ceil((len(maxima) + 1) * (1.0 - delta)))
    return float(maxima[rank - 1])


def velocity_normalized_acc(
    previous_points: torch.Tensor,
    current_points: torch.Tensor,
    *,
    execute_horizon: int,
    min_velocity: float,
    previous_ema: float | None = None,
    ema_alpha: float = 0.9,
) -> tuple[float, float]:
    """Compute VLA-FAIL ACC for two overlapping EEF prediction chunks.

    Inputs are absolute EEF positions ``[H, 3]``.  The old suffix and new
    prefix overlap by ``H - execute_horizon`` points.  Returns ``(raw, ema)``.
    """

    if previous_points.ndim != 2 or current_points.ndim != 2:
        raise ValueError("ACC points must have shape [horizon, 3]")
    if previous_points.shape != current_points.shape or previous_points.shape[1] != 3:
        raise ValueError("ACC needs equal [horizon, 3] predicted EEF point chunks")
    horizon = previous_points.shape[0]
    if not 0 < execute_horizon < horizon:
        raise ValueError("execute_horizon must be in [1, horizon - 1]")
    if min_velocity <= 0:
        raise ValueError("min_velocity must be positive")
    if not 0.0 <= ema_alpha < 1.0:
        raise ValueError("ema_alpha must lie in [0, 1)")

    old_suffix = previous_points[execute_horizon:].to(dtype=torch.float64)
    new_prefix = current_points[: horizon - execute_horizon].to(dtype=torch.float64)
    # VLA-FAIL Eq. (7) computes the range on the current overlapping prefix,
    # not on the entire newly predicted horizon.
    velocity = torch.clamp(
        new_prefix.amax(dim=0) - new_prefix.amin(dim=0),
        min=min_velocity,
    )
    raw = float(((old_suffix - new_prefix).abs() / velocity).mean().item())
    ema = raw if previous_ema is None else ema_alpha * previous_ema + (1.0 - ema_alpha) * raw
    return raw, float(ema)


def failure_alert(
    *, llmd_value: float, llmd_threshold: float, acc_value: float | None, acc_threshold: float | None
) -> bool:
    """Paper fusion: an alarm when either LLMD or ACC exceeds its band."""

    if llmd_value >= llmd_threshold:
        return True
    return acc_value is not None and acc_threshold is not None and acc_value >= acc_threshold


def assert_threshold_statistics_compatible(
    threshold_payload: Mapping[str, Any], statistics_sha256: str
) -> None:
    """Reject thresholds calibrated from a different persistent LLMD asset.

    Legacy files without a digest remain readable. Newly created calibration
    files carry the digest so a later experiment cannot accidentally pair a
    threshold with statistics from another checkpoint or feature distribution.
    """

    recorded = threshold_payload.get("llmd_statistics_sha256")
    if recorded is not None and str(recorded) != statistics_sha256:
        raise ValueError(
            "Threshold was calibrated from different LLMD statistics: "
            f"threshold={recorded}, current={statistics_sha256}"
        )
