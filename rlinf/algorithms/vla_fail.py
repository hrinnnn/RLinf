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


def compact_two_camera_prefix_probe_tokens(
    *,
    num_images: int,
    prefix_embs: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Remove OpenPI's all-padding extra-view template from a two-camera probe.

    The policy can internally reserve a third 256-token image slot. For this
    controlled two-camera protocol that slot must be fully invalid; compacting
    it exposes the registered 512 vision + 200 language/state token layout
    without changing the underlying action forward.
    """

    image_tokens, language_tokens = 256, 200
    two_camera_tokens = 2 * image_tokens + language_tokens
    if num_images == 2 and prefix_embs.shape[1] == two_camera_tokens:
        indices = torch.arange(two_camera_tokens, device=prefix_embs.device)
    elif num_images == 3 and prefix_embs.shape[1] == 3 * image_tokens + language_tokens:
        dummy_start, dummy_end = 2 * image_tokens, 3 * image_tokens
        if torch.any(prefix_valid_mask[:, dummy_start:dummy_end]):
            raise RuntimeError("three-image prefix has valid extra-view tokens; cannot treat it as two-camera input")
        indices = torch.cat(
            (
                torch.arange(dummy_start, device=prefix_embs.device),
                torch.arange(dummy_end, prefix_embs.shape[1], device=prefix_embs.device),
            )
        )
    else:
        raise RuntimeError(
            "prefix probe expects either two real images or a fully padded third image template, "
            f"got images={num_images}, prefix_shape={tuple(prefix_embs.shape)}"
        )
    source_ids = torch.full((two_camera_tokens,), 2, dtype=torch.int8, device=prefix_embs.device)
    source_ids[:image_tokens] = 0
    source_ids[image_tokens : 2 * image_tokens] = 1
    return (
        prefix_embs.index_select(1, indices),
        prefix_output.index_select(1, indices),
        prefix_valid_mask.index_select(1, indices),
        source_ids,
    )


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


@dataclass(frozen=True)
class KNNStatistics:
    """L2-normalized ID feature bank for Deep kNN-style OOD scoring.

    The bank remains token-wise so it can score both one-token VLM probes and
    the ten Action Expert tokens without silently averaging action positions.
    ``k`` follows the official KNN-OOD convention: return the squared L2
    distance to the k-th nearest normalized ID feature.
    """

    bank: torch.Tensor
    k: int
    normalization_epsilon: float

    def validate(self) -> None:
        if self.bank.ndim != 3:
            raise ValueError("kNN bank must have shape [tokens, observations, features]")
        tokens, observations, features = self.bank.shape
        if tokens < 1 or observations < self.k or features < 1:
            raise ValueError("kNN bank has incompatible token, observation, or feature dimensions")
        if self.k < 1:
            raise ValueError("kNN k must be positive")
        if self.normalization_epsilon <= 0:
            raise ValueError("kNN normalization epsilon must be positive")

    def state_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "bank": self.bank.cpu(),
            "k": self.k,
            "normalization_epsilon": self.normalization_epsilon,
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, object]) -> "KNNStatistics":
        result = cls(
            bank=torch.as_tensor(payload["bank"], dtype=torch.float32),
            k=int(payload["k"]),
            normalization_epsilon=float(payload["normalization_epsilon"]),
        )
        result.validate()
        return result


def _l2_normalize(features: torch.Tensor, *, epsilon: float) -> torch.Tensor:
    if epsilon <= 0:
        raise ValueError("normalization epsilon must be positive")
    return features / torch.linalg.vector_norm(features, dim=-1, keepdim=True).clamp_min(epsilon)


def fit_knn_statistics(
    features: torch.Tensor, *, k: int = 10, normalization_epsilon: float = 1e-10
) -> KNNStatistics:
    """Fit the normalized feature bank used by official Deep kNN OOD scoring."""

    if features.ndim != 3:
        raise ValueError("kNN features must have shape [observations, tokens, features]")
    observations, tokens, hidden_dim = features.shape
    if observations < k or tokens < 1 or hidden_dim < 1:
        raise ValueError("kNN needs at least k observations and non-empty token features")
    if not torch.isfinite(features).all():
        raise ValueError("kNN features must be finite")
    bank = _l2_normalize(features.detach().to(device="cpu", dtype=torch.float32), epsilon=normalization_epsilon)
    result = KNNStatistics(bank=bank.permute(1, 0, 2).contiguous(), k=k, normalization_epsilon=normalization_epsilon)
    result.validate()
    return result


def knn_token_scores(features: torch.Tensor, statistics: KNNStatistics) -> torch.Tensor:
    """Return the official k-th normalized squared-L2 score for each token."""

    statistics.validate()
    if features.ndim != 3:
        raise ValueError("kNN features must have shape [batch, tokens, features]")
    batch, tokens, hidden_dim = features.shape
    if (tokens, hidden_dim) != (statistics.bank.shape[0], statistics.bank.shape[2]):
        raise ValueError("kNN feature shape does not match fitted bank")
    values = _l2_normalize(features.to(dtype=torch.float32), epsilon=statistics.normalization_epsilon)
    bank = statistics.bank.to(values.device)
    per_token = []
    for token in range(tokens):
        # Both vectors have unit norm: ||q-b||² = 2 - 2 qᵀb. This is exactly
        # IndexFlatL2 on normalized features, without requiring FAISS at runtime.
        distances = (2.0 - 2.0 * values[:, token] @ bank[token].transpose(0, 1)).clamp_min(0.0)
        per_token.append(distances.kthvalue(statistics.k, dim=-1).values)
    scores = torch.stack(per_token, dim=-1)
    if scores.shape != (batch, tokens) or not torch.isfinite(scores).all():
        raise RuntimeError("kNN produced invalid token scores")
    return scores


def knn_score(features: torch.Tensor, statistics: KNNStatistics) -> torch.Tensor:
    """Conservatively aggregate token-wise kNN OOD scores with a maximum."""

    return knn_token_scores(features, statistics).amax(dim=-1)


@dataclass(frozen=True)
class PCAResidualStatistics:
    """Per-token principal subspaces for a ViM-inspired residual baseline.

    This intentionally stores only the feature-space residual. Full ViM also
    needs classifier logits and a classifier head, neither of which exists at
    the pi0.5 VLM-to-Action bridge.
    """

    mean: torch.Tensor
    principal_components: torch.Tensor
    principal_dim: int
    num_observations: int

    def validate(self) -> None:
        if self.mean.ndim != 2 or self.principal_components.ndim != 3:
            raise ValueError("PCA residual statistics have invalid rank")
        tokens, features = self.mean.shape
        if self.principal_components.shape[:2] != (tokens, features):
            raise ValueError("PCA principal components have incompatible shape")
        if self.principal_components.shape[2] != self.principal_dim:
            raise ValueError("PCA principal component count does not match principal_dim")
        if not 0 < self.principal_dim <= features or self.num_observations < 2:
            raise ValueError("PCA residual statistics have invalid dimensions")

    def state_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "mean": self.mean.cpu(),
            "principal_components": self.principal_components.cpu(),
            "principal_dim": self.principal_dim,
            "num_observations": self.num_observations,
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, object]) -> "PCAResidualStatistics":
        result = cls(
            mean=torch.as_tensor(payload["mean"], dtype=torch.float32),
            principal_components=torch.as_tensor(payload["principal_components"], dtype=torch.float32),
            principal_dim=int(payload["principal_dim"]),
            num_observations=int(payload["num_observations"]),
        )
        result.validate()
        return result


def vim_default_principal_dim(hidden_dim: int) -> int:
    """Use the official ViM dimension rule before removing its logit component."""

    if hidden_dim < 2:
        raise ValueError("PCA residual needs at least two hidden dimensions")
    if hidden_dim >= 2048:
        return 1000
    if hidden_dim >= 768:
        return 512
    return hidden_dim // 2


def fit_pca_residual_statistics(
    features: torch.Tensor, *, principal_dim: int | None = None
) -> PCAResidualStatistics:
    """Fit ViM's principal feature subspace without its classifier-logit term."""

    if features.ndim != 3:
        raise ValueError("PCA residual features must have shape [observations, tokens, features]")
    observations, tokens, hidden_dim = features.shape
    if observations < 2 or tokens < 1 or hidden_dim < 2:
        raise ValueError("PCA residual needs at least two observations and two hidden dimensions")
    if not torch.isfinite(features).all():
        raise ValueError("PCA residual features must be finite")
    dimension = vim_default_principal_dim(hidden_dim) if principal_dim is None else principal_dim
    if not 0 < dimension <= hidden_dim:
        raise ValueError("PCA principal_dim must lie in [1, hidden_dim]")
    values = features.detach().to(device="cpu", dtype=torch.float64)
    mean = values.mean(dim=0)
    centered = values - mean.unsqueeze(0)
    covariance = torch.einsum("nth,ntk->thk", centered, centered) / observations
    components = []
    for token in range(tokens):
        _eigenvalues, eigenvectors = torch.linalg.eigh(covariance[token])
        components.append(eigenvectors[:, -dimension:])
    result = PCAResidualStatistics(
        mean=mean.to(dtype=torch.float32),
        principal_components=torch.stack(components, dim=0).to(dtype=torch.float32),
        principal_dim=dimension,
        num_observations=observations,
    )
    result.validate()
    return result


def pca_residual_token_scores(features: torch.Tensor, statistics: PCAResidualStatistics) -> torch.Tensor:
    """Return L2 distance outside each token's ID principal subspace."""

    statistics.validate()
    if features.ndim != 3:
        raise ValueError("PCA residual features must have shape [batch, tokens, features]")
    batch, tokens, hidden_dim = features.shape
    if (tokens, hidden_dim) != tuple(statistics.mean.shape):
        raise ValueError("PCA residual feature shape does not match fitted statistics")
    values = features.to(dtype=torch.float32)
    mean = statistics.mean.to(values.device)
    components = statistics.principal_components.to(values.device)
    centered = values - mean.unsqueeze(0)
    coordinates = torch.einsum("bth,thr->btr", centered, components)
    reconstruction = torch.einsum("btr,thr->bth", coordinates, components)
    scores = torch.linalg.vector_norm(centered - reconstruction, dim=-1)
    if scores.shape != (batch, tokens) or not torch.isfinite(scores).all():
        raise RuntimeError("PCA residual produced invalid token scores")
    return scores


def pca_residual_score(features: torch.Tensor, statistics: PCAResidualStatistics) -> torch.Tensor:
    """Conservatively aggregate token-wise PCA residual scores with a maximum."""

    return pca_residual_token_scores(features, statistics).amax(dim=-1)


@dataclass(frozen=True)
class TokenwisePCAResidualStatistics:
    """Independent PCA subspaces and residual scales for prefix tokens.

    Unlike :class:`PCAResidualStatistics`, this payload explicitly carries the
    ID residual scale for every token position and an eligibility mask.  It is
    intended for token-wise TopK aggregation, where an invalid/padded position
    must never contribute a score.
    """

    mean: torch.Tensor
    principal_components: torch.Tensor
    residual_mean: torch.Tensor
    residual_std: torch.Tensor
    eligible_tokens: torch.Tensor
    observation_counts: torch.Tensor
    principal_dim: int
    min_observations: int

    def validate(self) -> None:
        if self.mean.ndim != 2 or self.principal_components.ndim != 3:
            raise ValueError("token-wise PCA mean/components have invalid rank")
        tokens, hidden_dim = self.mean.shape
        if self.principal_components.shape != (tokens, hidden_dim, self.principal_dim):
            raise ValueError("token-wise PCA principal component shape is invalid")
        if self.residual_mean.shape != (tokens,) or self.residual_std.shape != (tokens,):
            raise ValueError("token-wise PCA residual statistics have invalid shape")
        if self.eligible_tokens.shape != (tokens,) or self.eligible_tokens.dtype != torch.bool:
            raise ValueError("token-wise PCA eligibility mask has invalid shape or dtype")
        if self.observation_counts.shape != (tokens,):
            raise ValueError("token-wise PCA observation counts have invalid shape")
        if not 0 < self.principal_dim <= hidden_dim:
            raise ValueError("token-wise PCA principal_dim is invalid")
        if self.min_observations < self.principal_dim + 1:
            raise ValueError("token-wise PCA min_observations must exceed principal_dim")
        if torch.any(self.observation_counts < 0):
            raise ValueError("token-wise PCA observation counts must be non-negative")
        if torch.any(self.eligible_tokens & (self.observation_counts < self.min_observations)):
            raise ValueError("eligible token has too few ID observations")
        if not all(torch.isfinite(value).all() for value in (self.mean, self.principal_components, self.residual_mean, self.residual_std)):
            raise ValueError("token-wise PCA statistics must be finite")
        if torch.any(self.residual_std < 0):
            raise ValueError("token-wise PCA residual std must be non-negative")

    def state_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "mean": self.mean.cpu(),
            "principal_components": self.principal_components.cpu(),
            "residual_mean": self.residual_mean.cpu(),
            "residual_std": self.residual_std.cpu(),
            "eligible_tokens": self.eligible_tokens.cpu(),
            "observation_counts": self.observation_counts.cpu(),
            "principal_dim": self.principal_dim,
            "min_observations": self.min_observations,
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, object]) -> "TokenwisePCAResidualStatistics":
        result = cls(
            mean=torch.as_tensor(payload["mean"], dtype=torch.float32),
            principal_components=torch.as_tensor(payload["principal_components"], dtype=torch.float32),
            residual_mean=torch.as_tensor(payload["residual_mean"], dtype=torch.float32),
            residual_std=torch.as_tensor(payload["residual_std"], dtype=torch.float32),
            eligible_tokens=torch.as_tensor(payload["eligible_tokens"], dtype=torch.bool),
            observation_counts=torch.as_tensor(payload["observation_counts"], dtype=torch.int64),
            principal_dim=int(payload["principal_dim"]),
            min_observations=int(payload["min_observations"]),
        )
        result.validate()
        return result


def fit_tokenwise_pca_residual_statistics(
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    principal_dim: int,
    min_observations: int = 1001,
    compute_device: torch.device | str = "cpu",
) -> TokenwisePCAResidualStatistics:
    """Fit independent PCA subspaces with position-specific ID residual scales.

    This exact in-memory helper deliberately serves a *token block*.  The
    airplane asset builder feeds it small blocks from durable feature shards,
    which keeps the full prefix tensor and all 700+ PCA bases out of RAM.
    """

    if features.ndim != 3:
        raise ValueError("token-wise PCA features must have shape [observations, tokens, features]")
    observations, tokens, hidden_dim = features.shape
    if valid_mask.shape != (observations, tokens):
        raise ValueError("token-wise PCA valid_mask must have shape [observations, tokens]")
    if not 0 < principal_dim <= hidden_dim:
        raise ValueError("token-wise PCA principal_dim must lie in [1, hidden_dim]")
    if min_observations < principal_dim + 1:
        raise ValueError("token-wise PCA min_observations must exceed principal_dim")
    if not torch.isfinite(features).all():
        raise ValueError("token-wise PCA features must be finite")

    device = torch.device(compute_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("token-wise PCA requested CUDA but CUDA is unavailable")
    # Fitting runs in FP64, but only a small token block is resident. The
    # caller selects CPU for portable tests or an idle GPU for the many
    # independent rank-1000 factorizations used by the airplane protocol.
    values = features.detach().to(device=device, dtype=torch.float64)
    mask = valid_mask.detach().to(device=device, dtype=torch.bool)
    counts = mask.sum(dim=0, dtype=torch.int64)
    eligible = counts >= min_observations
    mean = torch.zeros((tokens, hidden_dim), dtype=torch.float64, device=device)
    components = torch.zeros((tokens, hidden_dim, principal_dim), dtype=torch.float64, device=device)
    residual_mean = torch.zeros(tokens, dtype=torch.float64, device=device)
    residual_std = torch.zeros(tokens, dtype=torch.float64, device=device)

    for token in torch.nonzero(eligible, as_tuple=False).flatten().tolist():
        token_values = values[mask[:, token], token]
        token_mean = token_values.mean(dim=0)
        centered = token_values - token_mean
        covariance = centered.transpose(0, 1) @ centered / token_values.shape[0]
        _eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        token_components = eigenvectors[:, -principal_dim:]
        residual = torch.linalg.vector_norm(
            centered - (centered @ token_components) @ token_components.transpose(0, 1), dim=-1
        )
        mean[token] = token_mean
        components[token] = token_components
        residual_mean[token] = residual.mean()
        residual_std[token] = residual.std(unbiased=False)

    result = TokenwisePCAResidualStatistics(
        mean=mean.cpu().to(dtype=torch.float32),
        principal_components=components.cpu().to(dtype=torch.float32),
        residual_mean=residual_mean.cpu().to(dtype=torch.float32),
        residual_std=residual_std.cpu().to(dtype=torch.float32),
        eligible_tokens=eligible.cpu(),
        observation_counts=counts.cpu(),
        principal_dim=principal_dim,
        min_observations=min_observations,
    )
    result.validate()
    return result


def tokenwise_pca_residual_scores(
    features: torch.Tensor, statistics: TokenwisePCAResidualStatistics
) -> torch.Tensor:
    """Return unscaled residuals; ineligible positions are set to zero."""

    statistics.validate()
    if features.ndim != 3:
        raise ValueError("token-wise PCA features must have shape [batch, tokens, features]")
    batch, tokens, hidden_dim = features.shape
    if (tokens, hidden_dim) != tuple(statistics.mean.shape):
        raise ValueError("token-wise PCA feature shape does not match fitted statistics")
    values = features.to(dtype=torch.float32)
    mean = statistics.mean.to(values.device)
    components = statistics.principal_components.to(values.device)
    centered = values - mean.unsqueeze(0)
    coordinates = torch.einsum("bth,thr->btr", centered, components)
    reconstruction = torch.einsum("btr,thr->bth", coordinates, components)
    residual = torch.linalg.vector_norm(centered - reconstruction, dim=-1)
    eligible = statistics.eligible_tokens.to(values.device)
    residual = torch.where(eligible.unsqueeze(0), residual, torch.zeros_like(residual))
    if residual.shape != (batch, tokens) or not torch.isfinite(residual).all():
        raise RuntimeError("token-wise PCA residual produced invalid scores")
    return residual


def tokenwise_pca_z_scores(
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    statistics: TokenwisePCAResidualStatistics,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Standardize residuals and mark padding/ineligible tokens as ``-inf``."""

    if epsilon <= 0:
        raise ValueError("token-wise PCA epsilon must be positive")
    if valid_mask.shape != tuple(features.shape[:2]):
        raise ValueError("token-wise PCA valid_mask does not match features")
    residual = tokenwise_pca_residual_scores(features, statistics)
    mean = statistics.residual_mean.to(residual.device)
    std = statistics.residual_std.to(residual.device).clamp_min(epsilon)
    z_scores = (residual - mean.unsqueeze(0)) / std.unsqueeze(0)
    usable = valid_mask.to(device=residual.device, dtype=torch.bool) & statistics.eligible_tokens.to(residual.device)
    return torch.where(usable, z_scores, torch.full_like(z_scores, -torch.inf))


def tokenwise_topk_mean(
    scores: torch.Tensor, *, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return TopK mean and token indices, rejecting rows with too few tokens."""

    if scores.ndim != 2:
        raise ValueError("token-wise TopK scores must have shape [batch, tokens]")
    if k < 1 or k > scores.shape[1]:
        raise ValueError("token-wise TopK k is outside the token dimension")
    usable = torch.isfinite(scores)
    if torch.any(usable.sum(dim=-1) < k):
        raise ValueError("not enough valid token-wise PCA scores for requested TopK")
    values, indices = torch.topk(scores, k=k, dim=-1)
    return values.mean(dim=-1), indices


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
