"""Persistent online AWBC buffer primitives.

This module is intentionally independent of the older manifest-wide AWBC
samplers.  An online round owns a small set of decision-aligned chunks and
mixes them with randomly anchored, complete expert demonstrations.  The
online weights are computed once at admission time and are never rescaled by a
particular minibatch.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import torch


@dataclass(frozen=True)
class OnlineChunk:
    """A single persisted 10-step decision anchor eligible for AWBC."""

    episode_id: int
    frame_index: int
    next_frame_index: int
    phi: float | None
    phi_next: float | None
    valid: bool
    episode_length_chunks: int
    controller: str
    weight: float

    def validate(self, *, horizon: int = 10) -> None:
        if self.episode_id < 0 or self.frame_index < 0:
            raise ValueError("online chunk indices must be non-negative")
        if self.next_frame_index - self.frame_index != horizon:
            raise ValueError("online chunk must cover exactly one full horizon")
        if self.episode_length_chunks <= 0:
            raise ValueError("episode_length_chunks must be positive")
        if self.controller not in {"policy", "expert"}:
            raise ValueError("controller must be policy or expert")
        if self.weight < 0 or not math.isfinite(self.weight):
            raise ValueError("online chunk weight must be finite and non-negative")
        if self.valid and (self.phi is None or self.phi_next is None):
            raise ValueError("valid online chunk requires phi and phi_next")


@dataclass(frozen=True)
class ExpertAnchor:
    """A 10-step training anchor selected from a permanent expert trajectory."""

    episode_id: int
    frame_index: int
    next_frame_index: int

    def validate(self, *, horizon: int = 10) -> None:
        if self.episode_id < 0 or self.frame_index < 0:
            raise ValueError("expert anchor indices must be non-negative")
        if self.next_frame_index - self.frame_index != horizon:
            raise ValueError("expert anchor must cover exactly one full horizon")


@dataclass(frozen=True)
class ActiveAWBCBatch:
    """One round's no-replacement online/expert pairing."""

    online: tuple[OnlineChunk, ...]
    expert: tuple[ExpertAnchor, ...]

    def __post_init__(self) -> None:
        if not self.online:
            raise ValueError("active AWBC batch requires at least one online chunk")
        if len(self.online) != len(self.expert):
            raise ValueError("active AWBC batch must contain equal online and expert anchors")

    @property
    def size(self) -> int:
        return len(self.online) + len(self.expert)


def persistent_flux_weight(
    *,
    phi: float | None,
    phi_next: float | None,
    valid: bool,
    episode_length_chunks: int,
    mean_episode_length_chunks: float,
    progress_threshold: float = 0.01,
    soft_scale: float = 0.01,
) -> float:
    """Return a one-time Flux-style quality weight without minibatch scaling.

    Negative and invalid estimates are rejected.  Small positive progress is
    linearly softened; clear progress receives the episode-length factor.
    This keeps the numerical identity of an admitted chunk stable across
    future replay samples.
    """

    if episode_length_chunks <= 0 or mean_episode_length_chunks <= 0:
        raise ValueError("episode lengths must be positive")
    if progress_threshold < 0 or soft_scale <= 0:
        raise ValueError("progress_threshold must be non-negative and soft_scale positive")
    if not valid or phi is None or phi_next is None:
        return 0.0
    delta = float(phi_next) - float(phi)
    if not math.isfinite(delta) or delta < 0:
        return 0.0
    length_factor = float(episode_length_chunks) / float(mean_episode_length_chunks)
    if delta > progress_threshold:
        return length_factor
    return length_factor * min(1.0, delta / soft_scale)


def quality_online_chunks(
    chunks: Iterable[OnlineChunk], *, minimum_weight: float = 0.1
) -> tuple[OnlineChunk, ...]:
    """Apply the immutable admission gate without discarding raw trajectory data."""

    if minimum_weight < 0:
        raise ValueError("minimum_weight must be non-negative")
    admitted: list[OnlineChunk] = []
    for chunk in chunks:
        chunk.validate()
        if chunk.valid and chunk.weight >= minimum_weight:
            admitted.append(chunk)
    return tuple(admitted)


def sample_active_batch(
    online_chunks: Sequence[OnlineChunk],
    expert_episode_lengths: dict[int, int],
    *,
    generator: torch.Generator,
    max_online_chunks: int = 32,
    horizon: int = 10,
) -> ActiveAWBCBatch:
    """Draw K online chunks and K hierarchical expert anchors without replacement.

    Expert trajectories are selected uniformly before selecting a valid anchor
    inside each trajectory.  Repeating an expert trajectory in the same active
    batch is allowed because the permanent expert dataset has only 128 episodes;
    online chunks themselves are never repeated within the batch.
    """

    if max_online_chunks <= 0:
        raise ValueError("max_online_chunks must be positive")
    if not online_chunks:
        raise ValueError("cannot construct an active batch without online chunks")
    eligible_episodes = sorted(
        episode_id
        for episode_id, length in expert_episode_lengths.items()
        if int(length) >= horizon
    )
    if not eligible_episodes:
        raise ValueError("expert replay has no trajectory with a full anchor")

    k = min(len(online_chunks), max_online_chunks)
    online_order = torch.randperm(len(online_chunks), generator=generator)[:k].tolist()
    online = tuple(online_chunks[index] for index in online_order)
    for chunk in online:
        chunk.validate(horizon=horizon)

    expert: list[ExpertAnchor] = []
    for _ in range(k):
        episode_id = eligible_episodes[
            int(torch.randint(len(eligible_episodes), (1,), generator=generator).item())
        ]
        length = int(expert_episode_lengths[episode_id])
        start = int(torch.randint(length - horizon + 1, (1,), generator=generator).item())
        anchor = ExpertAnchor(episode_id, start, start + horizon)
        anchor.validate(horizon=horizon)
        expert.append(anchor)
    return ActiveAWBCBatch(online=online, expert=tuple(expert))


def adaptive_update_steps(*, online_count: int, active_online_count: int) -> int:
    """Bound replay reuse to roughly four uses per admitted online anchor."""

    if online_count <= 0 or active_online_count <= 0:
        raise ValueError("online_count and active_online_count must be positive")
    return min(50, max(1, (4 * int(online_count)) // int(active_online_count)))


def weighted_online_expert_loss(
    expert_per_sample_loss: torch.Tensor,
    online_per_sample_loss: torch.Tensor,
    online_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the round loss without separately averaging expert and online terms.

    Returns the differentiable loss, numerator, and denominator.  The latter two
    can be summed over microbatches before taking the one final quotient.
    """

    expert = torch.as_tensor(expert_per_sample_loss).reshape(-1)
    online = torch.as_tensor(online_per_sample_loss, device=expert.device).reshape(-1)
    weights = torch.as_tensor(online_weights, device=expert.device, dtype=expert.dtype).reshape(-1)
    if online.numel() != weights.numel():
        raise ValueError("online losses and weights must have the same length")
    if expert.numel() == 0 or online.numel() == 0:
        raise ValueError("both expert and online losses are required")
    if torch.any(weights < 0):
        raise ValueError("online weights must be non-negative")
    numerator = expert.sum() + (online * weights).sum()
    denominator = expert.new_tensor(float(expert.numel())) + weights.sum()
    return numerator / denominator.clamp_min(torch.finfo(expert.dtype).eps), numerator, denominator


def aggregate_microbatch_loss(
    numerators: Sequence[torch.Tensor], denominators: Sequence[torch.Tensor]
) -> torch.Tensor:
    """Compute exactly one quotient after accumulating all microbatches."""

    if not numerators or len(numerators) != len(denominators):
        raise ValueError("numerators and denominators must be non-empty and aligned")
    numerator = torch.stack([value.reshape(()) for value in numerators]).sum()
    denominator = torch.stack([value.reshape(()) for value in denominators]).sum()
    return numerator / denominator.clamp_min(torch.finfo(numerator.dtype).eps)
