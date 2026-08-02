# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AWBCWeightResult:
    weights: torch.Tensor
    gains: torch.Tensor
    mean_episode_length: torch.Tensor
    gain_mean: torch.Tensor
    gain_std: torch.Tensor
    valid_count: int
    used_fallback: bool

    @property
    def effective_sample_size(self) -> torch.Tensor:
        denominator = self.weights.square().sum()
        if denominator <= 0:
            return self.weights.new_zeros(())
        return self.weights.sum().square() / denominator


def _as_1d_tensor(value, *, name: str, device=None, dtype=None) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return tensor


def _all_gather_variable_1d(tensor: torch.Tensor) -> torch.Tensor:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return tensor

    world_size = torch.distributed.get_world_size()
    local_size = torch.tensor([tensor.numel()], device=tensor.device, dtype=torch.long)
    sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    torch.distributed.all_gather(sizes, local_size)
    max_size = max(int(size.item()) for size in sizes)

    padded = tensor.new_zeros(max_size)
    padded[: tensor.numel()] = tensor
    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, padded)
    return torch.cat(
        [value[: int(size.item())] for value, size in zip(gathered, sizes)], dim=0
    )


def compute_arm_awbc_weights(
    delta_phi,
    episode_lengths,
    *,
    valid=None,
    confidence=None,
    sigma_multiplier: float = 2.0,
    epsilon: float = 1e-6,
    negative_delta_policy: str = "continuous",
    confidence_power: float = 0.0,
    weight_floor: float = 0.0,
    gain_clip: tuple[float, float] | None = None,
    distributed: bool = False,
    global_delta_phi=None,
    global_episode_lengths=None,
    global_valid=None,
) -> AWBCWeightResult:
    """Compute ARM-style advantage weights for a local training batch.

    ``global_*`` inputs represent an already gathered batch. They are useful for
    callers that own distributed communication and for deterministic tests. When
    omitted with ``distributed=True``, this function gathers statistics from all
    initialized torch.distributed ranks.
    """

    delta = _as_1d_tensor(delta_phi, name="delta_phi", dtype=torch.float32)
    lengths = _as_1d_tensor(
        episode_lengths,
        name="episode_lengths",
        device=delta.device,
        dtype=torch.float32,
    )
    if delta.numel() != lengths.numel():
        raise ValueError("delta_phi and episode_lengths must have the same length")
    if torch.any(lengths <= 0):
        raise ValueError("episode_lengths must be positive")

    if valid is None:
        valid_tensor = torch.ones_like(delta, dtype=torch.bool)
    else:
        valid_tensor = _as_1d_tensor(valid, name="valid", device=delta.device).bool()
    if valid_tensor.numel() != delta.numel():
        raise ValueError("valid and delta_phi must have the same length")

    if confidence is None:
        confidence_tensor = torch.ones_like(delta)
    else:
        confidence_tensor = _as_1d_tensor(
            confidence, name="confidence", device=delta.device, dtype=torch.float32
        )
    if confidence_tensor.numel() != delta.numel():
        raise ValueError("confidence and delta_phi must have the same length")
    if negative_delta_policy not in {"continuous", "zero"}:
        raise ValueError(
            "negative_delta_policy must be either 'continuous' or 'zero'"
        )
    if sigma_multiplier <= 0:
        raise ValueError("sigma_multiplier must be positive")
    if not 0.0 <= weight_floor <= 1.0:
        raise ValueError("weight_floor must be in [0, 1]")
    if confidence_power < 0:
        raise ValueError("confidence_power must be non-negative")
    if gain_clip is not None and gain_clip[0] >= gain_clip[1]:
        raise ValueError("gain_clip must have increasing bounds")

    if global_delta_phi is not None:
        if global_episode_lengths is None:
            raise ValueError("global_episode_lengths is required with global_delta_phi")
        stats_delta = _as_1d_tensor(
            global_delta_phi, name="global_delta_phi", device=delta.device, dtype=torch.float32
        )
        stats_lengths = _as_1d_tensor(
            global_episode_lengths,
            name="global_episode_lengths",
            device=delta.device,
            dtype=torch.float32,
        )
        stats_valid = (
            torch.ones_like(stats_delta, dtype=torch.bool)
            if global_valid is None
            else _as_1d_tensor(
                global_valid, name="global_valid", device=delta.device
            ).bool()
        )
    elif distributed:
        stats_delta = _all_gather_variable_1d(delta.detach())
        stats_lengths = _all_gather_variable_1d(lengths.detach())
        stats_valid = _all_gather_variable_1d(valid_tensor.to(torch.uint8)).bool()
    else:
        stats_delta = delta
        stats_lengths = lengths
        stats_valid = valid_tensor

    if not (
        stats_delta.numel() == stats_lengths.numel() == stats_valid.numel()
    ):
        raise ValueError("global AWBC statistic inputs must have the same length")
    if torch.any(stats_lengths <= 0):
        raise ValueError("global_episode_lengths must be positive")

    valid_count = int(stats_valid.sum().item())
    zero = delta.new_zeros(())
    if valid_count == 0:
        return AWBCWeightResult(
            weights=torch.zeros_like(delta),
            gains=torch.zeros_like(delta),
            mean_episode_length=zero,
            gain_mean=zero,
            gain_std=zero,
            valid_count=0,
            used_fallback=True,
        )

    mean_episode_length = stats_lengths[stats_valid].mean()
    gains = delta * lengths / mean_episode_length
    stats_gains = stats_delta * stats_lengths / mean_episode_length
    if gain_clip is not None:
        gains = gains.clamp(*gain_clip)
        stats_gains = stats_gains.clamp(*gain_clip)

    valid_stats_gains = stats_gains[stats_valid]
    gain_mean = valid_stats_gains.mean()
    gain_std = valid_stats_gains.std(correction=0)
    used_fallback = valid_count < 2 or bool(gain_std <= epsilon)
    if used_fallback:
        weights = valid_tensor.to(dtype=delta.dtype)
    else:
        lower = gain_mean - sigma_multiplier * gain_std
        upper = gain_mean + sigma_multiplier * gain_std
        weights = ((gains - lower) / (upper - lower + epsilon)).clamp(0.0, 1.0)
        weights = torch.where(valid_tensor, weights, torch.zeros_like(weights))

    if confidence_power > 0:
        confidence_scale = confidence_tensor.clamp(0.0, 1.0).pow(confidence_power)
        weights = weights * confidence_scale
    if weight_floor > 0:
        weights = torch.where(
            valid_tensor,
            weight_floor + (1.0 - weight_floor) * weights,
            torch.zeros_like(weights),
        )
    if negative_delta_policy == "zero":
        weights = torch.where(delta < 0, torch.zeros_like(weights), weights)
    weights = torch.where(valid_tensor, weights, torch.zeros_like(weights))

    return AWBCWeightResult(
        weights=weights,
        gains=gains,
        mean_episode_length=mean_episode_length,
        gain_mean=gain_mean,
        gain_std=gain_std,
        valid_count=valid_count,
        used_fallback=used_fallback,
    )


def compute_flux_awbc_weights(
    delta_phi,
    episode_lengths,
    *,
    valid=None,
    progress_threshold: float = 0.01,
    sigma_multiplier: float = 2.0,
    epsilon: float = 1e-6,
    distributed: bool = False,
    global_delta_phi=None,
    global_episode_lengths=None,
    global_valid=None,
) -> AWBCWeightResult:
    """Compute the thresholded AWBC rule used by the FluxVLA-style smoke.

    Negative progress is rejected, progress above ``progress_threshold`` gets
    full base weight, and the narrow near-zero band receives a continuous
    mean/std weight. Episode-length scaling is applied before normalizing the
    positive local batch to mean one.
    """

    if progress_threshold < 0:
        raise ValueError("progress_threshold must be non-negative")
    if sigma_multiplier <= 0:
        raise ValueError("sigma_multiplier must be positive")

    delta = _as_1d_tensor(delta_phi, name="delta_phi", dtype=torch.float32)
    lengths = _as_1d_tensor(
        episode_lengths,
        name="episode_lengths",
        device=delta.device,
        dtype=torch.float32,
    )
    if delta.numel() != lengths.numel():
        raise ValueError("delta_phi and episode_lengths must have the same length")
    if torch.any(lengths <= 0):
        raise ValueError("episode_lengths must be positive")
    valid_tensor = (
        torch.ones_like(delta, dtype=torch.bool)
        if valid is None
        else _as_1d_tensor(valid, name="valid", device=delta.device).bool()
    )
    if valid_tensor.numel() != delta.numel():
        raise ValueError("valid and delta_phi must have the same length")

    if global_delta_phi is not None:
        if global_episode_lengths is None:
            raise ValueError("global_episode_lengths is required with global_delta_phi")
        stats_delta = _as_1d_tensor(
            global_delta_phi,
            name="global_delta_phi",
            device=delta.device,
            dtype=torch.float32,
        )
        stats_lengths = _as_1d_tensor(
            global_episode_lengths,
            name="global_episode_lengths",
            device=delta.device,
            dtype=torch.float32,
        )
        stats_valid = (
            torch.ones_like(stats_delta, dtype=torch.bool)
            if global_valid is None
            else _as_1d_tensor(
                global_valid, name="global_valid", device=delta.device
            ).bool()
        )
    elif distributed:
        stats_delta = _all_gather_variable_1d(delta.detach())
        stats_lengths = _all_gather_variable_1d(lengths.detach())
        stats_valid = _all_gather_variable_1d(valid_tensor.to(torch.uint8)).bool()
    else:
        stats_delta, stats_lengths, stats_valid = delta, lengths, valid_tensor
    if not (
        stats_delta.numel() == stats_lengths.numel() == stats_valid.numel()
    ):
        raise ValueError("global AWBC statistic inputs must have the same length")

    valid_count = int(stats_valid.sum().item())
    zero = delta.new_zeros(())
    if valid_count == 0:
        return AWBCWeightResult(
            weights=torch.zeros_like(delta),
            gains=torch.zeros_like(delta),
            mean_episode_length=zero,
            gain_mean=zero,
            gain_std=zero,
            valid_count=0,
            used_fallback=False,
        )

    valid_stats = stats_delta[stats_valid]
    delta_mean = torch.clamp(valid_stats.mean(), min=0.0)
    delta_std = valid_stats.std(correction=0)
    lower = delta_mean - sigma_multiplier * delta_std
    upper = delta_mean + sigma_multiplier * delta_std
    soft = ((delta - lower) / (upper - lower + epsilon)).clamp(0.0, 1.0)
    base_weights = torch.where(
        delta > progress_threshold,
        torch.ones_like(delta),
        torch.where(delta >= 0, soft, torch.zeros_like(delta)),
    )
    base_weights = torch.where(
        valid_tensor, base_weights, torch.zeros_like(base_weights)
    )

    mean_episode_length = stats_lengths[stats_valid].mean()
    length_factor = lengths / mean_episode_length
    weights = base_weights * length_factor
    positive = weights > 0
    if positive.any():
        weights = weights / weights[positive].mean().clamp_min(epsilon)

    gains = delta * length_factor
    return AWBCWeightResult(
        weights=weights,
        gains=gains,
        mean_episode_length=mean_episode_length,
        gain_mean=delta_mean,
        gain_std=delta_std,
        valid_count=valid_count,
        used_fallback=False,
    )


def weighted_flow_matching_loss(
    element_loss: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    *,
    element_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce PI0 flow loss with optional sample and temporal weights."""

    if element_loss.ndim == 0:
        raise ValueError("element_loss must include a batch dimension")
    if element_mask is None:
        per_sample = (
            element_loss
            if element_loss.ndim == 1
            else element_loss.reshape(element_loss.shape[0], -1).mean(dim=1)
        )
    else:
        mask = torch.as_tensor(
            element_mask, device=element_loss.device, dtype=element_loss.dtype
        )
        if mask.ndim == element_loss.ndim - 1:
            mask = mask.unsqueeze(-1)
        try:
            mask = torch.broadcast_to(mask, element_loss.shape)
        except RuntimeError as error:
            raise ValueError(
                "element_mask must broadcast to the per-element flow loss shape"
            ) from error
        flat_mask = mask.reshape(mask.shape[0], -1)
        valid_count = flat_mask.sum(dim=1)
        if torch.any(valid_count <= 0):
            raise ValueError(
                "every flow-matching sample must contain a valid action target"
            )
        per_sample = (
            (element_loss * mask).reshape(mask.shape[0], -1).sum(dim=1)
            / valid_count
        )
    if sample_weights is None:
        return per_sample.mean(), per_sample

    weights = torch.as_tensor(
        sample_weights, device=per_sample.device, dtype=per_sample.dtype
    ).reshape(-1)
    if weights.numel() != per_sample.numel():
        raise ValueError("sample_weights must match the loss batch dimension")
    if torch.any(weights < 0):
        raise ValueError("sample_weights must be non-negative")
    denominator = weights.sum()
    if denominator <= 0:
        return per_sample.sum() * 0.0, per_sample
    return (per_sample * weights).sum() / denominator, per_sample
