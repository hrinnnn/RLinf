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

"""Velocity-field disagreement utilities for flow-matching policies.

This module adapts the Euler solver, time-grid helpers, and ``VfdOneway``
metric from ``learnsyslab/uq_vla``.  The source project is Apache-2.0 licensed.
Only imports, type annotations, and validation were adjusted for RLinf.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
from torch import Tensor


def process_velocity_eval_times(velocity_eval_times: Sequence[float]) -> list[float]:
    """Validate VFD evaluation times and ensure that the path starts at zero."""

    times = [float(time) for time in velocity_eval_times]
    for time in times:
        if not 0.0 <= time < 1.0:
            raise ValueError(
                "VFD velocity_eval_times entries must satisfy 0.0 <= t < 1.0; "
                f"got {times}."
            )
    if not times or times[0] != 0.0:
        times = [0.0, *times]
    if any(next_time <= time for time, next_time in zip(times, times[1:])):
        raise ValueError(
            f"VFD velocity_eval_times must be strictly increasing; got {times}."
        )
    return times


def make_sampling_time_grid(
    step_size: float,
    device: torch.device,
    dtype: torch.dtype,
    extra_times: Sequence[float] | Tensor | None = None,
) -> Tensor:
    """Build the uq_vla fixed-step grid plus requested VFD evaluation times."""

    if not 0.0 < step_size <= 1.0:
        raise ValueError("step_size must be > 0 and <= 1.")

    num_full_steps = math.floor(1.0 / step_size)
    time_grid = torch.linspace(
        0.0,
        num_full_steps * step_size,
        steps=num_full_steps + 1,
        device=device,
        dtype=dtype,
    )
    if time_grid[-1] < 1.0:
        time_grid = torch.cat(
            [time_grid, torch.tensor([1.0], device=device, dtype=dtype)]
        )
    if extra_times is not None:
        extra_tensor = torch.as_tensor(extra_times, device=device, dtype=dtype)
        time_grid = torch.cat([time_grid, extra_tensor.clamp(0.0, 1.0)])

    time_grid, _ = torch.sort(time_grid)
    keep = torch.ones_like(time_grid, dtype=torch.bool)
    keep[1:] = torch.diff(time_grid) > 1e-4
    time_grid = time_grid[keep]
    if time_grid[0].item() != 0.0 or time_grid[-1].item() != 1.0:
        raise RuntimeError("Sampling time grid must start at 0.0 and end at 1.0.")
    return time_grid


def select_ode_states(
    time_grid: Tensor,
    ode_states: Tensor,
    requested_times: Tensor,
) -> tuple[Tensor, Tensor]:
    """Select states whose timestamps match the requested VFD evaluation times."""

    if time_grid.size(0) != ode_states.size(0):
        raise ValueError(
            "time_grid and ode_states must have the same length; "
            f"got {time_grid.size(0)} and {ode_states.size(0)}."
        )

    matched_indices: list[int] = []
    for requested_time in requested_times:
        time_mask = torch.isclose(time_grid, requested_time, atol=1e-5, rtol=0)
        match_count = int(time_mask.sum().item())
        if match_count == 0:
            raise ValueError(
                f"Requested time {requested_time.item()} not found in time_grid."
            )
        if match_count > 1:
            raise ValueError(
                f"Requested time {requested_time.item()} matched {match_count} "
                "entries in time_grid; expected exactly one."
            )
        matched_indices.append(time_mask.nonzero(as_tuple=True)[0].item())

    return ode_states[matched_indices], time_grid[matched_indices]


def euler_integrate(
    x_0: Tensor,
    velocity_fn: Callable[..., Tensor],
    time_grid: Tensor,
) -> tuple[Tensor, Tensor]:
    """Run the explicit Euler integration used by uq_vla's pi0.5 adapter."""

    states: list[Tensor] = []
    velocities: list[Tensor] = []
    x_t = x_0
    for time, next_time in zip(time_grid[:-1], time_grid[1:]):
        states.append(x_t)
        velocity = velocity_fn(x_t=x_t, t=time)
        velocities.append(velocity)
        x_t = x_t + (next_time - time) * velocity

    states.append(x_t)
    velocities.append(velocity_fn(x_t=x_t, t=time_grid[-1]))
    return torch.stack(states, dim=0), torch.stack(velocities, dim=0)


class VfdOneway:
    """uq_vla's one-way velocity-field disagreement metric for OT flows."""

    name = "vfd_oneway"

    def __init__(
        self,
        velocity_eval_times: Sequence[float],
        sampling_time_grid: Tensor,
    ):
        self.velocity_eval_times = process_velocity_eval_times(velocity_eval_times)
        self.sampling_time_grid = sampling_time_grid

    def __call__(
        self,
        ref_ode_states: Tensor,
        ref_velocity_fn: Callable[..., Tensor],
        cmp_ode_states: Tensor,
        cmp_velocity_fn: Callable[..., Tensor],
    ) -> Tensor:
        """Integrate weighted squared velocity disagreement over flow time."""

        device = ref_ode_states.device
        dtype = ref_ode_states.dtype
        requested_times = torch.tensor(
            self.velocity_eval_times,
            device=device,
            dtype=dtype,
        )
        selected_ref_states, selected_ref_times = select_ode_states(
            self.sampling_time_grid,
            ref_ode_states,
            requested_times,
        )
        selected_cmp_states, selected_cmp_times = select_ode_states(
            self.sampling_time_grid,
            cmp_ode_states,
            requested_times,
        )
        if not torch.equal(selected_ref_times, selected_cmp_times):
            raise ValueError(
                "Reference and comparison VFD evaluation times do not match."
            )

        batch_size = ref_ode_states.shape[1]
        score = torch.zeros(batch_size, device=device, dtype=dtype)
        for index, (time, ref_state, cmp_state) in enumerate(
            zip(
                selected_ref_times,
                selected_ref_states,
                selected_cmp_states,
                strict=False,
            )
        ):
            if index < len(selected_ref_times) - 1:
                delta_time = selected_ref_times[index + 1] - time
            else:
                delta_time = 1.0 - time

            ref_velocity = ref_velocity_fn(x_t=ref_state, t=time)
            cmp_velocity = cmp_velocity_fn(x_t=cmp_state, t=time)
            velocity_difference = torch.norm(
                ref_velocity - cmp_velocity,
                dim=(1, 2),
            ).square()
            score += time / (1.0 - time) * velocity_difference * delta_time
        return score
