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

import pytest
import torch

from rlinf.algorithms.vfd import (
    VfdOneway,
    euler_integrate,
    make_sampling_time_grid,
    process_velocity_eval_times,
    select_ode_states,
)


def test_process_velocity_eval_times_matches_uq_vla_rules():
    assert process_velocity_eval_times([0.25, 0.5]) == [0.0, 0.25, 0.5]
    assert process_velocity_eval_times([0.0, 0.5]) == [0.0, 0.5]

    with pytest.raises(ValueError, match="strictly increasing"):
        process_velocity_eval_times([0.0, 0.5, 0.5])
    with pytest.raises(ValueError, match="0.0 <= t < 1.0"):
        process_velocity_eval_times([0.0, 1.0])


def test_sampling_grid_contains_regular_and_requested_times():
    grid = make_sampling_time_grid(
        step_size=0.25,
        extra_times=[0.1, 0.9, 0.25],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        grid,
        torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]),
    )


def test_select_ode_states_requires_exact_grid_membership():
    grid = torch.tensor([0.0, 0.25, 0.5, 1.0])
    states = torch.arange(4, dtype=torch.float32).reshape(4, 1, 1, 1)
    selected, times = select_ode_states(grid, states, torch.tensor([0.25, 0.5]))
    torch.testing.assert_close(times, torch.tensor([0.25, 0.5]))
    torch.testing.assert_close(selected[:, 0, 0, 0], torch.tensor([1.0, 2.0]))

    with pytest.raises(ValueError, match="not found"):
        select_ode_states(grid, states, torch.tensor([0.4]))


def test_euler_integrate_matches_constant_velocity_solution():
    initial = torch.zeros(2, 1, 1)
    grid = torch.tensor([0.0, 0.25, 0.5, 1.0])

    def velocity_fn(*, x_t, t):
        del t
        return torch.full_like(x_t, 2.0)

    states, velocities = euler_integrate(initial, velocity_fn, grid)
    torch.testing.assert_close(
        states[:, 0, 0, 0],
        torch.tensor([0.0, 0.5, 1.0, 2.0]),
    )
    torch.testing.assert_close(velocities, torch.full_like(velocities, 2.0))


def test_vfd_oneway_is_zero_for_identical_velocity_fields():
    grid = torch.tensor([0.0, 0.5, 1.0])
    states = torch.zeros(3, 4, 2, 3)

    def velocity_fn(*, x_t, t):
        del t
        return x_t + 1.0

    metric = VfdOneway([0.0, 0.5], grid)
    scores = metric(states, velocity_fn, states, velocity_fn)
    torch.testing.assert_close(scores, torch.zeros(4))


def test_vfd_oneway_matches_uq_vla_weighted_l2_formula():
    grid = torch.tensor([0.0, 0.5, 1.0])
    states = torch.zeros(3, 2, 1, 1)

    def reference_velocity(*, x_t, t):
        del t
        return torch.zeros_like(x_t)

    def comparison_velocity(*, x_t, t):
        del t
        return torch.full_like(x_t, 2.0)

    # At t=0 the OT scale is zero. At t=0.5, kappa=1, ||0-2||^2=4,
    # and dt=0.5, so each sample has VFD 2.
    metric = VfdOneway([0.0, 0.5], grid)
    scores = metric(
        states,
        reference_velocity,
        states,
        comparison_velocity,
    )
    torch.testing.assert_close(scores, torch.tensor([2.0, 2.0]))


def test_vfd_sample_reduction_preserves_observation_batch():
    batch_size = 2
    num_action_samples = 3
    grid = torch.tensor([0.0, 0.5, 1.0])
    states = torch.zeros(3, batch_size * num_action_samples, 1, 1)

    def reference_velocity(*, x_t, t):
        del t
        return torch.zeros_like(x_t)

    def comparison_velocity(*, x_t, t):
        del t
        values = torch.arange(x_t.shape[0], dtype=x_t.dtype).reshape(-1, 1, 1)
        return values.expand_as(x_t)

    per_sample = VfdOneway([0.0, 0.5], grid)(
        states,
        reference_velocity,
        states,
        comparison_velocity,
    )
    scores = per_sample.reshape(batch_size, num_action_samples).mean(dim=1)
    expected = torch.tensor([(0**2 + 1**2 + 2**2) / 6, (3**2 + 4**2 + 5**2) / 6])
    torch.testing.assert_close(scores, expected)
