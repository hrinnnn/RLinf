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

from types import SimpleNamespace

import torch

from rlinf.models.embodiment.openpi import openpi_vfd_adapter as adapter_module
from rlinf.models.embodiment.openpi.openpi_vfd_adapter import OpenPi05VFDAdapter


class _FakeOpenPiModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(action_horizon=2, action_dim=3, num_steps=4)
        self.prefix_cache_calls = 0
        self.last_timesteps = None

    def obs_processor(self, env_obs):
        return env_obs

    def input_transform(self, env_obs, transpose):
        assert transpose is False
        return env_obs

    def precision_processor(self, env_obs):
        return env_obs

    def _preprocess_observation(self, observation, train):
        assert train is False
        batch_size = observation["states"].shape[0]
        images = [torch.arange(batch_size, dtype=torch.float32).reshape(-1, 1, 1, 1)]
        image_masks = [torch.ones(batch_size, dtype=torch.bool)]
        language_tokens = torch.arange(batch_size).reshape(-1, 1)
        language_masks = torch.ones(batch_size, 1, dtype=torch.bool)
        return (
            images,
            image_masks,
            language_tokens,
            language_masks,
            observation["states"],
        )

    def _build_prefix_cache(self, images, image_masks, tokens, masks):
        self.prefix_cache_calls += 1
        assert images[0].shape[0] == 6
        assert image_masks[0].shape[0] == 6
        assert tokens.shape[0] == 6
        assert masks.shape[0] == 6
        return None, torch.ones(6, 2, dtype=torch.bool), "fake-cache"

    def get_velocity(
        self,
        state,
        x_t,
        timestep,
        prefix_pad_masks,
        past_key_values,
    ):
        assert state.shape[0] == x_t.shape[0]
        assert prefix_pad_masks.shape[0] == x_t.shape[0]
        assert past_key_values == "fake-cache"
        self.last_timesteps = timestep
        velocity = timestep[:, None, None].expand_as(x_t)
        return velocity, None


def test_prepare_conditioning_expands_samples_and_builds_prefix_once(monkeypatch):
    monkeypatch.setattr(
        adapter_module,
        "_observation_from_dict",
        lambda payload: payload,
    )
    model = _FakeOpenPiModel()
    adapter = OpenPi05VFDAdapter(model)
    conditioning = adapter.prepare_conditioning(
        {"states": torch.tensor([[1.0], [2.0]])},
        num_action_samples=3,
    )

    assert model.prefix_cache_calls == 1
    assert conditioning["state"].shape == (6, 1)
    torch.testing.assert_close(
        conditioning["state"].squeeze(-1),
        torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0]),
    )


def test_velocity_fn_inverts_pi05_time_and_velocity_sign():
    model = _FakeOpenPiModel()
    adapter = OpenPi05VFDAdapter(model)
    conditioning = {
        "state": torch.zeros(2, 1),
        "prefix_pad_masks": torch.ones(2, 2, dtype=torch.bool),
        "past_key_values": "fake-cache",
    }
    velocity_fn = adapter.make_velocity_fn(conditioning)
    velocity = velocity_fn(t=torch.tensor(0.25), x_t=torch.zeros(2, 2, 3))

    torch.testing.assert_close(model.last_timesteps, torch.tensor([0.75, 0.75]))
    torch.testing.assert_close(velocity, torch.full((2, 2, 3), -0.75))


def test_sample_prior_shape_dtype_and_generator_reproducibility():
    adapter = OpenPi05VFDAdapter(_FakeOpenPiModel())
    first = adapter.sample_prior(5, generator=torch.Generator().manual_seed(7))
    second = adapter.sample_prior(5, generator=torch.Generator().manual_seed(7))

    assert first.shape == (5, 2, 3)
    assert first.dtype == torch.float32
    torch.testing.assert_close(first, second)
