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

"""RLinf adapter for uq_vla's pi0.5 VFD interface.

The conditioning and velocity-function structure follows
``learnsyslab/uq_vla``'s ``Pi05Adapter``.  RLinf-specific observation
transforms and ``get_velocity`` replace the corresponding LeRobot calls.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor


def _observation_from_dict(payload: dict[str, Any]) -> Any:
    from openpi.models import model as _model

    return _model.Observation.from_dict(payload)


def _expand_batch(tensor: Tensor, num_action_samples: int) -> Tensor:
    batch_size = tensor.shape[0]
    return (
        tensor.unsqueeze(1)
        .expand(batch_size, num_action_samples, *tensor.shape[1:])
        .reshape(batch_size * num_action_samples, *tensor.shape[1:])
    )


class OpenPi05VFDAdapter:
    """Expose an RLinf pi0.5 model through uq_vla's flow adapter contract."""

    def __init__(self, model: Any):
        self.model = model
        self.config = model.config

    @property
    def horizon(self) -> int:
        return self.config.action_horizon

    @property
    def action_dim(self) -> int:
        return self.config.action_dim

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        # uq_vla's Pi05Adapter explicitly performs VFD integration in float32.
        return torch.float32

    @property
    def ode_solver_config(self) -> dict[str, Any]:
        return {
            "solver_method": "euler",
            "step_size": 1.0 / self.config.num_steps,
            "atol": None,
            "rtol": None,
        }

    @property
    def cond_vf_config(self) -> dict[str, Any]:
        return {
            "type": "ot",
            "sigma_min": 0,
            "beta_min": None,
            "beta_max": None,
        }

    @torch.no_grad()
    def prepare_conditioning(
        self,
        env_obs: dict[str, Any],
        num_action_samples: int,
    ) -> dict[str, Any]:
        """Encode an observation once and retain the prefix KV cache."""

        if num_action_samples < 1:
            raise ValueError("num_action_samples must be positive.")

        to_process_obs = self.model.obs_processor(env_obs)
        processed_obs = self.model.input_transform(to_process_obs, transpose=False)
        processed_obs = self.model.precision_processor(processed_obs)
        observation = _observation_from_dict(processed_obs)
        images, img_masks, lang_tokens, lang_masks, state = (
            self.model._preprocess_observation(observation, train=False)  # noqa: SLF001
        )

        device = self.device
        images = [
            _expand_batch(image.to(device), num_action_samples) for image in images
        ]
        img_masks = [
            _expand_batch(mask.to(device), num_action_samples) for mask in img_masks
        ]
        lang_tokens = _expand_batch(lang_tokens.to(device), num_action_samples)
        lang_masks = _expand_batch(lang_masks.to(device), num_action_samples)
        state = _expand_batch(state.to(device), num_action_samples)

        _, prefix_pad_masks, past_key_values = self.model._build_prefix_cache(  # noqa: SLF001
            images,
            img_masks,
            lang_tokens,
            lang_masks,
        )
        return {
            "state": state,
            "prefix_pad_masks": prefix_pad_masks,
            "past_key_values": past_key_values,
        }

    def make_velocity_fn(
        self,
        conditioning: dict[str, Any],
    ) -> Callable[..., Tensor]:
        """Return velocity in the generic noise-at-zero ODE convention."""

        def velocity_fn(t: Tensor, x_t: Tensor) -> Tensor:
            # pi0.5 uses 1=noise and 0=action. uq_vla's ODE integrates 0->1.
            if not isinstance(t, Tensor):
                t = torch.tensor(t, device=x_t.device, dtype=x_t.dtype)
            pi05_t = 1.0 - t
            pi05_velocity, _ = self.model.get_velocity(
                conditioning["state"],
                x_t,
                pi05_t.expand(x_t.shape[0]),
                conditioning["prefix_pad_masks"],
                conditioning["past_key_values"],
            )
            return -pi05_velocity

        return velocity_fn

    @torch.no_grad()
    def sample_prior(
        self,
        num_samples: int,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample the shared Gaussian prior used by both ensemble members."""

        return torch.randn(
            (num_samples, self.horizon, self.action_dim),
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )
