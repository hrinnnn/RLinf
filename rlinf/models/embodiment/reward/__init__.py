# Copyright 2025 The RLinf Authors.
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

"""Reward models for embodied RL."""

from rlinf.models.embodiment.reward.base_reward_model import BaseRewardModel

__all__ = [
    "BaseRewardModel",
    "get_reward_model_class",
]

reward_model_registry = {
    "resnet": "rlinf.models.embodiment.reward.resnet_reward_model:ResNetRewardModel",
    "dopamine_grm": (
        "rlinf.models.embodiment.reward.dopamine_grm_reward_model:"
        "DopamineGRMRewardModel"
    ),
    "vlm": "rlinf.models.embodiment.reward.vlm_reward_model:VLMRewardModel",
    "history_vlm": (
        "rlinf.models.embodiment.reward.vlm_reward_model:HistoryVLMRewardModel"
    ),
}


def get_reward_model_class(reward_model_type: str):
    if reward_model_type not in reward_model_registry:
        raise ValueError(f"Unsupported reward model type: {reward_model_type}")

    import importlib

    module_name, class_name = reward_model_registry[reward_model_type].split(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
