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

import json

import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from rlinf.models.embodiment.reward.dopamine_grm_reward_model import (
    DopamineGRMRewardModel,
    consistency_aware_phi,
    dopamine_grm_score_to_phi,
    parse_dopamine_grm_score,
)


class FakeDopamineGRMRewardModel(DopamineGRMRewardModel):
    def __init__(self, cfg, responses):
        self.responses = list(responses)
        super().__init__(cfg)

    def _request_grm(self, payloads):
        del payloads
        if not self.responses:
            return [""] * len(self.modes), [0.0] * len(self.modes)
        outputs = self.responses.pop(0)
        return outputs, [0.0] * len(outputs)


def _write_goal_bank(tmp_path):
    task_dir = tmp_path / "goal_bank" / "task_000"
    task_dir.mkdir(parents=True)
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(task_dir / "goal_main.png")
    Image.new("RGB", (4, 4), color=(0, 255, 0)).save(task_dir / "goal_wrist.png")
    with open(task_dir / "meta.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "task_id": 0,
                "task_description": "put the block in the bowl",
                "source_demo": "demo.hdf5",
                "views": {"main": "goal_main.png", "wrist": "goal_wrist.png"},
            },
            file,
        )
    return tmp_path / "goal_bank"


def _cfg(goal_bank, **overrides):
    data = {
        "grm_endpoint": "http://fake/v1/chat/completions",
        "model_name": "fake-grm",
        "goal_bank_dir": str(goal_bank),
        "modes": ["incremental", "forward", "backward"],
        "gamma": 0.99,
        "grm_interval_chunks": 1,
        "num_envs": 1,
    }
    data.update(overrides)
    return OmegaConf.create(data)


def _obs(success=False, done=False):
    image = torch.zeros((1, 4, 4, 3), dtype=torch.uint8)
    wrist = torch.ones((1, 4, 4, 3), dtype=torch.uint8)
    return {
        "main_images": image,
        "wrist_images": wrist,
        "reference_start_main_images": image.clone(),
        "reference_start_wrist_images": wrist.clone(),
        "task_descriptions": ["put the block in the bowl"],
        "task_ids": torch.tensor([0]),
        "dones": torch.tensor([done]),
        "env_infos": {"episode": {"success_once": torch.tensor([success])}},
    }


def test_parse_dopamine_grm_score_clips_and_marks_invalid():
    assert parse_dopamine_grm_score("<score>+35%</score>").raw_score == 0.35
    assert parse_dopamine_grm_score("<score>-4%</score>").raw_score == -0.04
    assert parse_dopamine_grm_score("<score>+999%</score>").raw_score == 1.0
    assert parse_dopamine_grm_score("not a score").valid is False


@pytest.mark.parametrize(
    ("mode", "raw", "prev", "expected"),
    [
        ("incremental", 0.5, 0.4, 0.7),
        ("incremental", -0.5, 0.4, 0.2),
        ("forward", 0.6, 0.4, 0.6),
        ("backward", -0.2, 0.4, 0.8),
    ],
)
def test_score_to_phi(mode, raw, prev, expected):
    assert dopamine_grm_score_to_phi(mode, raw, prev) == pytest.approx(expected)


def test_three_mode_fusion_updates_prev_phi(tmp_path):
    model = FakeDopamineGRMRewardModel(
        _cfg(_write_goal_bank(tmp_path)),
        responses=[
            ["<score>+50%</score>", "<score>+60%</score>", "<score>-20%</score>"]
        ],
    )

    reward = model.compute_reward(_obs())

    # Eq. 9--11: mean global phi=.7, delta incremental=.5, w=exp(-1*(.2/.7)^2).
    phi_next, _, _, _ = consistency_aware_phi(0.0, 0.5, 0.6, 0.8, 1.0, 1e-6)
    assert reward.item() == pytest.approx(0.99 * phi_next)
    assert model.prev_phi[0].item() == pytest.approx(phi_next)


def test_consistency_aware_phi_matches_paper_equations():
    phi, global_mean, discrepancy, confidence = consistency_aware_phi(
        prev_phi=0.2,
        incremental_phi=0.5,
        forward_phi=0.4,
        backward_phi=0.8,
        alpha=2.0,
        epsilon=1e-6,
    )
    assert global_mean == pytest.approx(0.6)
    expected_discrepancy = 0.4 / (0.6 + 1e-6)
    assert discrepancy == pytest.approx(expected_discrepancy)
    assert confidence == pytest.approx(
        torch.exp(torch.tensor(-2.0 * expected_discrepancy**2)).item()
    )
    assert phi == pytest.approx(0.2 + confidence / 2 * (0.6 - 0.2 + 0.3))


def test_all_invalid_does_not_update_prev_phi(tmp_path):
    model = FakeDopamineGRMRewardModel(
        _cfg(_write_goal_bank(tmp_path)),
        responses=[
            ["<score>+50%</score>", "<score>+50%</score>", "<score>+50%</score>"],
            ["bad", "bad", "bad"],
        ],
    )
    model.compute_reward(_obs())
    prev_phi = model.prev_phi[0].item()

    reward = model.compute_reward(_obs())

    assert reward.item() == 0.0
    assert model.prev_phi[0].item() == prev_phi


def test_terminal_transition_uses_zero_potential_and_resets_on_done(tmp_path):
    model = FakeDopamineGRMRewardModel(
        _cfg(_write_goal_bank(tmp_path)),
        responses=[
            ["<score>+50%</score>", "<score>+50%</score>", "<score>+50%</score>"],
            ["bad", "bad", "bad"],
        ],
    )
    model.compute_reward(_obs())
    prev_phi = model.prev_phi[0].item()

    reward = model.compute_reward(_obs(success=True, done=True))

    assert reward.item() == pytest.approx(-prev_phi)
    assert model.prev_phi[0].item() == 0.0
    assert not model.has_prev_phi[0].item()


def test_low_frequency_skips_without_request(tmp_path):
    model = FakeDopamineGRMRewardModel(
        _cfg(_write_goal_bank(tmp_path), grm_interval_chunks=2),
        responses=[
            ["<score>+50%</score>", "<score>+50%</score>", "<score>+50%</score>"]
        ],
    )

    skipped = model.compute_reward(_obs())
    reward = model.compute_reward(_obs())

    phi_next, _, _, _ = consistency_aware_phi(0.0, 0.5, 0.5, 1.0, 1.0, 1e-6)
    assert skipped.item() == 0.0
    assert reward.item() == pytest.approx((0.99**2) * phi_next)
