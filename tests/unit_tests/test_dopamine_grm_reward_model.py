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
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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


def _chunk_obs(*, success=False, done_step=None, chunk_size=10):
    obs = _obs(success=success, done=done_step is not None)
    chunk_dones = torch.zeros((1, chunk_size), dtype=torch.bool)
    if done_step is not None:
        chunk_dones[0, done_step] = True
    obs["chunk_dones"] = chunk_dones
    return obs


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


def test_missing_goal_bank_fails_fast(tmp_path):
    with pytest.raises(FileNotFoundError, match="goal bank"):
        FakeDopamineGRMRewardModel(
            _cfg(tmp_path / "missing"),
            responses=[],
        )


def test_goal_bank_with_missing_image_fails_fast(tmp_path):
    task_dir = tmp_path / "goal_bank" / "task_000"
    task_dir.mkdir(parents=True)
    (task_dir / "meta.json").write_text(
        json.dumps(
            {
                "task_id": 0,
                "task_description": "insert the peg in the hole",
                "views": {"main": "missing.png"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="goal main image"):
        FakeDopamineGRMRewardModel(
            _cfg(tmp_path / "goal_bank"),
            responses=[],
        )


def test_chunk_reward_is_emitted_once_with_actual_horizon(tmp_path):
    model = FakeDopamineGRMRewardModel(
        _cfg(
            _write_goal_bank(tmp_path),
            reward_output_mode="chunk_first",
            num_action_chunks=10,
        ),
        responses=[
            ["<score>+50%</score>", "<score>+60%</score>", "<score>-20%</score>"]
        ],
    )

    reward = model.compute_reward(_chunk_obs())

    phi_next, _, _, _ = consistency_aware_phi(0.0, 0.5, 0.6, 0.8, 1.0, 1e-6)
    assert reward.shape == (1, 10)
    assert reward[0, 0].item() == pytest.approx((0.99**10) * phi_next)
    assert torch.count_nonzero(reward[0, 1:]).item() == 0


def test_chunk_interval_greater_than_one_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="grm_interval_chunks=1"):
        FakeDopamineGRMRewardModel(
            _cfg(
                _write_goal_bank(tmp_path),
                reward_output_mode="chunk_first",
                num_action_chunks=10,
                grm_interval_chunks=2,
            ),
            responses=[],
        )


def test_chunk_terminal_reward_resets_and_is_not_broadcast(tmp_path):
    model = FakeDopamineGRMRewardModel(
        _cfg(
            _write_goal_bank(tmp_path),
            reward_output_mode="chunk_first",
            num_action_chunks=10,
        ),
        responses=[
            ["<score>+50%</score>", "<score>+50%</score>", "<score>-50%</score>"],
            ["bad", "bad", "bad"],
        ],
    )
    model.compute_reward(_chunk_obs())
    prev_phi = model.prev_phi[0].item()

    reward = model.compute_reward(_chunk_obs(success=True, done_step=3))

    assert reward.shape == (1, 10)
    assert reward[0, 0].item() == pytest.approx(-prev_phi)
    assert torch.count_nonzero(reward[0, 1:]).item() == 0
    assert model.prev_phi[0].item() == 0.0
    assert not model.has_prev_phi[0].item()


@pytest.mark.parametrize("terminal_kind", ["failure", "truncation"])
def test_chunk_failure_and_truncation_use_zero_terminal_potential(
    tmp_path, terminal_kind
):
    model = FakeDopamineGRMRewardModel(
        _cfg(
            _write_goal_bank(tmp_path),
            reward_output_mode="chunk_first",
            num_action_chunks=10,
        ),
        responses=[
            ["<score>+40%</score>", "<score>+40%</score>", "<score>-60%</score>"],
            ["<score>-20%</score>", "<score>+20%</score>", "<score>-80%</score>"],
        ],
    )
    model.compute_reward(_chunk_obs())
    prev_phi = model.prev_phi[0].item()

    terminal_obs = _chunk_obs(done_step=4)
    terminal_obs["terminal_kind"] = terminal_kind
    reward = model.compute_reward(terminal_obs)

    assert reward[0, 0].item() == pytest.approx(-prev_phi)
    assert torch.count_nonzero(reward[0, 1:]).item() == 0
    assert model.prev_phi[0].item() == 0.0
    assert not model.has_prev_phi[0].item()


def test_prompt_has_eight_images_and_wrist_falls_back_to_main(tmp_path):
    model = FakeDopamineGRMRewardModel(
        _cfg(_write_goal_bank(tmp_path)),
        responses=[],
    )
    main = torch.zeros((4, 4, 3), dtype=torch.uint8)
    reference = {
        "main_images": main,
        "reference_start_main_images": main,
    }
    goal = model.goal_bank["id:0"]
    payload = model._build_mode_payload(
        task="insert the peg in the hole",
        reference_start=reference,
        goal_entry=goal,
        before={"main_images": main},
        after={"main_images": main},
    )

    assert len(payload["images"]) == 8
    assert payload["images"][3] is main
    assert payload["images"][4] is main
    assert payload["images"][6] is main
    assert payload["images"][7] is main
    messages = model._build_messages(payload["task"], payload["images"])
    image_parts = [
        part for part in messages[0]["content"] if part.get("type") == "image_url"
    ]
    assert len(image_parts) == 8


def test_openai_compatible_http_endpoint_receives_three_payloads(tmp_path):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers["Content-Length"])
            requests.append(json.loads(self.rfile.read(length)))
            body = json.dumps(
                {"choices": [{"message": {"content": "<score>+25%</score>"}}]}
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *args):
            del args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        model = DopamineGRMRewardModel(
            _cfg(
                _write_goal_bank(tmp_path),
                grm_endpoint=f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
            )
        )
        reward = model.compute_reward(_obs())
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert torch.isfinite(reward).all()
    assert len(requests) == 3
    assert all(request["model"] == "fake-grm" for request in requests)
    assert all(len(request["messages"][0]["content"]) >= 8 for request in requests)


def test_parallel_envs_keep_independent_potentials(tmp_path):
    model = FakeDopamineGRMRewardModel(
        _cfg(_write_goal_bank(tmp_path), num_envs=2),
        responses=[
            ["<score>+20%</score>", "<score>+20%</score>", "<score>-80%</score>"],
            ["<score>+80%</score>", "<score>+80%</score>", "<score>-20%</score>"],
        ],
    )
    image = torch.zeros((2, 4, 4, 3), dtype=torch.uint8)
    observations = {
        "main_images": image,
        "wrist_images": image.clone(),
        "reference_start_main_images": image.clone(),
        "reference_start_wrist_images": image.clone(),
        "task_descriptions": ["put the block in the bowl"] * 2,
        "task_ids": torch.tensor([0, 0]),
        "dones": torch.tensor([False, False]),
        "env_infos": {"episode": {"success_once": torch.tensor([False, False])}},
    }

    rewards = model.compute_reward(observations)

    assert rewards.shape == (2,)
    assert model.prev_phi[0].item() == pytest.approx(0.2)
    assert model.prev_phi[1].item() == pytest.approx(0.8)
    assert rewards[0].item() != pytest.approx(rewards[1].item())
