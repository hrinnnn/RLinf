# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.data.embodied_io_struct import EnvOutput
from rlinf.envs.maniskill.maniskill_rlt_env import ManiskillRLTEnv
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACLossMixin
from rlinf.workers.env.env_worker import _build_reward_done_payload
from toolkits.lerobot.collect_maniskill_peg_lerobot_joint import (
    FrameRecord,
    _build_frames,
    _extract_record,
    _write_grm_goal_bank,
)


def test_env_output_preserves_grm_reference_fields():
    image = torch.zeros((2, 4, 4, 3), dtype=torch.uint8)
    wrist = torch.ones((2, 4, 4, 3), dtype=torch.uint8)
    output = EnvOutput(
        obs={
            "main_images": image,
            "wrist_images": wrist,
            "extra_view_images": None,
            "states": torch.zeros((2, 9)),
            "task_descriptions": ["insert the peg in the hole"] * 2,
            "task_ids": torch.tensor([0, 0]),
            "reference_start_main_images": image + 2,
            "reference_start_wrist_images": wrist + 2,
        }
    ).to_dict()["obs"]

    assert output["task_ids"].tolist() == [0, 0]
    assert torch.equal(output["reference_start_main_images"], image + 2)
    assert torch.equal(output["reference_start_wrist_images"], wrist + 2)


def test_chunk_first_shaping_is_not_discounted_or_repeated():
    dummy = object.__new__(RLTACLossMixin)
    dummy.cfg = OmegaConf.create({"algorithm": {"gamma": 0.96}})
    dummy.torch_dtype = torch.float32
    rewards = torch.zeros((1, 10), dtype=torch.float32)
    rewards[0, 0] = 0.4
    rewards[0, 3] = 1.0

    chunk_return = RLTACLossMixin._discounted_chunk_rewards(dummy, rewards)

    assert chunk_return.item() == pytest.approx(0.4 + 0.96**3)


def test_reward_payload_keeps_mid_chunk_done():
    chunk_dones = torch.tensor(
        [[False, False, True, False], [False, False, False, False]]
    )

    payload = _build_reward_done_payload(chunk_dones)

    assert torch.equal(payload["chunk_dones"], chunk_dones)
    assert payload["dones"].tolist() == [True, False]


def test_partial_reset_updates_only_reset_start_reference():
    env = object.__new__(ManiskillRLTEnv)
    env.reference_start_obs = {
        "main_images": torch.zeros((2, 2, 2, 3), dtype=torch.uint8),
        "wrist_images": torch.ones((2, 2, 2, 3), dtype=torch.uint8),
    }
    reset_obs = {
        "main_images": torch.stack(
            [
                torch.full((2, 2, 3), 3, dtype=torch.uint8),
                torch.full((2, 2, 3), 7, dtype=torch.uint8),
            ]
        ),
        "wrist_images": torch.stack(
            [
                torch.full((2, 2, 3), 5, dtype=torch.uint8),
                torch.full((2, 2, 3), 9, dtype=torch.uint8),
            ]
        ),
    }

    env._update_reference_start_obs(reset_obs, env_idx=torch.tensor([1]))
    attached = env._attach_reference_start_obs(dict(reset_obs))

    assert torch.count_nonzero(env.reference_start_obs["main_images"][0]).item() == 0
    assert torch.all(env.reference_start_obs["main_images"][1] == 7)
    assert torch.all(env.reference_start_obs["wrist_images"][1] == 9)
    assert torch.equal(
        attached["reference_start_main_images"],
        env.reference_start_obs["main_images"],
    )


def test_partial_reset_accepts_subset_observation_batch():
    env = object.__new__(ManiskillRLTEnv)
    env.reference_start_obs = {
        "main_images": torch.zeros((2, 2, 2, 3), dtype=torch.uint8),
    }
    reset_obs = {
        "main_images": torch.full((1, 2, 2, 3), 11, dtype=torch.uint8),
    }

    env._update_reference_start_obs(reset_obs, env_idx=torch.tensor([1]))

    assert torch.count_nonzero(env.reference_start_obs["main_images"][0]).item() == 0
    assert torch.all(env.reference_start_obs["main_images"][1] == 11)


def test_successful_terminal_record_writes_goal_bank(tmp_path):
    main = np.full((8, 8, 3), 17, dtype=np.uint8)
    wrist = np.full((8, 8, 3), 31, dtype=np.uint8)
    record = FrameRecord(
        obs={
            "sensor_data": {
                "3rd_view_camera": {"rgb": main[None]},
                "wide_hand_camera": {"rgb": wrist[None]},
            }
        },
        state=np.zeros(9, dtype=np.float32),
        qpos=np.zeros(9, dtype=np.float32),
    )

    task_dir = _write_grm_goal_bank(
        output_dir=tmp_path / "goal_bank",
        terminal_record=record,
        task="insert the peg in the hole",
        seed=7,
        solver_module="official.peg_solver",
        main_camera="3rd_view_camera",
        wrist_camera="wide_hand_camera",
        target_control_mode="pd_joint_delta_pos",
    )

    assert (task_dir / "goal_main.png").is_file()
    assert (task_dir / "goal_wrist.png").is_file()
    metadata = json.loads((task_dir / "meta.json").read_text(encoding="utf-8"))
    assert metadata["task_id"] == 0
    assert metadata["seed"] == 7
    assert metadata["environment_id"].endswith("ObserverWideWrist-v1")
    assert metadata["views"] == {
        "main": "goal_main.png",
        "wrist": "goal_wrist.png",
    }


def test_extract_record_snapshots_reused_maniskill_rgb_buffers():
    main = np.full((1, 4, 4, 3), 7, dtype=np.uint8)
    wrist = np.full((1, 4, 4, 3), 13, dtype=np.uint8)
    qpos = np.zeros((1, 9), dtype=np.float32)
    observation = {
        "agent": {"qpos": qpos},
        "sensor_data": {
            "base_camera": {"rgb": main},
            "hand_camera": {"rgb": wrist},
        },
    }

    first = _extract_record(observation)

    # Simulate ManiSkill filling the same backing arrays on the next step.
    main.fill(101)
    wrist.fill(211)
    qpos.fill(3.0)
    second = _extract_record(observation)
    frames = _build_frames(
        records=[first, second],
        actions=[np.zeros(8, dtype=np.float32)],
        task="plug the charger into the receptacle",
        main_camera="base_camera",
        wrist_camera="hand_camera",
    )

    assert np.all(frames[0]["image"] == 7)
    assert np.all(frames[0]["wrist_image"] == 13)
    assert np.all(first.qpos == 0.0)
    assert np.all(second.qpos == 3.0)
