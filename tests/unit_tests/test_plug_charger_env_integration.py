from __future__ import annotations

import numpy as np
import pytest


def test_controlled_plug_envs_reset_into_their_respective_yaw_ranges():
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("mani_skill")
    import mani_skill.envs  # noqa: F401

    from rlinf.envs.maniskill.plug_charger_variants import (
        PLUG_CHARGER_ID_ENV_ID,
        PLUG_CHARGER_ID_YAW_RANGE,
        PLUG_CHARGER_OOD_ENV_ID,
        PLUG_CHARGER_OOD_YAW_RANGE,
        register_controlled_plug_charger_variants,
        reset_metadata,
    )

    register_controlled_plug_charger_variants()
    for env_id, yaw_range in (
        (PLUG_CHARGER_ID_ENV_ID, PLUG_CHARGER_ID_YAW_RANGE),
        (PLUG_CHARGER_OOD_ENV_ID, PLUG_CHARGER_OOD_YAW_RANGE),
    ):
        env = gym.make(
            env_id,
            num_envs=1,
            obs_mode="rgb",
            control_mode="pd_joint_delta_pos",
            sim_backend="physx_cpu",
            reward_mode="sparse",
            sensor_configs={"width": 64, "height": 64},
        )
        try:
            obs, _ = env.reset(seed=123)
            metadata = reset_metadata(env)
            assert yaw_range[0] <= metadata["relative_yaw"] <= yaw_range[1]
            assert tuple(obs["agent"]["qpos"].shape) == (1, 9)
            assert {"base_camera", "hand_camera"}.issubset(obs["sensor_data"])
        finally:
            env.close()


def test_plug_observation_wrapper_uses_official_camera_keys():
    torch = pytest.importorskip("torch")
    from rlinf.envs.maniskill.plug_charger_variants import wrap_plug_charger_openpi_joint_obs

    raw_obs = {
        "agent": {"qpos": torch.zeros((1, 9))},
        "sensor_data": {
            "base_camera": {"rgb": torch.zeros((1, 8, 8, 3), dtype=torch.uint8)},
            "hand_camera": {"rgb": torch.ones((1, 8, 8, 3), dtype=torch.uint8)},
        },
        "sensor_param": {},
    }
    wrapped = wrap_plug_charger_openpi_joint_obs(raw_obs)
    assert tuple(wrapped["states"].shape) == (1, 9)
    assert tuple(wrapped["main_images"].shape) == (1, 8, 8, 3)
    assert wrapped["task_descriptions"] == ["plug the charger into the receptacle"]
