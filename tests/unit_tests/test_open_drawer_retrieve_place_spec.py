import numpy as np

from rlinf.envs.maniskill.open_drawer_retrieve_place_spec import (
    GOAL_OOD_CENTER,
    GRASP_OOD_YAW_RANGE,
    HANDLE_OFFSET_BY_SPLIT,
    ID_GOAL_CENTER,
    ID_OBJECT_YAW_RANGE,
    paired_common_factors,
    sample_episode_parameters,
)


def _sample(split: str):
    return sample_episode_parameters(np.random.default_rng(1701), 256, split=split)


def test_sampling_is_reproducible():
    first = _sample("id")
    second = _sample("id")
    np.testing.assert_allclose(first.drawer_qpos, second.drawer_qpos)
    np.testing.assert_allclose(first.object_local_xy, second.object_local_xy)
    np.testing.assert_allclose(first.object_yaw, second.object_yaw)
    np.testing.assert_allclose(first.goal_xy, second.goal_xy)


def test_paired_splits_preserve_all_common_random_draws():
    reference = paired_common_factors(_sample("id"))
    for split in ("handle_ood", "grasp_ood", "goal_ood"):
        candidate = paired_common_factors(_sample(split))
        for name in reference:
            np.testing.assert_allclose(candidate[name], reference[name])


def test_handle_ood_changes_only_handle_location():
    assert HANDLE_OFFSET_BY_SPLIT["handle_ood"] != HANDLE_OFFSET_BY_SPLIT["id"]
    np.testing.assert_allclose(_sample("handle_ood").object_yaw, _sample("id").object_yaw)
    np.testing.assert_allclose(_sample("handle_ood").goal_xy, _sample("id").goal_xy)


def test_grasp_ood_yaw_is_disjoint_from_id():
    assert ID_OBJECT_YAW_RANGE[1] < GRASP_OOD_YAW_RANGE[0]
    sample = _sample("grasp_ood").object_yaw
    assert np.all(sample >= GRASP_OOD_YAW_RANGE[0])
    assert np.all(sample <= GRASP_OOD_YAW_RANGE[1])


def test_goal_ood_moves_only_the_goal_stage():
    np.testing.assert_allclose(
        _sample("goal_ood").goal_xy - GOAL_OOD_CENTER,
        _sample("id").goal_xy - ID_GOAL_CENTER,
    )
    np.testing.assert_allclose(_sample("goal_ood").object_yaw, _sample("id").object_yaw)
