import numpy as np

from rlinf.envs.maniskill.stack_cube_variants import (
    STACK_CUBE_ID_ANGLE_HALF_WIDTH,
    STACK_CUBE_ID_BASE_JITTER,
    STACK_CUBE_ID_DISTANCE_RANGE,
    geometry_is_stack_cube_id,
    geometry_is_stack_cube_ood,
    sample_stack_cube_id_xy,
    sample_stack_cube_xy,
    stack_cube_id_geometry,
)


def test_stack_cube_id_samples_are_reproducible_and_in_range():
    first = sample_stack_cube_id_xy(np.random.default_rng(17), 256)
    second = sample_stack_cube_id_xy(np.random.default_rng(17), 256)
    np.testing.assert_allclose(first[0], second[0])
    np.testing.assert_allclose(first[1], second[1])
    assert np.all(np.abs(first[0]) <= STACK_CUBE_ID_BASE_JITTER)
    assert np.all(geometry_is_stack_cube_id(first[1], first[0]))


def test_stack_cube_id_geometry_rejects_opposite_direction():
    cube_b = np.array([[0.0, 0.0]])
    cube_a = np.array([[0.0, 0.09]])
    opposite = np.array([[0.0, -0.09]])
    assert geometry_is_stack_cube_id(cube_a, cube_b).item()
    assert not geometry_is_stack_cube_id(opposite, cube_b).item()


def test_stack_cube_id_geometry_uses_narrow_sector_and_distance():
    base, obj = sample_stack_cube_id_xy(np.random.default_rng(3), 512)
    geometry = stack_cube_id_geometry(obj, base)
    assert geometry["distance"].min() >= STACK_CUBE_ID_DISTANCE_RANGE[0]
    assert geometry["distance"].max() <= STACK_CUBE_ID_DISTANCE_RANGE[1]
    assert np.abs(geometry["angle_offset"]).max() <= STACK_CUBE_ID_ANGLE_HALF_WIDTH


def test_stack_cube_ood_is_opposite_and_disjoint_from_id():
    base, obj = sample_stack_cube_xy(np.random.default_rng(23), 512, split="ood")
    assert np.all(geometry_is_stack_cube_ood(obj, base))
    assert not np.any(geometry_is_stack_cube_id(obj, base))
