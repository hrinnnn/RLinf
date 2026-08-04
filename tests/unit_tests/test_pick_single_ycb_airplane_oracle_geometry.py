import numpy as np


def test_tilted_airplane_closing_axis_can_be_projected_for_top_down_grasp():
    approach = np.array([0.0, 0.0, -1.0])
    closing = np.array([0.6, 0.2, 0.77])
    closing = closing - approach * float(approach @ closing)
    closing /= np.linalg.norm(closing)
    assert abs(float(approach @ closing)) <= 1e-12
    assert np.isclose(np.linalg.norm(closing), 1.0)
