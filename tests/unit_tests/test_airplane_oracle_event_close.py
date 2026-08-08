from __future__ import annotations

from toolkits.lerobot import diagnose_pick_single_ycb_airplane_oracle as oracle


class _Planner:
    def __init__(self) -> None:
        self.calls = 0

    def close_gripper(self, *, t: int) -> None:
        assert t == 1
        self.calls += 1


def test_close_stops_after_stable_grasp(monkeypatch) -> None:
    planner = _Planner()
    grasp_sequence = iter([False, True, True, True, True])
    monkeypatch.setattr(oracle, "_is_grasping", lambda _unwrapped: next(grasp_sequence))

    grasped, executed = oracle._close_gripper_until_stable_grasp(
        planner, object(), max_steps=60, stable_steps=4
    )

    assert grasped
    assert executed == 5
    assert planner.calls == 5


def test_close_resets_stability_after_contact_is_lost(monkeypatch) -> None:
    planner = _Planner()
    grasp_sequence = iter([True, True, False, True, True, True])
    monkeypatch.setattr(oracle, "_is_grasping", lambda _unwrapped: next(grasp_sequence))

    grasped, executed = oracle._close_gripper_until_stable_grasp(
        planner, object(), max_steps=10, stable_steps=3
    )

    assert grasped
    assert executed == 6


def test_close_uses_full_budget_when_grasp_never_stabilizes(monkeypatch) -> None:
    planner = _Planner()
    monkeypatch.setattr(oracle, "_is_grasping", lambda _unwrapped: False)

    grasped, executed = oracle._close_gripper_until_stable_grasp(
        planner, object(), max_steps=7, stable_steps=3
    )

    assert not grasped
    assert executed == 7
    assert planner.calls == 7
