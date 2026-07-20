from __future__ import annotations

from rlinf.data.maniskill_plug_progress import PLUG_PROGRESS_LEVELS


def test_plug_progress_levels_are_ordered_and_bounded():
    values = list(PLUG_PROGRESS_LEVELS.values())
    assert values == sorted(values)
    assert values[0] == 0.0
    assert values[-1] == 1.0
