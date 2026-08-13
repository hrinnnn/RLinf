#!/usr/bin/env python3
"""Validate OpenDrawerRetrievePlace oracle continuation after mid-episode takeover."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

RLINF_ROOT = Path(__file__).resolve().parents[2]
if str(RLINF_ROOT) not in sys.path:
    sys.path.insert(0, str(RLINF_ROOT))

from rlinf.envs.maniskill.open_drawer_retrieve_place_spec import (  # noqa: E402
    DRAWER_OPEN_THRESHOLD,
    ENV_IDS,
)
from toolkits.lerobot.validate_open_drawer_retrieve_place_oracle import (  # noqa: E402
    PandaPosePlannerClient,
    continue_episode,
    solve_episode,
)


class TakeoverPause(RuntimeError):
    pass


class PredicatePauseEnv:
    """Proxy that pauses the underlying simulator exactly at a task predicate."""

    def __init__(self, env: Any, predicate: Callable[[Any], bool]):
        self._env = env
        self._predicate = predicate

    @property
    def unwrapped(self):
        return self._env.unwrapped

    def reset(self, *args, **kwargs):
        return self._env.reset(*args, **kwargs)

    def step(self, action, *args, **kwargs):
        result = self._env.step(action, *args, **kwargs)
        if self._predicate(self._env.unwrapped):
            raise TakeoverPause
        return result


def _scalar(value: Any) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value).reshape(-1)[0])


def _qpos_open(base: Any) -> bool:
    if hasattr(base.drawer.get_qpos(), "detach"):
        value = base.drawer.get_qpos().detach().cpu().numpy()
    else:
        value = np.asarray(base.drawer.get_qpos())
    return float(np.asarray(value).reshape(-1)[0]) <= -DRAWER_OPEN_THRESHOLD


def _drawer_open_predicate(base: Any) -> bool:
    return _qpos_open(base) and not _scalar(base.agent.is_grasping(base.obj))


def _object_grasp_predicate(base: Any) -> bool:
    return _scalar(base.agent.is_grasping(base.obj))


def _run_case(env: Any, planner: PandaPosePlannerClient, seed: int, name: str, predicate):
    proxy = PredicatePauseEnv(env, predicate)
    try:
        solve_episode(proxy, seed, planner)
    except TakeoverPause:
        pass
    else:
        raise RuntimeError(f"seed {seed} never reached takeover predicate {name}")
    result = continue_episode(env, planner, seed=seed)
    return {
        "seed": seed,
        "takeover_stage": name,
        "success": bool(result.get("success", False)),
        "result": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-drawer-open", type=int, default=76000)
    parser.add_argument("--seed-object-grasp", type=int, default=76001)
    args = parser.parse_args()

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import rlinf.envs.maniskill.open_drawer_retrieve_place  # noqa: F401

    env = gym.make(
        ENV_IDS["id"],
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode=None,
        sim_backend="cpu",
        max_episode_steps=400,
    )
    planner = PandaPosePlannerClient()
    results = []
    try:
        env.reset(seed=args.seed_drawer_open)
        results.append(
            _run_case(
                env,
                planner,
                args.seed_drawer_open,
                "after_drawer_opened",
                _drawer_open_predicate,
            )
        )
        env.reset(seed=args.seed_object_grasp)
        results.append(
            _run_case(
                env,
                planner,
                args.seed_object_grasp,
                "after_object_grasped",
                _object_grasp_predicate,
            )
        )
    finally:
        planner.close()
        env.close()

    summary = {
        "task": "OpenDrawerRetrievePlace",
        "split": "id",
        "cases": results,
        "successes": sum(int(case["success"]) for case in results),
        "total": len(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if summary["successes"] != summary["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
