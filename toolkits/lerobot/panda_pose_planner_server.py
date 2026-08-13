#!/usr/bin/env python3
"""Isolated Panda pose planner for scenes that contain custom articulations."""

from __future__ import annotations

import json
import os
import site
import sys

external_site_packages = os.environ.get("PANDA_PLANNER_SITE_PACKAGES")
if external_site_packages:
    site.addsitedir(external_site_packages)

import gymnasium as gym
import numpy as np
import sapien

import mani_skill.envs  # noqa: F401
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)


def main() -> None:
    env = gym.make(
        "PickCube-v1",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode=None,
        sim_backend="cpu",
    )
    env.reset(seed=0)
    base = env.unwrapped
    solver = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=base.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )
    print("READY", flush=True)
    try:
        for line in sys.stdin:
            request = json.loads(line)
            if request.get("command") == "close":
                break
            target_world = sapien.Pose(request["target_p"], request["target_q"])
            planning_pose = solver._transform_pose_for_planning(target_world)
            target = np.concatenate([planning_pose.p, planning_pose.q])
            qpos = np.asarray(request["qpos"], dtype=np.float64)
            timestep = float(request["time_step"])
            result = solver.planner.plan_screw(
                target,
                qpos,
                time_step=timestep,
                use_point_cloud=False,
            )
            if result.get("status") != "Success":
                result = solver.planner.plan_qpos_to_pose(
                    target,
                    qpos,
                    time_step=timestep,
                    wrt_world=True,
                )
            response = {"status": result.get("status", "Unknown")}
            if result.get("status") == "Success":
                response["positions"] = np.asarray(result["position"]).tolist()
            print("RESULT " + json.dumps(response, separators=(",", ":")), flush=True)
    finally:
        solver.close()
        env.close()


if __name__ == "__main__":
    main()
