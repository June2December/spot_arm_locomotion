#!/usr/bin/env python3
"""Print training gate metrics from a TensorBoard run (any round).

Usage::

    python docs/training/watch_gates.py [run_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOGS = Path("/home/june/isaac_projects/spot_arm_locomotion/logs/rsl_rl/rough_spot_with_arm")
DT_POLICY = 0.02

# Standard gates — see TRAINING.md
GATES = [
    ("Curriculum/terrain_levels", "terrain", 1.0),
    ("Train/mean_episode_length", "episode_s", DT_POLICY),
    ("Episode_Termination/time_out", "time_out", 1.0),
    ("Episode_Termination/base_contact", "base_hit", 1.0),
    ("Episode_Termination/bad_orientation", "bad_orient", 1.0),
    ("Metrics/base_velocity/error_vel_xy", "err_vxy", 1.0),
    ("Episode_Reward/gait", "gait", 1.0),
    ("Episode_Reward/foot_clearance", "foot_clr", 1.0),
    ("Episode_Reward/feet_air_time", "air_time", 1.0),
    ("Episode_Reward/leg_joint_deviation", "leg_dev", 1.0),
    ("Episode_Reward/hip_x_deviation", "hip_x_dev", 1.0),
    ("Episode_Reward/track_lin_vel_xy_dot", "cmd_dot_v", 1.0),
    ("Policy/mean_std", "mean_std", 1.0),
    ("Loss/entropy", "entropy", 1.0),
    ("Loss/learning_rate", "lr", 1.0),
    ("Train/mean_reward", "mean_rw", 1.0),
]


def main() -> None:
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(p for p in LOGS.iterdir() if p.is_dir())[-1]
    ea = EventAccumulator(str(run), size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags().get("scalars", []))
    if "Train/mean_reward" not in tags:
        print(f"{run.name}: no scalars yet")
        return

    steps = np.array([e.step for e in ea.Scalars("Train/mean_reward")])
    print(f"{run.name}  iter {steps[0]} -> {steps[-1]}  ({len(steps)} logged)")
    print("See docs/training/TRAINING.md for what each column means.\n")
    picks = np.unique(np.linspace(0, len(steps) - 1, min(9, len(steps))).round().astype(int))

    header = f"{'iter':>6} " + " ".join(f"{name:>10}" for _, name, _ in GATES)
    print(header)
    print("-" * len(header))
    series = {}
    for tag, name, mult in GATES:
        series[name] = np.array([e.value for e in ea.Scalars(tag)]) * mult if tag in tags else None
    for i in picks:
        row = f"{steps[i]:6d} "
        for _, name, _ in GATES:
            v = series[name]
            row += f" {float(v[i]):10.4f}" if v is not None and i < len(v) else f" {'-':>10}"
        print(row)


if __name__ == "__main__":
    main()
