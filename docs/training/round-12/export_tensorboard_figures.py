#!/usr/bin/env python3
"""Export TensorBoard scalars from round-12 (default-pose pull), overlaid on round-11."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parent
LOGS = Path("/home/june/isaac_projects/spot_arm_locomotion/logs/rsl_rl/rough_spot_with_arm")
RUN11 = LOGS / "2026-08-25_15-27-52"
FIG = ROOT / "figures"
DATA = ROOT / "data"
DT_POLICY = 0.02

C11 = "#9ca3af"
C12 = "#1d4ed8"


def load_scalars(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    ea = EventAccumulator(str(path), size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags().get("scalars", []):
        evs = ea.Scalars(tag)
        out[tag] = (
            np.array([e.step for e in evs], dtype=np.int32),
            np.array([e.value for e in evs], dtype=np.float64),
        )
    return out


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
            "figure.dpi": 140,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def main() -> None:
    run12 = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(p for p in LOGS.iterdir() if p.is_dir())[-1]
    FIG.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    style()
    s12 = load_scalars(run12)
    s11 = load_scalars(RUN11)
    pair = ((s11, C11, "round 11"), (s12, C12, "round 12"))

    panels = [
        ("Curriculum/terrain_levels", 1.0, "Terrain level (0-9)"),
        ("Train/mean_episode_length", DT_POLICY, "Mean episode length (s)"),
        ("Metrics/base_velocity/error_vel_xy", 1.0, "error_vel_xy"),
        ("Episode_Reward/foot_clearance", 1.0, "foot_clearance"),
        ("Episode_Reward/dof_pos_limits", 1.0, "dof_pos_limits (legs only in r12)"),
        ("Policy/mean_std", 1.0, "Policy mean_std"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2))
    fig.suptitle(f"Round 12 — default-pose pull on the legs  ({run12.name})  vs round 11", fontsize=12)
    for ax, (tag, mult, title) in zip(axes.ravel(), panels):
        for s, color, name in pair:
            if tag not in s:
                continue
            step, val = s[tag]
            ax.plot(step - step[0], val * mult, color=color, lw=1.2, label=name)
        ax.set_title(title)
        ax.set_xlabel("iterations since resume")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(FIG / "00-dashboard.png")
    plt.close(fig)

    # The two new terms only exist in round 12; plot the deviation they measure.
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for tag, weight, label, color in (
        ("Episode_Reward/hip_x_deviation", -1.0, "hip_x deviation", "#b91c1c"),
        ("Episode_Reward/leg_joint_deviation", -0.2, "all 12 leg joints", "#7c3aed"),
    ):
        if tag not in s12:
            continue
        step, val = s12[tag]
        ax.plot(step - step[0], val / weight, color=color, lw=1.3, label=label)
    ax.set_title("Leg deviation from the default pose (rad, summed over joints)")
    ax.set_xlabel("iterations since resume")
    ax.set_ylabel("rad")
    ax.legend()
    fig.savefig(FIG / "01-leg-deviation.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for s, color, name in pair:
        step, val = s["Curriculum/terrain_levels"]
        ax.plot(step - step[0], val, color=color, lw=1.2, label=name)
    ax.set_title("Terrain level — posture pull did not cost the curriculum")
    ax.set_xlabel("iterations since resume")
    ax.legend()
    fig.savefig(FIG / "02-terrain-level.png")
    plt.close(fig)

    metrics = {
        "run_round12": str(run12),
        "run_round11": str(RUN11),
        "iterations_round12": int(s12["Train/mean_reward"][0][-1]) + 1,
        "last_round12": {k: float(v[1][-1]) for k, v in s12.items()},
        "last_round11": {k: float(v[1][-1]) for k, v in s11.items()},
    }
    (DATA / "round12_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("wrote", FIG)


if __name__ == "__main__":
    main()
