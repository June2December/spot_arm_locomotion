#!/usr/bin/env python3
"""Export TensorBoard scalars from round-15, overlaid on round-14."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parent
LOGS = Path("/home/june/isaac_projects/spot_arm_locomotion/logs/rsl_rl/rough_spot_with_arm")
RUN14 = LOGS / "2026-08-26_12-52-45"
FIG = ROOT / "figures"
DATA = ROOT / "data"
DT_POLICY = 0.02

C14 = "#9ca3af"
C15 = "#0f766e"


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
    run15 = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(p for p in LOGS.iterdir() if p.is_dir())[-1]
    FIG.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    style()
    s15 = load_scalars(run15)
    s14 = load_scalars(RUN14)
    pair = ((s14, C14, "round 14"), (s15, C15, "round 15"))

    panels = [
        ("Curriculum/terrain_levels", 1.0, "Terrain level (0-9)"),
        ("Train/mean_episode_length", DT_POLICY, "Mean episode length (s)"),
        ("Episode_Reward/leg_joint_deviation", 1.0, "leg_joint_deviation (hip_y+knee)"),
        ("Episode_Reward/gait", 1.0, "gait"),
        ("Policy/mean_std", 1.0, "Policy mean_std"),
        ("Metrics/base_velocity/error_vel_xy", 1.0, "error_vel_xy"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2))
    fig.suptitle(f"Round 15 — knee default pull  ({run15.name})  vs round 14", fontsize=12)
    for ax, (tag, mult, title) in zip(axes.ravel(), panels):
        for s, color, name in pair:
            if tag not in s:
                continue
            step, val = s[tag]
            ax.plot(step - step[0], val * mult, color=color, lw=1.2, label=name)
        ax.set_title(title)
        ax.set_xlabel("iterations since resume")
        if tag == "Policy/mean_std":
            ax.set_ylim(0, 3)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(FIG / "00-dashboard.png")
    plt.close(fig)

    metrics = {
        "run_round15": str(run15),
        "run_round14": str(RUN14),
        "iterations_round15": int(s15["Train/mean_reward"][0][-1]) + 1 if "Train/mean_reward" in s15 else 0,
        "last_round15": {k: float(v[1][-1]) for k, v in s15.items()},
        "last_round14": {k: float(v[1][-1]) for k, v in s14.items()},
    }
    (DATA / "round15_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("wrote", FIG)


if __name__ == "__main__":
    main()
