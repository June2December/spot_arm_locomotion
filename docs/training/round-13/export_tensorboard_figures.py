#!/usr/bin/env python3
"""Export TensorBoard scalars from round-13 (clipped trot), overlaid on round-12."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parent
LOGS = Path("/home/june/isaac_projects/spot_arm_locomotion/logs/rsl_rl/rough_spot_with_arm")
RUN12 = LOGS / "2026-08-25_16-31-31"
FIG = ROOT / "figures"
DATA = ROOT / "data"
DT_POLICY = 0.02

C12 = "#9ca3af"
C13 = "#0f766e"


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
    run13 = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(p for p in LOGS.iterdir() if p.is_dir())[-1]
    FIG.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    style()
    s13 = load_scalars(run13)
    s12 = load_scalars(RUN12)
    pair = ((s12, C12, "round 12"), (s13, C13, "round 13"))

    panels = [
        ("Curriculum/terrain_levels", 1.0, "Terrain level (0-9)"),
        ("Train/mean_episode_length", DT_POLICY, "Mean episode length (s)"),
        ("Metrics/base_velocity/error_vel_xy", 1.0, "error_vel_xy"),
        ("Episode_Reward/gait", 1.0, "gait (new in r13)"),
        ("Episode_Reward/foot_clearance", 1.0, "foot_clearance"),
        ("Policy/mean_std", 1.0, "Policy mean_std"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2))
    fig.suptitle(f"Round 13 — clip + speed limits + trot  ({run13.name})  vs round 12", fontsize=12)
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

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for s, color, name in pair:
        if "Curriculum/terrain_levels" not in s:
            continue
        step, val = s["Curriculum/terrain_levels"]
        ax.plot(step - step[0], val, color=color, lw=1.2, label=name)
    ax.set_title("Terrain level")
    ax.set_xlabel("iterations since resume")
    ax.legend()
    fig.savefig(FIG / "01-terrain-level.png")
    plt.close(fig)

    metrics = {
        "run_round13": str(run13),
        "run_round12": str(RUN12),
        "iterations_round13": int(s13["Train/mean_reward"][0][-1]) + 1 if "Train/mean_reward" in s13 else 0,
        "last_round13": {k: float(v[1][-1]) for k, v in s13.items()},
        "last_round12": {k: float(v[1][-1]) for k, v in s12.items()},
    }
    (DATA / "round13_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("wrote", FIG)


if __name__ == "__main__":
    main()
