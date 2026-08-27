#!/usr/bin/env python3
"""Export TensorBoard scalars from round-11 (terrain-relative foot clearance).

Round 10 is overlaid so the effect of the one changed term is visible.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parent
LOGS = Path("/home/june/isaac_projects/spot_arm_locomotion/logs/rsl_rl/rough_spot_with_arm")
RUN10 = LOGS / "2026-08-17_15-28-30"
FIG = ROOT / "figures"
DATA = ROOT / "data"
DT_POLICY = 0.02

C10 = "#9ca3af"
C11 = "#1d4ed8"


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
    run11 = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(p for p in LOGS.iterdir() if p.is_dir())[-1]
    FIG.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    style()
    s11 = load_scalars(run11)
    s10 = load_scalars(RUN10)

    # Round 11 resumes mid-run; plot both against iterations since the resume so
    # the two curricula (which both restart at level 0) line up.
    def plot(ax, tag: str, mult: float = 1.0, label: str | None = None) -> None:
        for s, color, name in ((s10, C10, "round 10"), (s11, C11, "round 11")):
            if tag not in s:
                continue
            step, val = s[tag]
            ax.plot(step - step[0], val * mult, color=color, lw=1.2, label=name if label is None else f"{name} {label}")

    panels = [
        ("Curriculum/terrain_levels", 1.0, "Terrain level (0-9)"),
        ("Train/mean_episode_length", DT_POLICY, "Mean episode length (s)"),
        ("Episode_Reward/foot_clearance", 1.0, "foot_clearance"),
        ("Episode_Reward/feet_air_time", 1.0, "feet_air_time"),
        ("Metrics/base_velocity/error_vel_xy", 1.0, "error_vel_xy"),
        ("Policy/mean_std", 1.0, "Policy mean_std"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2))
    fig.suptitle(f"Round 11 — terrain-relative foot clearance  ({run11.name})  vs round 10", fontsize=12)
    for ax, (tag, mult, title) in zip(axes.ravel(), panels):
        plot(ax, tag, mult)
        ax.set_title(title)
        ax.set_xlabel("iterations since resume")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(FIG / "00-dashboard.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for s, color, name in ((s10, C10, "round 10 (world z)"), (s11, C11, "round 11 (terrain-relative)")):
        step, val = s["Episode_Reward/foot_clearance"]
        ax.plot(step - step[0], val, color=color, lw=1.2, label=name)
    ax.set_title("foot_clearance: same magnitude, now earned by lifting a foot")
    ax.set_xlabel("iterations since resume")
    ax.legend()
    fig.savefig(FIG / "01-foot-clearance.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for s, color, name in ((s10, C10, "round 10"), (s11, C11, "round 11")):
        step, val = s["Curriculum/terrain_levels"]
        ax.plot(step - step[0], val, color=color, lw=1.2, label=name)
    ax.axhline(6.06, color="#b91c1c", ls="--", lw=1, label="round 10 plateau")
    ax.set_title("Terrain level")
    ax.set_xlabel("iterations since resume")
    ax.legend()
    fig.savefig(FIG / "02-terrain-level.png")
    plt.close(fig)

    def summary(s: dict) -> dict:
        return {k: float(v[1][-1]) for k, v in s.items()}

    metrics = {
        "run_round11": str(run11),
        "run_round10": str(RUN10),
        "iterations_round11": int(s11["Train/mean_reward"][0][-1]) + 1,
        "last_round11": summary(s11),
        "last_round10": summary(s10),
    }
    (DATA / "round11_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("wrote", FIG)


if __name__ == "__main__":
    main()
