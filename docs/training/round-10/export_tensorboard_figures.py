#!/usr/bin/env python3
"""Export TensorBoard scalars from round-10 (round-9 walk + rough curriculum)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parent
RUN = Path("/home/june/isaac_projects/spot_arm_locomotion/logs/rsl_rl/rough_spot_with_arm/2026-08-17_15-28-30")
FIG = ROOT / "figures"
DATA = ROOT / "data"
DT_POLICY = 0.02


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


def downsample(y: np.ndarray, n: int = 21) -> list[float]:
    if len(y) <= n:
        return [float(v) for v in y]
    idx = np.linspace(0, len(y) - 1, n).round().astype(int)
    return [float(y[i]) for i in idx]


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    style()
    s = load_scalars(RUN)

    print(f"tags: {len(s)}")
    for tag in sorted(s):
        step, val = s[tag]
        print(f"{tag:56s} n={len(step):5d} first={val[0]:+11.4f} last={val[-1]:+11.4f}")

    def y(tag: str) -> np.ndarray:
        return s[tag][1] if tag in s else np.zeros(1)

    def x(tag: str) -> np.ndarray:
        return s[tag][0] if tag in s else np.zeros(1)

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2))
    fig.suptitle("Round 10 PPO — 2026-08-17_15-28-30  (resume from round-9, rough curriculum)", fontsize=12)

    ax = axes[0, 0]
    ax.plot(x("Train/mean_episode_length"), y("Train/mean_episode_length") * DT_POLICY, color="#1d4ed8")
    ax.axhline(20.0, color="#9ca3af", ls="--", lw=1)
    ax.set_title("Mean episode length")
    ax.set_xlabel("iteration")
    ax.set_ylabel("seconds")

    ax = axes[0, 1]
    for tag, label, color in [
        ("Episode_Termination/time_out", "time_out", "#15803d"),
        ("Episode_Termination/bad_orientation", "bad_orientation", "#b91c1c"),
        ("Episode_Termination/base_contact", "base_contact", "#d97706"),
    ]:
        if tag in s:
            ax.plot(x(tag), y(tag), label=label, color=color)
    ax.set_title("Termination mix")
    ax.set_xlabel("iteration")
    ax.set_ylim(-0.02, 1.05)
    ax.legend()

    ax = axes[0, 2]
    if "Curriculum/terrain_levels" in s:
        ax.plot(x("Curriculum/terrain_levels"), y("Curriculum/terrain_levels"), color="#7c3aed")
    ax.set_title("Terrain level")
    ax.set_xlabel("iteration")

    ax = axes[1, 0]
    ax.plot(x("Metrics/base_velocity/error_vel_xy"), y("Metrics/base_velocity/error_vel_xy"), color="#b91c1c")
    ax.set_title("error_vel_xy")
    ax.set_xlabel("iteration")

    ax = axes[1, 1]
    ax.plot(x("Episode_Reward/track_lin_vel_xy_dot"), y("Episode_Reward/track_lin_vel_xy_dot"), color="#1d4ed8")
    ax.set_title("track_lin_vel_xy_dot")
    ax.set_xlabel("iteration")

    ax = axes[1, 2]
    ax.plot(x("Episode_Reward/foot_clearance"), y("Episode_Reward/foot_clearance"), label="foot_clearance", color="#7c3aed")
    ax.plot(x("Episode_Reward/feet_air_time"), y("Episode_Reward/feet_air_time"), label="feet_air_time", color="#0f766e")
    ax.set_title("Gait terms")
    ax.set_xlabel("iteration")
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIG / "00-dashboard.png")
    plt.close(fig)

    last = {k: float(v[1][-1]) for k, v in s.items()}
    first = {k: float(v[1][0]) for k, v in s.items()}
    metrics = {
        "run": str(RUN),
        "iterations": int(x("Train/mean_reward")[-1]) + 1,
        "first": first,
        "last": last,
    }
    (DATA / "round10_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("wrote", FIG)


if __name__ == "__main__":
    main()
