#!/usr/bin/env python3
"""Export TensorBoard scalars from round-9 (plane walk, Kp=80)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parent
RUN = Path("/home/june/isaac_projects/spot_arm_locomotion/logs/rsl_rl/rough_spot_with_arm/2026-08-17_14-38-27")
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

    def y(tag: str) -> np.ndarray:
        return s[tag][1]

    def x(tag: str) -> np.ndarray:
        return s[tag][0]

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2))
    fig.suptitle("Round 9 PPO — 2026-08-17_14-38-27  (3000 iter, Kp=80, plane)", fontsize=12)

    ax = axes[0, 0]
    ax.plot(x("Train/mean_episode_length"), y("Train/mean_episode_length") * DT_POLICY, color="#1d4ed8")
    ax.axhline(20.0, color="#9ca3af", ls="--", lw=1)
    ax.set_title("Mean episode length")
    ax.set_xlabel("iteration")
    ax.set_ylabel("seconds")

    ax = axes[0, 1]
    ax.plot(x("Episode_Termination/time_out"), y("Episode_Termination/time_out"), label="time_out", color="#15803d")
    ax.plot(
        x("Episode_Termination/bad_orientation"),
        y("Episode_Termination/bad_orientation"),
        label="bad_orientation",
        color="#b91c1c",
    )
    ax.set_title("Termination mix")
    ax.set_xlabel("iteration")
    ax.set_ylim(-0.02, 1.05)
    ax.legend()

    ax = axes[0, 2]
    ax.plot(x("Train/mean_reward"), y("Train/mean_reward"), color="#111827")
    ax.set_title("Mean episode reward")
    ax.set_xlabel("iteration")

    ax = axes[1, 0]
    ax.plot(x("Episode_Reward/track_lin_vel_xy_dot"), y("Episode_Reward/track_lin_vel_xy_dot"), color="#1d4ed8")
    ax.set_title("track_lin_vel_xy_dot")
    ax.set_xlabel("iteration")

    ax = axes[1, 1]
    ax.plot(x("Episode_Reward/foot_clearance"), y("Episode_Reward/foot_clearance"), color="#7c3aed")
    ax.set_title("foot_clearance")
    ax.set_xlabel("iteration")

    ax = axes[1, 2]
    ax.plot(x("Episode_Reward/feet_air_time"), y("Episode_Reward/feet_air_time"), color="#0f766e")
    ax.set_title("feet_air_time")
    ax.set_xlabel("iteration")

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
        "canvas_series": {
            "iteration": [str(int(v)) for v in downsample(x("Train/mean_reward"), 21)],
            "episode_length_s": downsample(y("Train/mean_episode_length") * DT_POLICY, 21),
            "track_lin_dot": downsample(y("Episode_Reward/track_lin_vel_xy_dot"), 21),
            "foot_clearance": downsample(y("Episode_Reward/foot_clearance"), 21),
            "feet_air_time": downsample(y("Episode_Reward/feet_air_time"), 21),
            "error_vel_xy": downsample(y("Metrics/base_velocity/error_vel_xy"), 21),
            "time_out": downsample(y("Episode_Termination/time_out"), 21),
        },
    }
    (DATA / "round9_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("wrote", FIG)


if __name__ == "__main__":
    main()
