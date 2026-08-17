#!/usr/bin/env python3
"""Export TensorBoard scalars from round-5 (move, no gait) into PNG + JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parent
RUN_A = Path("/home/june/isaac_projects/spot_arm_locomotion/logs/rsl_rl/rough_spot_with_arm/2026-08-17_12-43-51")
RUN_B = Path("/home/june/isaac_projects/spot_arm_locomotion/logs/rsl_rl/rough_spot_with_arm/2026-08-17_13-23-37")
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


def merge_runs(a: dict, b: dict) -> dict:
    """Use run A up to the resume point, then run B (1750–2999)."""
    out = {}
    tags = set(a) | set(b)
    for tag in tags:
        if tag not in a:
            out[tag] = b[tag]
            continue
        if tag not in b:
            out[tag] = a[tag]
            continue
        xa, ya = a[tag]
        xb, yb = b[tag]
        cut = int(xb[0])
        keep = xa < cut
        out[tag] = (np.concatenate([xa[keep], xb]), np.concatenate([ya[keep], yb]))
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
    return [float(y[i]) for v in y]


def downsample_arr(y: np.ndarray, n: int = 21) -> list[float]:
    if len(y) <= n:
        return [float(v) for v in y]
    idx = np.linspace(0, len(y) - 1, n).round().astype(int)
    return [float(y[i]) for i in idx]


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    style()
    s = merge_runs(load_scalars(RUN_A), load_scalars(RUN_B))

    def y(tag: str) -> np.ndarray:
        return s[tag][1]

    def x(tag: str) -> np.ndarray:
        return s[tag][0]

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2))
    fig.suptitle("Round 5 PPO — move without lifting feet (3000 iter, 4096 envs, plane)", fontsize=12)

    ax = axes[0, 0]
    ax.plot(x("Train/mean_episode_length"), y("Train/mean_episode_length") * DT_POLICY, color="#1d4ed8")
    ax.axhline(20.0, color="#9ca3af", ls="--", lw=1, label="timeout 20 s")
    ax.set_title("Mean episode length")
    ax.set_xlabel("iteration")
    ax.set_ylabel("seconds")
    ax.legend(loc="lower right")

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
    ax.set_title("track_lin_vel_xy_dot (cmd · vel)")
    ax.set_xlabel("iteration")

    ax = axes[1, 1]
    ax.plot(x("Metrics/base_velocity/error_vel_xy"), y("Metrics/base_velocity/error_vel_xy"), color="#7c3aed")
    ax.axhline(1.0, color="#9ca3af", ls="--", lw=1, label="~command mag if v=0")
    ax.set_title("xy velocity error")
    ax.set_xlabel("iteration")
    ax.legend()

    ax = axes[1, 2]
    ax.plot(x("Episode_Reward/feet_air_time"), y("Episode_Reward/feet_air_time"), color="#0f766e")
    ax.set_title("feet_air_time (stays ~0)")
    ax.set_xlabel("iteration")

    fig.tight_layout()
    fig.savefig(FIG / "00-dashboard.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(x("Metrics/base_velocity/error_vel_xy"), y("Metrics/base_velocity/error_vel_xy"), color="#7c3aed")
    ax.axhline(1.0, color="#9ca3af", ls="--", lw=1)
    ax.set_title("Velocity error dropped — the body is moving")
    ax.set_xlabel("iteration")
    ax.set_ylabel("error_vel_xy")
    fig.savefig(FIG / "01-velocity-error.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(x("Episode_Reward/track_lin_vel_xy_dot"), y("Episode_Reward/track_lin_vel_xy_dot"), color="#1d4ed8")
    ax.set_title("cmd · vel rose from 0 to ~1.8")
    ax.set_xlabel("iteration")
    fig.savefig(FIG / "02-cmd-dot-vel.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(x("Episode_Reward/feet_air_time"), y("Episode_Reward/feet_air_time"), color="#0f766e")
    ax.set_title("No steps: feet_air_time never leaves ~0")
    ax.set_xlabel("iteration")
    fig.savefig(FIG / "03-feet-air-time.png")
    plt.close(fig)

    last = {k: float(v[1][-1]) for k, v in s.items()}
    first = {k: float(v[1][0]) for k, v in s.items()}
    metrics = {
        "run_a": str(RUN_A),
        "run_b": str(RUN_B),
        "iterations": int(x("Train/mean_reward")[-1]) + 1,
        "first": first,
        "last": last,
        "canvas_series": {
            "iteration": [str(int(v)) for v in downsample_arr(x("Train/mean_reward"), 21)],
            "episode_length_s": downsample_arr(y("Train/mean_episode_length") * DT_POLICY, 21),
            "track_lin_dot": downsample_arr(y("Episode_Reward/track_lin_vel_xy_dot"), 21),
            "track_lin_xy": downsample_arr(y("Episode_Reward/track_lin_vel_xy_exp"), 21),
            "error_vel_xy": downsample_arr(y("Metrics/base_velocity/error_vel_xy"), 21),
            "time_out": downsample_arr(y("Episode_Termination/time_out"), 21),
            "feet_air_time": downsample_arr(y("Episode_Reward/feet_air_time"), 21),
            "mean_std": downsample_arr(y("Policy/mean_std"), 21),
            "mean_reward": downsample_arr(y("Train/mean_reward"), 21),
        },
    }
    (DATA / "round5_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("wrote", FIG)


if __name__ == "__main__":
    main()
