#!/usr/bin/env bash
# Source Isaac Lab + Isaac Sim so Kit/PhysX are on PYTHONPATH.
# Usage: source scripts/isaaclab_env.sh

_ISAACLAB_ROOT="${ISAACLAB_PATH:-/home/june/IsaacLab}"
_ISAAC_SIM="${_ISAACLAB_ROOT}/_isaac_sim"

if [ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate isaaclab
fi

export ISAACLAB_PATH="${_ISAACLAB_ROOT}"
export CARB_APP_PATH="${_ISAAC_SIM}/kit"
export ISAAC_PATH="${_ISAAC_SIM}"
export EXP_PATH="${_ISAAC_SIM}/apps"

# Newer Isaac Sim ships setup_python_env.sh instead of setup_conda_env.sh.
if [ -f "${_ISAAC_SIM}/setup_python_env.sh" ]; then
    # shellcheck disable=SC1091
    source "${_ISAAC_SIM}/setup_python_env.sh"
elif [ -f "${_ISAAC_SIM}/setup_conda_env.sh" ]; then
    # shellcheck disable=SC1091
    source "${_ISAAC_SIM}/setup_conda_env.sh"
else
    echo "[ERROR] Isaac Sim env setup script not found under ${_ISAAC_SIM}" >&2
    return 1 2>/dev/null || exit 1
fi

# Kit's stdlib on PYTHONPATH shadows conda CPython and breaks platform.py.
_KIT_STDLIB="${_ISAAC_SIM}/kit/python/lib/python3.12"
export PYTHONPATH="${PYTHONPATH//${_KIT_STDLIB}:/}"
