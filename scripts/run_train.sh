#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/isaaclab_env.sh"
cd "${ROOT}"
exec python scripts/train.py --task Isaac-Velocity-Rough-Spot-Arm-v0 "$@"
