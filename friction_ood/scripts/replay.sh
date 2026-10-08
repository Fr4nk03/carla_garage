#!/bin/bash
# Opens a CARLA window on the NVIDIA GPU and replays a recording with the camera following the ego.
# Usage: bash friction_ood/scripts/replay.sh <recording.log> [replay.py options, e.g. --speed 0.5 --view top]
# Leave the CARLA window open and re-run replay.py to watch again; close the window to quit.
set -e
GARAGE=$(cd "$(dirname "$0")/../.." && pwd)
LOG=$(realpath "$1"); shift

# Render on the NVIDIA GPU even when the desktop runs on another GPU (PRIME render offload).
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __NV_PRIME_RENDER_OFFLOAD=1
export PYTHONPATH="${GARAGE}/carla/PythonAPI/carla/":${PYTHONPATH}

if ! ss -ltn | grep -q ":2000 "; then
  "${GARAGE}/carla/CarlaUE4.sh" -windowed -ResX=1280 -ResY=720 -nosound -carla-rpc-port=2000 > /tmp/carla_replay.log 2>&1 &
  echo "Starting CARLA window..."
  until ss -ltn | grep -q ":2000 "; do sleep 2; done
  sleep 10
fi
python "${GARAGE}/friction_ood/scripts/replay.py" "${LOG}" "$@"
