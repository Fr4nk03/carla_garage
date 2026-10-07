#!/bin/bash
# Pilot: TF++ on friction_ood/routes/pilot10.xml, dry vs. low friction (ego mode), one GPU, sequential.
# Usage: bash friction_ood/scripts/run_pilot.sh [scale ...]   (default: 1.0 0.1)
# Re-running resumes from the checkpoint json (RESUME=True in run_evaluation.sh).
set -e

PROJECT_ROOT=${PROJECT_ROOT:-/home/imrl/Documents/Frank_thesis}
GARAGE=${PROJECT_ROOT}/carla_garage

export CARLA_ROOT=${GARAGE}/carla
# Expose only the NVIDIA Vulkan driver to CARLA. Otherwise -graphicsadapter=0 picks the AMD GPU on this machine
# (Vulkan and CUDA order the GPUs differently) and all sensor rendering runs there.
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export WORK_DIR=${GARAGE}/Bench2Drive
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${WORK_DIR}/scenario_runner":"${WORK_DIR}/leaderboard":"${GARAGE}/team_code":"${GARAGE}":${PYTHONPATH}

ROUTES=${GARAGE}/friction_ood/routes/pilot10.xml
TEAM_AGENT=${GARAGE}/team_code/sensor_agent.py
TEAM_CONFIG=${PROJECT_ROOT}/weights/tfpp/pretrained_models/all_towns
IS_BENCH2DRIVE=True
PLANNER_TYPE=traj
GPU_RANK=0
PORT=30000
TM_PORT=50000
export FRICTION_MODE=ego

SCALES=("$@")
[ ${#SCALES[@]} -eq 0 ] && SCALES=(1.0 0.1)

for SCALE in "${SCALES[@]}"; do
  OUT=${PROJECT_ROOT}/results/pilot/tfpp/${FRICTION_MODE}_mu${SCALE}/seed0
  mkdir -p "${OUT}"
  export FRICTION_SCALE=${SCALE}
  export TICK_LOG_DIR=${OUT}/ticks
  echo "=== TF++ pilot: mode=${FRICTION_MODE} scale=${SCALE} -> ${OUT}"
  bash ${WORK_DIR}/leaderboard/scripts/run_evaluation.sh ${PORT} ${TM_PORT} ${IS_BENCH2DRIVE} ${ROUTES} \
    ${TEAM_AGENT} ${TEAM_CONFIG} ${OUT}/simulation_results.json ${OUT}/agent ${PLANNER_TYPE} ${GPU_RANK} \
    > ${OUT}/eval.log 2>&1
done
