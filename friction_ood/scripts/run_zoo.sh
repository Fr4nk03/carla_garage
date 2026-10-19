#!/bin/bash
# Bench2DriveZoo agents (UniAD-Base, VAD) on a route file, dry vs. low friction (ego mode), one GPU, sequential.
# Usage: MODEL=uniad|vad [SEED=k] [ROUTES=file.xml] [EXP=name] [RECORD=1] bash friction_ood/scripts/run_zoo.sh [scale ...]
#   defaults: scales 1.0 0.1, SEED=0, ROUTES=friction_ood/routes/pilot10.xml, EXP=pilot -> results/<EXP>/<MODEL>/...
#   Needs the b2d_zoo conda env (envs/install_b2d_zoo.sh). Runs from carla_garage/Bench2Drive, where the agents
#   expect ./Bench2DriveZoo and ./data (links created by the install script).
# Note: set ZOO_SAVE_EVERY=N to save the agents' camera images every Nth step to <out>/agent (off by default).
set -e

PROJECT_ROOT=${PROJECT_ROOT:-/home/imrl/Documents/Frank_thesis}
GARAGE=${PROJECT_ROOT}/carla_garage
MODEL=${MODEL:?set MODEL=uniad or MODEL=vad}

source "${CONDA:-$HOME/anaconda3}/etc/profile.d/conda.sh"
conda activate b2d_zoo

export CARLA_ROOT=${GARAGE}/carla
# Expose only the NVIDIA Vulkan driver to CARLA (see run_pilot.sh).
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export HF_HUB_OFFLINE=1
# CARLA rendering six 1600x900 cameras takes ~8 GB; let PyTorch reuse fragmented memory so UniAD fits beside it.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# The agents' 50 m visualisation camera is off unless ZOO_BEV_CAMERA=1 (agent patch in envs/b2d_zoo_patches.diff).
export ZOO_BEV_CAMERA=${ZOO_BEV_CAMERA:-0}
# Saving the six camera images every step is slow (PNG, 1600x900); off by default, ZOO_SAVE_EVERY=N saves every Nth.
export ZOO_SAVE_EVERY=${ZOO_SAVE_EVERY:-0}
export WORK_DIR=${GARAGE}/Bench2Drive
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${WORK_DIR}/scenario_runner":"${WORK_DIR}/leaderboard":"${WORK_DIR}/Bench2DriveZoo":"${GARAGE}":${PYTHONPATH}

case ${MODEL} in
  uniad)
    TEAM_AGENT=${WORK_DIR}/Bench2DriveZoo/team_code/uniad_b2d_agent.py
    TEAM_CONFIG=Bench2DriveZoo/adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py+${PROJECT_ROOT}/weights/uniad_base_b2d/uniad_base_b2d.pth ;;
  vad)
    TEAM_AGENT=${WORK_DIR}/Bench2DriveZoo/team_code/vad_b2d_agent.py
    TEAM_CONFIG=Bench2DriveZoo/adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py+${PROJECT_ROOT}/weights/vad_b2d/vad_b2d_base.pth ;;
  *) echo "unknown MODEL=${MODEL}"; exit 1 ;;
esac

ROUTES=${ROUTES:-${GARAGE}/friction_ood/routes/pilot10.xml}
EXP=${EXP:-pilot}
IS_BENCH2DRIVE=True
PLANNER_TYPE=traj
GPU_RANK=0
PORT=30000
# Outside the OS ephemeral port range (32768-60999 here), so no other program can be holding it
TM_PORT=8000
export FRICTION_MODE=ego
SEED=${SEED:-0}
export TM_SEED=${SEED}

SCALES=("$@")
[ ${#SCALES[@]} -eq 0 ] && SCALES=(1.0 0.1)

cd ${WORK_DIR}  # the agents import their plugins from ./Bench2DriveZoo and read ./data
for SCALE in "${SCALES[@]}"; do
  OUT=${PROJECT_ROOT}/results/${EXP}/${MODEL}/${FRICTION_MODE}_mu${SCALE}/seed${SEED}
  mkdir -p "${OUT}"
  # Refuse to mix route files in one output folder: results, logs and resume state would be overwritten.
  if [ -f "${OUT}/routes_file.txt" ] && [ "$(cat "${OUT}/routes_file.txt")" != "${ROUTES}" ]; then
    echo "ERROR: ${OUT} was created for $(cat "${OUT}/routes_file.txt"), not ${ROUTES}; use another EXP."; exit 1
  fi
  echo "${ROUTES}" > "${OUT}/routes_file.txt"
  if [ "${RECORD:-0}" = 1 ]; then mkdir -p "${OUT}/recording"; export RECORD_PATH=${OUT}/recording; else export RECORD_PATH=; fi
  export FRICTION_SCALE=${SCALE}
  export TICK_LOG_DIR=${OUT}/ticks
  echo "=== ${MODEL}: mode=${FRICTION_MODE} scale=${SCALE} -> ${OUT}"
  bash ${WORK_DIR}/leaderboard/scripts/run_evaluation.sh ${PORT} ${TM_PORT} ${IS_BENCH2DRIVE} ${ROUTES} \
    ${TEAM_AGENT} ${TEAM_CONFIG} ${OUT}/simulation_results.json ${OUT}/agent ${PLANNER_TYPE} ${GPU_RANK} \
    > ${OUT}/eval.log 2>&1
done
