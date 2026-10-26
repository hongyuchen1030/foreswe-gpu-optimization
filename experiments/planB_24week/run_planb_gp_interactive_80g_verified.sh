#!/usr/bin/env bash
set -euo pipefail

PLANB_ROOT="/global/homes/h/hyvchen/foreswe-gpu-optimization/experiments/planB_24week"
ENV_PY="/pscratch/sd/h/hyvchen/conda_envs/foreswe_repro_att/bin/python"
LOG_PATH="${PLANB_ROOT}/outputs/planB_24week_gp_interactive_80g_verified.log"

mkdir -p "${PLANB_ROOT}/outputs"

echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "${LOG_PATH}"
echo "cwd=$(pwd)" | tee -a "${LOG_PATH}"
echo "hostname=$(hostname)" | tee -a "${LOG_PATH}"
echo "gpu_inventory_begin" | tee -a "${LOG_PATH}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "${LOG_PATH}"
echo "gpu_inventory_end" | tee -a "${LOG_PATH}"

PLANB_GP_SMOKE=0 \
PLANB_GP_EPOCHS=2 \
PLANB_GP_BATCH=4096 \
PLANB_GP_INDUCING=7000 \
DEVICE=cuda \
  "${ENV_PY}" "${PLANB_ROOT}/planB_24week_gp_model.py" 2>&1 | tee -a "${LOG_PATH}"
