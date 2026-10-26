#!/usr/bin/env bash
set -euo pipefail

PLANB_ROOT="/global/homes/h/hyvchen/foreswe-gpu-optimization/experiments/planB_24week"
REPO_ROOT="/global/homes/h/hyvchen/foreswe-gpu-optimization"
ENV_PY="/pscratch/sd/h/hyvchen/conda_envs/foreswe_repro_att/bin/python"
HF_CACHE="/pscratch/sd/h/hyvchen/foreswe_model_cache"
TMPDIR_ROOT="/pscratch/sd/h/hyvchen/tmp"
WATCH_LOG="${PLANB_ROOT}/outputs/planB_chain_watch.log"
ATTN_DONE_FILE="${PLANB_ROOT}/results/predict/2018_weekly_window_predict.pkl"
START_TS="$(date +%s)"

mkdir -p "${TMPDIR_ROOT}"
mkdir -p "${PLANB_ROOT}/outputs"

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) watch_start start_ts=${START_TS}" >> "${WATCH_LOG}"

while true; do
  NOW="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ -f "${ATTN_DONE_FILE}" ]]; then
    MT="$(stat -c %Y "${ATTN_DONE_FILE}")"
    echo "${NOW} attention_poll mtime=${MT} start_ts=${START_TS}" >> "${WATCH_LOG}"
    if [[ "${MT}" -gt "${START_TS}" ]]; then
      echo "${NOW} attention_outputs_refreshed launching_gp80" >> "${WATCH_LOG}"
      break
    fi
  else
    echo "${NOW} attention_poll file_missing" >> "${WATCH_LOG}"
  fi
  sleep 1800
done

salloc -N 1 -n 1 -c 32 -t 04:00:00 -C "gpu&hbm80g" --gpus-per-node=1 -q interactive -A m2637_g \
  /usr/bin/bash -lc "
    cd '${REPO_ROOT}' && \
    export FSW_HF_CACHE='${HF_CACHE}' && \
    export TMPDIR='${TMPDIR_ROOT}' && \
    mkdir -p \"\$TMPDIR\" && \
    '${ENV_PY}' '${PLANB_ROOT}/planB_24week_gp_model.py' 2>&1 | tee '${PLANB_ROOT}/outputs/planB_24week_gp_interactive_80g.log'
  "
