#!/usr/bin/env bash
# Create a fresh, isolated output directory for a (re)run of an upstream-style ForeSWE script.
# Upstream scripts read {BASE_URL}/data and write {BASE_URL}/{checkpoints,outputs,results};
# pointing BASE_URL at a run dir keeps every write out of the repo and away from historical runs.
#
#   source fsw_env.sh
#   RUN=$(bash scripts/make_run_dir.sh daily_rerun_01)          # data -> $FSW_DATA_DIR
#   RUN=$(bash scripts/make_run_dir.sh D2_sierra_plus_500km --d2) # data_subset -> $FSW_D2_DATA_SUBSET
#   BASE_URL=$RUN python -u fore_swe/forecasting_att_daily_model.py
set -euo pipefail
name="${1:?usage: make_run_dir.sh <name> [--d2]}"
: "${FSW_RUN_ROOT:?source fsw_env.sh first}"
run="$FSW_RUN_ROOT/$name"
if [[ -e "$run/checkpoints" && -n "$(ls -A "$run/checkpoints" 2>/dev/null)" ]]; then
  echo "refusing: $run already holds checkpoints (upstream scripts would resume from them)" >&2
  exit 1
fi
mkdir -p "$run"/{checkpoints,outputs,results/actual,results/predict,logs}
if [[ "${2:-}" == "--d2" ]]; then
  ln -sfn "$FSW_D2_DATA_SUBSET" "$run/data_subset"
else
  ln -sfn "$FSW_DATA_DIR" "$run/data"
fi
echo "$run"
