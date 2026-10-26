#!/bin/bash
set -euo pipefail

cd /global/homes/h/hyvchen/Snow-Predication-at-Sierra

export FSW_ORIG_REPO="/global/homes/h/hyvchen/Snow-Predication-at-Sierra/thirdparties/SWE-Forecasting"
export FSW_DOMAIN_ROOT="/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km"
export FSW_RUNTIME_ROOT="/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km"
export FSW_DATA_DIR="$FSW_DOMAIN_ROOT/data_subset"
export FSW_STATION_SUBSET_CSV="$FSW_DOMAIN_ROOT/station_subset.csv"
export FSW_OUTPUTS="$FSW_RUNTIME_ROOT/outputs"
export FSW_RESULTS_ACTUAL="$FSW_RUNTIME_ROOT/results/actual"
export FSW_RESULTS_PREDICT="$FSW_RUNTIME_ROOT/results/predict"
export FSW_CHECKPOINTS="$FSW_RUNTIME_ROOT/checkpoints"
export FSW_ENV="$SCRATCH/conda_envs/foreswe_repro_att"
export PYTHONPATH="$FSW_DOMAIN_ROOT/scripts:$FSW_ORIG_REPO"
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore:To copy construct from a tensor"
export HF_HOME="/pscratch/sd/h/hyvchen/foreswe_model_cache/huggingface"
export TRANSFORMERS_CACHE="/pscratch/sd/h/hyvchen/foreswe_model_cache/huggingface/hub"
export SENTENCE_TRANSFORMERS_HOME="/pscratch/sd/h/hyvchen/foreswe_model_cache/sentence_transformers"
export TMPDIR="$SCRATCH/tmp"

mkdir -p \
  "$FSW_OUTPUTS" \
  "$FSW_RESULTS_ACTUAL" \
  "$FSW_RESULTS_PREDICT" \
  "$FSW_CHECKPOINTS" \
  "$FSW_RUNTIME_ROOT/logs" \
  "$HF_HOME" \
  "$TRANSFORMERS_CACHE" \
  "$SENTENCE_TRANSFORMERS_HOME" \
  "$TMPDIR"

echo "Interactive D2 attention debug started at: $(date)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "FSW_DOMAIN_ROOT=$FSW_DOMAIN_ROOT"
echo "FSW_RUNTIME_ROOT=$FSW_RUNTIME_ROOT"
echo "FSW_DATA_DIR=$FSW_DATA_DIR"
echo "FSW_STATION_SUBSET_CSV=$FSW_STATION_SUBSET_CSV"

echo "Checking GPU on allocated node:"
nvidia-smi

echo "Checking PyTorch CUDA visibility:"
conda run --no-capture-output -p "$FSW_ENV" \
  python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('device count', torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"

echo "Launching D2 attention model:"
conda run --no-capture-output -p "$FSW_ENV" \
  python -u "$FSW_DOMAIN_ROOT/scripts/forecasting_att_daily_model.py" \
  2>&1 | tee "$FSW_RUNTIME_ROOT/logs/d2_att_interactive_debug.log"

echo "Interactive D2 attention debug finished at: $(date)"
