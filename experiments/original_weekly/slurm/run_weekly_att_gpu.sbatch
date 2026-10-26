#!/bin/bash
#SBATCH --job-name=weekly_att_full512_gpu
#SBATCH --account=m2637_g
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_weekly_reproduction_full512/logs/%x_%j.out
#SBATCH --error=/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_weekly_reproduction_full512/logs/%x_%j.err

set -euo pipefail

cd /global/homes/h/hyvchen/Snow-Predication-at-Sierra

module purge
module load conda

export FSW_ORIG_REPO="/global/homes/h/hyvchen/Snow-Predication-at-Sierra/thirdparties/SWE-Forecasting"
export FSW_WEEKLY_ROOT="/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_weekly_reproduction_full512"
export FSW_RUNTIME_ROOT="/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_weekly_reproduction_full512"
export BASE_URL="$FSW_WEEKLY_ROOT"
export FSW_DATA_DIR="$FSW_WEEKLY_ROOT/data"
export FSW_OUTPUTS="$FSW_RUNTIME_ROOT/outputs"
export FSW_RESULTS_ACTUAL="$FSW_RUNTIME_ROOT/results/actual"
export FSW_RESULTS_PREDICT="$FSW_RUNTIME_ROOT/results/predict"
export FSW_CHECKPOINTS="$FSW_RUNTIME_ROOT/checkpoints"
export FSW_ENV="$SCRATCH/conda_envs/foreswe_repro_att"
export PYTHONPATH="$FSW_WEEKLY_ROOT/scripts:$FSW_ORIG_REPO"
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

echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES before srun=${CUDA_VISIBLE_DEVICES:-unset}"
echo "FSW_WEEKLY_ROOT=$FSW_WEEKLY_ROOT"
echo "FSW_RUNTIME_ROOT=$FSW_RUNTIME_ROOT"

nvidia-smi

conda run --no-capture-output -p "$FSW_ENV" python -u "$FSW_WEEKLY_ROOT/scripts/check_weekly_pipeline.py"

srun \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  --gpus=1 \
  --gpu-bind=none \
  conda run --no-capture-output -p "$FSW_ENV" \
  python -u "$FSW_WEEKLY_ROOT/scripts/forecasting_att_weekly_model.py"

echo "Job finished at: $(date)"
