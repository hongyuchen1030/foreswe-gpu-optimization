#!/bin/bash
#SBATCH --job-name=foreswe_att_daily
#SBATCH --account=m2637
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=/global/homes/h/hyvchen/foreswe-gpu-optimization/experiments/original_daily/logs/%x_%j.out
#SBATCH --error=/global/homes/h/hyvchen/foreswe-gpu-optimization/experiments/original_daily/logs/%x_%j.err

set -euo pipefail

module purge
module load conda

cd /global/homes/h/hyvchen/foreswe-gpu-optimization

export FSW_REPO="$PWD"
export FSW_ENV="$SCRATCH/conda_envs/foreswe_repro_att"
export BASE_URL="${FSW_RUN_ROOT:-/pscratch/sd/h/hyvchen/foreswe-gpu-optimization/runs}/original_daily"  # create via scripts/make_run_dir.sh original_daily
export PYTHONPATH="$FSW_REPO"
export PYTHONUNBUFFERED=1
export TMPDIR="$SCRATCH/tmp"

mkdir -p "$TMPDIR"
mkdir -p "$FSW_REPO/checkpoints"
mkdir -p "$FSW_REPO/outputs"
mkdir -p /global/homes/h/hyvchen/foreswe-gpu-optimization/experiments/original_daily/logs

echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES before srun=${CUDA_VISIBLE_DEVICES:-unset}"
echo "FSW_REPO=$FSW_REPO"
echo "FSW_ENV=$FSW_ENV"

echo "Checking GPU on allocated node:"
nvidia-smi

echo "Checking PyTorch CUDA visibility before srun:"
conda run --no-capture-output -p "$FSW_ENV" \
  python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('device count', torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"

echo "Checking PyTorch CUDA visibility inside srun:"
srun \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  --gpus=1 \
  --gpu-bind=none \
  conda run --no-capture-output -p "$FSW_ENV" \
  python -c "import os, torch; print('CUDA_VISIBLE_DEVICES inside srun=', os.environ.get('CUDA_VISIBLE_DEVICES')); print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('device count', torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"

echo "Starting from-scratch ForeSWE attention training:"
srun \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  --gpus=1 \
  --gpu-bind=none \
  conda run --no-capture-output -p "$FSW_ENV" \
  python -u fore_swe/forecasting_att_daily_model.py

echo "Job finished at: $(date)"