#!/bin/bash
#SBATCH --job-name=foreswe_gp_daily_cpu
#SBATCH --account=m2637
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_reproduction/logs/%x_%j.out
#SBATCH --error=/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_reproduction/logs/%x_%j.err

set -euo pipefail

module purge
module load conda

export HOME_REPO="/global/homes/h/hyvchen/Snow-Predication-at-Sierra"
export FSW_REPO="$HOME_REPO/thirdparties/SWE-Forecasting"
export FSW_ENV="$SCRATCH/conda_envs/foreswe_repro_att"

export BASE_URL="$FSW_REPO"
export PYTHONPATH="$FSW_REPO"
export PYTHONUNBUFFERED=1
export TMPDIR="$SCRATCH/tmp"
export CUDA_VISIBLE_DEVICES=""

cd "$FSW_REPO"

echo "Checking required GP input files:"
ls -lh outputs/forecasting_modified_daily_att_inputs_for_gp_act.pt
ls -lh outputs/forecasting_modified_daily_att_outputs_for_gp_act.pt
ls -lh outputs/forecasting_modified_daily_att_predicted_outputs_for_gp_act.pt
ls -lh outputs/daily_test*_inputs_for_gp_act.pt
ls -lh results/actual/*daily_window_actual.pkl

echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "FSW_REPO=$FSW_REPO"
echo "FSW_ENV=$FSW_ENV"
echo "BASE_URL=$BASE_URL"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "results/predict resolves to: $(readlink -f results/predict)"



srun -N 1 -n 1 -c "$SLURM_CPUS_PER_TASK" \
  conda run --no-capture-output -p "$FSW_ENV" \
  python -u fore_swe/forecasting_daily_gp_model.py



echo "Job finished at: $(date)"