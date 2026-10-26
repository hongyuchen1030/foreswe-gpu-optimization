
#!/bin/bash
#SBATCH --job-name=sierra_gp_daily_gpu80
#SBATCH --account=m2637_g
#SBATCH --qos=regular
#SBATCH --constraint=gpu&hbm80g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_sierra_station/logs/%x_%j.out
#SBATCH --error=/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_sierra_station/logs/%x_%j.err

set -euo pipefail

module purge
module load conda

cd /global/homes/h/hyvchen/Snow-Predication-at-Sierra

export FSW_ORIG_REPO="/global/homes/h/hyvchen/Snow-Predication-at-Sierra/thirdparties/SWE-Forecasting"
export FSW_HOME_SIERRA_ROOT="/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_sierra_station"
export FSW_SIERRA_ROOT="/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_sierra_station"
export FSW_DATA_DIR="$FSW_HOME_SIERRA_ROOT/data_subset"
export FSW_OUTPUTS="$FSW_SIERRA_ROOT/outputs"
export FSW_RESULTS_ACTUAL="$FSW_SIERRA_ROOT/results/actual"
export FSW_RESULTS_PREDICT="$FSW_SIERRA_ROOT/results/predict"
export FSW_CHECKPOINTS="$FSW_SIERRA_ROOT/checkpoints"
export FSW_ENV="$SCRATCH/conda_envs/foreswe_repro_att"
export FSW_EXPECTED_STATIONS=43
export PYTHONPATH="$FSW_HOME_SIERRA_ROOT/scripts:$FSW_ORIG_REPO"
export PYTHONUNBUFFERED=1
export TMPDIR="$SCRATCH/tmp"

mkdir -p \
  "$FSW_OUTPUTS" \
  "$FSW_RESULTS_ACTUAL" \
  "$FSW_RESULTS_PREDICT" \
  "$FSW_CHECKPOINTS" \
  "$FSW_SIERRA_ROOT/logs" \
  "$TMPDIR"

echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES before srun=${CUDA_VISIBLE_DEVICES:-unset}"
echo "FSW_ORIG_REPO=$FSW_ORIG_REPO"
echo "FSW_HOME_SIERRA_ROOT=$F<truncated omitted_approx_tokens="25" />FSW_RESULTS_ACTUAL=$FSW_RESULTS_ACTUAL"
echo "FSW_RESULTS_PREDICT=$FSW_RESULTS_PREDICT"
echo "FSW_CHECKPOINTS=$FSW_CHECKPOINTS"
echo "FSW_ENV=$FSW_ENV"

echo "Checking required Sierra GP input files:"
ls -lh "$FSW_DATA_DIR/snotel_location_info_2.csv"
ls -lh "$FSW_OUTPUTS/forecasting_modified_daily_att_inputs_for_gp_act.pt"
ls -lh "$FSW_OUTPUTS/forecasting_modified_daily_att_outputs_for_gp_act.pt"
ls -lh "$FSW_OUTPUTS/forecasting_modified_daily_att_predicted_outputs_for_gp_act.pt"
ls -lh "$FSW_OUTPUTS"/daily_test*_inputs_for_gp_act.pt
ls -lh "$FSW_RESULTS_ACTUAL"/*daily_window_actual.pkl

echo "Checking GPU on allocated node:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

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

echo "Starting Sierra ForeSWE daily GP head on 80GB GPU:"
srun \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  --gpus=1 \
  --gpu-bind=none \
  conda run --no-capture-output -p "$FSW_ENV" \
  python -u "$FSW_HOME_SIERRA_ROOT/scripts/forecasting_daily_gp_model.py"

echo "Job finished at: $(date)"
