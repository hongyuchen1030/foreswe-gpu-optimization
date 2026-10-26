#!/usr/bin/env bash
# Request an interactive GPU allocation (no shell) and print its job id.
#   JOB=$(bash scripts/slurm/alloc.sh fsw_node1)            # 4 h, 1 x A100-40GB
#   JOB=$(bash scripts/slurm/alloc.sh fsw_node2 --perf)     # + hardware counters for Nsight Compute
# Policy: at most two concurrent allocations, <= 4 h each; release with `scancel $JOB` when done.
set -euo pipefail
name="${1:?usage: alloc.sh <job-name> [--perf] [hours]}"
perf=""; hours="04:00:00"
for arg in "${@:2}"; do
  case "$arg" in
    --perf) perf="--perf=generic" ;;   # pauses DCGM so ncu can read counters
    *:*) hours="$arg" ;;
  esac
done
n_active=$(squeue -u "$USER" -h -t R,PD | wc -l)
if (( n_active >= 2 )); then echo "refusing: already $n_active allocations" >&2; exit 1; fi
salloc -N 1 -C "gpu&hbm40g" -q interactive -A "${FSW_ACCOUNT:-m2637_g}" -t "$hours" --gpus=1 -c 32 \
  --no-shell --job-name="$name" $perf >&2
squeue -u "$USER" -h -n "$name" -o %i | head -1
