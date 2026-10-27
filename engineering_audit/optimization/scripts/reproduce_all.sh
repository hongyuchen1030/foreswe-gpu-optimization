#!/usr/bin/env bash
# Reproduce every result in engineering_audit/optimization/ from a Perlmutter login node.
# The implementation lives in foreswe_bench/ + benchmarks/ (library + CLIs) and scripts/slurm/ (node
# workloads); this wrapper only orders them. Uses two interactive A100-40GB allocations (<= 4 h each).
#
#   bash engineering_audit/optimization/scripts/reproduce_all.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
J1=$(bash scripts/slurm/alloc.sh fsw_node1)
J2=$(bash scripts/slurm/alloc.sh fsw_node2 --perf)
trap 'scancel "$J1" "$J2"' EXIT
echo "node1=$J1 node2=$J2"

bash scripts/slurm/run.sh "$J1" bash scripts/setup_venv.sh
bash scripts/slurm/run.sh "$J1" env STAGES=cache bash scripts/slurm/node1_precision.sh
# node 2: own E0-E5 baseline, nsys, ncu, fused AdamW (needs the cache built above)
bash scripts/slurm/run.sh "$J2" bash scripts/slurm/node2_optimizer.sh > node2.log 2>&1 &
# node 1: tests, numerics, error origin, matrix, convergence (300 steps + one epoch), profiler
bash scripts/slurm/run.sh "$J1" env STAGES="tests numerics origin matrix converge300 profile converge_epoch" \
  bash scripts/slurm/node1_precision.sh
wait
bash scripts/slurm/run.sh "$J1" bash -c "python -m benchmarks.summarize_results && python -m benchmarks.make_figures"
