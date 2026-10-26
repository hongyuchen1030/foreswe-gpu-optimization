#!/usr/bin/env bash
# Run a command inside an existing allocation, from the repo root, with fsw_env.sh sourced.
#   bash scripts/slurm/run.sh $JOB python -m benchmarks.bench_matrix --tag node1
set -euo pipefail
job="${1:?usage: run.sh <jobid> <command...>}"; shift
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
printf -v cmd '%q ' "$@"
srun --jobid="$job" --overlap -n 1 -c 32 --gpus=1 bash -lc "cd '$REPO' && source fsw_env.sh && \
  case \$(hostname) in nid*) ;; *) echo 'not a compute node' >&2; exit 1;; esac && $cmd"
