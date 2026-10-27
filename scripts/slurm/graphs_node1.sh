#!/usr/bin/env bash
# Node 1 (graph/compile study): full G-matrix timing (3 repeats, fresh process each, cold+warm compile).
# Run INSIDE an allocation:  bash scripts/slurm/run.sh $JOB bash scripts/slurm/graphs_node1.sh
set -euo pipefail
cd "$FSW_REPO"
CFGS="${CFGS:-G0 G0c G1 G1L G2 G2L G3f G3 G4}"
echo "[$(date -u +%FT%TZ)] graph matrix on $(hostname) job ${SLURM_JOB_ID:-?}: $CFGS"
nvidia-smi --query-gpu=name,uuid,clocks.max.sm --format=csv,noheader
python -m benchmarks.graph_matrix --tag "${TAG:-nodeA}" --cfgs $CFGS --repeats "${REPEATS:-3}"
echo "[$(date -u +%FT%TZ)] done"
