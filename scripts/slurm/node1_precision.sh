#!/usr/bin/env bash
# Node 1 workload: data cache, tests, precision numerics + error origin, E0-E5 timing matrix,
# convergence (300 steps, then one epoch), torch.profiler breakdown per configuration.
# Run INSIDE an allocation:  bash scripts/slurm/run.sh $JOB bash scripts/slurm/node1_precision.sh
set -euo pipefail
cd "$FSW_REPO"
STAGES="${STAGES:-cache tests numerics origin matrix converge300 profile converge_epoch}"
R=engineering_audit/optimization/benchmark_results
mkdir -p "$R" "$FSW_SCRATCH_ROOT/logs"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
for s in $STAGES; do
  log "stage $s on $(hostname) job ${SLURM_JOB_ID:-?}"
  case $s in
    cache)   [[ -f "$FSW_CACHE_DIR/planB24_real_data_cache.pt" ]] || python -m benchmarks.build_data_cache ;;
    tests)   python -m pytest -q tests | tee "$R/pytest_node1.txt" ;;
    numerics) python -m benchmarks.numerics_precision --weights init planB ;;
    origin)  python -m benchmarks.error_origin ;;
    matrix)  python -m benchmarks.bench_matrix --tag node1_synthetic --repeats 3 --warmup 10 --iters 40 ;;
    matrix_real) python -m benchmarks.bench_matrix --tag node1_real --repeats 1 --data real ;;
    converge300) for e in E0 E1 E2 E3 E4 E5; do python -m benchmarks.convergence --exp $e --steps 300; done ;;
    converge_epoch) for e in E0 E1 E2 E3 E4 E5; do python -m benchmarks.convergence --exp $e --steps 0; done ;;
    profile) for e in E0 E1 E2 E3 E4 E5; do python -m benchmarks.profile_config --exp $e; done ;;
    *) echo "unknown stage $s"; exit 1 ;;
  esac
done
log done
