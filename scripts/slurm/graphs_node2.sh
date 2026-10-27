#!/usr/bin/env bash
# Node 2 (graph/compile study): correctness, local E4 baseline, Nsight Systems timelines, convergence.
# Run INSIDE an allocation:  bash scripts/slurm/run.sh $JOB bash scripts/slurm/graphs_node2.sh
set -euo pipefail
cd "$FSW_REPO"
STAGES="${STAGES:-correct baseline nsys converge300}"
NSYS=${NSYS:-/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/profilers/13.2/Nsight_Systems/bin/nsys}
A="$FSW_SCRATCH_ROOT/artifacts/cuda_graphs"; P=engineering_audit/optimization/cuda_graphs/profiler_summaries
mkdir -p "$A/nsys" "$P/nsys"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
for s in $STAGES; do
  log "stage $s on $(hostname) job ${SLURM_JOB_ID:-?}"
  case $s in
    correct)  python -m benchmarks.graph_correctness --cfgs ${CORRECT_CFGS:-G0c G1 G2 G3f G3 G4} ;;
    baseline) python -m benchmarks.graph_matrix --tag nodeB --cfgs ${BASE_CFGS:-G0 G3} --repeats 3 ;;
    nsys)
      for c in ${NSYS_CFGS:-G0 G1 G2 G3 G4}; do
        TORCHINDUCTOR_CACHE_DIR="$FSW_SCRATCH_ROOT/inductor_cache/nsys/$c" \
        $NSYS profile -o "$A/nsys/$c" --force-overwrite true --trace=cuda,nvtx,osrt --sample=none \
          --cuda-graph-trace=node python -m benchmarks.graph_bench --cfg $c --warmup 3 --iters 8 --nvtx \
          --out "$A/nsys/bench_$c.json" > "$A/nsys/$c.log" 2>&1 || { log "nsys $c failed"; continue; }
        $NSYS export --type sqlite --force-overwrite true -o "$A/nsys/$c.sqlite" "$A/nsys/$c.nsys-rep" > /dev/null
        python -m benchmarks.nsys_summary "$A/nsys/$c.sqlite" --exp $c --warmup 3 --out "$P/nsys/nsys_$c.json"
      done ;;
    converge300) for c in ${CONV_CFGS:-G0 G1 G2 G3 G4}; do python -m benchmarks.graph_convergence --cfg $c --steps 300 || log "conv $c failed"; done ;;
    epoch) for c in ${EPOCH_CFGS:-G0}; do python -m benchmarks.graph_convergence --cfg $c --steps 0 || log "epoch $c failed"; done ;;
    *) echo "unknown stage $s"; exit 1 ;;
  esac
done
log done
