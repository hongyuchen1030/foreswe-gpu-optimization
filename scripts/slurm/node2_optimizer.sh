#!/usr/bin/env bash
# Node 2 workload (allocate with --perf for ncu): local E0-E5 baseline matrix, fused-AdamW study,
# Nsight Systems per configuration, Nsight Compute Tensor-pipe metrics for GEMM kernels.
# Run INSIDE an allocation:  bash scripts/slurm/run.sh $JOB bash scripts/slurm/node2_optimizer.sh
set -euo pipefail
cd "$FSW_REPO"
STAGES="${STAGES:-matrix nsys ncu adamw}"
NSYS=${NSYS:-/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/profilers/13.2/Nsight_Systems/bin/nsys}
NCU=${NCU:-/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/profilers/13.2/Nsight_Compute/ncu}
A="$FSW_SCRATCH_ROOT/artifacts"; P=engineering_audit/optimization/profiling_summaries
mkdir -p "$A/nsys" "$A/ncu" "$P/nsys" "$P/ncu"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
for s in $STAGES; do
  log "stage $s on $(hostname) job ${SLURM_JOB_ID:-?}"
  case $s in
    matrix) python -m benchmarks.bench_matrix --tag node2_synthetic --repeats 3 --warmup 10 --iters 40 ;;
    adamw)  python -m benchmarks.fused_adamw_study --steps 50 --free-running --with-adam-algorithm-compare ;;
    nsys)
      for e in E0 E1 E2 E3 E4 E5; do
        $NSYS profile -o "$A/nsys/$e" --force-overwrite true --trace=cuda,nvtx,osrt --sample=none \
          python -m benchmarks.bench_one --exp $e --warmup 3 --iters 8 --nvtx --out "$A/nsys/bench_$e.json" > "$A/nsys/$e.log" 2>&1
        $NSYS export --type sqlite --force-overwrite true -o "$A/nsys/$e.sqlite" "$A/nsys/$e.nsys-rep" > /dev/null
        python -m benchmarks.nsys_summary "$A/nsys/$e.sqlite" --exp $e --warmup 3 --out "$P/nsys/nsys_$e.json"
        $NSYS stats --report cuda_gpu_kern_sum --format csv -o "$A/nsys/$e" "$A/nsys/$e.nsys-rep" > /dev/null 2>&1 || true
        head -40 "$A/nsys/${e}_cuda_gpu_kern_sum.csv" > "$P/nsys/kern_sum_top40_$e.csv" 2>/dev/null || true
      done ;;
    ncu)
      M=gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed
      M=$M,sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active
      M=$M,sm__warps_active.avg.pct_of_peak_sustained_active
      for e in E0 E1 E2; do   # precision decides the GEMM kernels; optimizer kernels are covered by nsys
        $NCU --target-processes all -k "regex:gemm|xmma|cutlass" --launch-skip 300 --launch-count 12 --metrics "$M" \
          --csv --page raw python -m benchmarks.bench_one --exp $e --warmup 2 --iters 1 --out "$A/ncu/bench_$e.json" \
          > "$A/ncu/ncu_$e.csv" 2> "$A/ncu/ncu_$e.err" || echo "ncu $e failed (see $A/ncu/ncu_$e.err)"
        python -m benchmarks.ncu_summary "$A/ncu/ncu_$e.csv" --exp $e --out "$P/ncu/ncu_$e.json" || true
      done ;;
    *) echo "unknown stage $s"; exit 1 ;;
  esac
done
log done
