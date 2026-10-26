#!/usr/bin/env bash
# Run the bounded ForeSWE GPU profiling suite on ONE GPU of an existing allocation.
# Usage (from a login node):
#   salloc -N 1 -C gpu -q interactive -A m2637_g -t 01:00:00 --gpus=1 -c 32
#   srun --jobid=<id> --overlap -n 1 -c 32 --gpus=1 bash -l run_profiling.sh
# Heavy outputs (traces, .nsys-rep) go to pscratch; small JSON/TXT summaries are
# copied into engineering_audit/profiling/results/.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The historical env $SCRATCH/conda_envs/foreswe_repro_att (torch 2.4.1+cu121) was
# partly deleted by the pscratch purge (its stdlib is gone), so use the NERSC module.
module load "${TORCH_MODULE:-pytorch/2.6.0}" > /dev/null 2>&1
PY=${PY:-python}
OUT=${OUT:-/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/foreswe_gpu_profiling_v1}
NSYS=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/profilers/13.2/Nsight_Systems/bin/nsys
NCU=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/profilers/13.2/Nsight_Compute/ncu
STAGES=${STAGES:-"bench blocking variants nsys ncu"}
mkdir -p "$OUT/logs" "$HERE/results"
export TMPDIR=/pscratch/sd/h/hyvchen/tmp PYTHONUNBUFFERED=1

case "$(hostname)" in nid*) ;; *) echo "refusing to run on $(hostname): not a compute node"; exit 1;; esac
nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm --format=csv | tee "$OUT/logs/gpu.txt"
P="$PY $HERE/profile_foreswe.py --out-dir $OUT"

for stage in $STAGES; do
  echo "=== stage $stage $(date -u +%FT%TZ)"
  case $stage in
    bench)     # adapted Plan B path, launch-blocking off (as Plan B runs), real checkpoint, + torch.profiler
      env -u CUDA_LAUNCH_BLOCKING $P --variant planB24 --mode both --load-checkpoint --torch-profiler \
        --warmup 5 --iters 20 2>&1 | tee "$OUT/logs/planB24_both.log" ;;
    blocking)  # same path with CUDA_LAUNCH_BLOCKING=1, as hard-coded in the original daily/weekly scripts
      CUDA_LAUNCH_BLOCKING=1 $P --variant planB24 --mode both --warmup 3 --iters 10 --tag _clb1 \
        2>&1 | tee "$OUT/logs/planB24_both_clb1.log" ;;
    variants)  # original weekly / daily heads, both launch regimes
      for v in weekly4 daily10; do
        CUDA_LAUNCH_BLOCKING=1 $P --variant $v --mode both --warmup 3 --iters 10 --tag _clb1 2>&1 | tee "$OUT/logs/${v}_both_clb1.log"
        env -u CUDA_LAUNCH_BLOCKING $P --variant $v --mode both --warmup 3 --iters 10 2>&1 | tee "$OUT/logs/${v}_both.log"
      done ;;
    nsys)
      env -u CUDA_LAUNCH_BLOCKING $NSYS profile -o "$OUT/nsys_planB24" --force-overwrite true \
        --trace=cuda,nvtx,osrt --sample=none --cuda-memory-usage=true \
        $P --variant planB24 --mode both --warmup 2 --iters 5 --nvtx --tag _nsys > "$OUT/logs/nsys_profile.log" 2>&1
      for r in cuda_gpu_kern_sum cuda_api_sum cuda_gpu_mem_time_sum cuda_gpu_mem_size_sum nvtx_sum; do
        $NSYS stats --report $r --format csv -o "$OUT/nsys_planB24" "$OUT/nsys_planB24.nsys-rep" > /dev/null 2>&1
      done
      ls "$OUT"/nsys_planB24* ;;
    ncu)       # a handful of forward kernels only (SpeedOfLight + occupancy)
      env -u CUDA_LAUNCH_BLOCKING $NCU --target-processes all --launch-skip 200 --launch-count 40 \
        --section SpeedOfLight --section Occupancy --csv --page details \
        $P --variant planB24 --mode infer --warmup 1 --iters 1 --tag _ncu > "$OUT/ncu_planB24_infer.csv" 2> "$OUT/logs/ncu.err"
      echo "ncu exit $?" ;;
  esac
done

cp "$OUT"/bench_*.json "$OUT"/torch_ops_*.txt "$HERE/results/" 2>/dev/null
cp "$OUT"/nsys_planB24_*.csv "$HERE/results/" 2>/dev/null
cp "$OUT/logs/gpu.txt" "$HERE/results/" 2>/dev/null
echo "done $(date -u +%FT%TZ)"
