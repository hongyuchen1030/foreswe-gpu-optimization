# ForeSWE GPU profiling suite

Bounded, reproducible profiling of the ForeSWE attention model. Production files are only read,
never imported or edited. Model classes are extracted with `ast` (see `profile_foreswe.py`).

```bash
# login node: get one GPU (never run these scripts on a login node)
salloc -N 1 -C gpu -q interactive -A m2637_g -t 01:00:00 --gpus=1 -c 32 --no-shell
# full suite: bench, CUDA_LAUNCH_BLOCKING=1, weekly/daily variants, nsys, ncu
srun --jobid=<JOBID> --overlap -n 1 -c 32 --gpus=1 bash -l run_profiling.sh
# a subset
STAGES="bench nsys" srun --jobid=<JOBID> --overlap -n 1 -c 32 --gpus=1 bash -l run_profiling.sh
```

Nsight Compute needs hardware counters, which DCGM holds by default. Allocate with
`--perf=generic` for the `ncu` stage.

| Script | Purpose |
|---|---|
| `profile_foreswe.py` | Event-timed train/infer steps, memory peaks, optional `torch.profiler` and NVTX |
| `run_profiling.sh` | Stage runner (`bench blocking variants nsys ncu`); copies small summaries to `results/` |
| `bench_attention_core.py` | Isolated timing of the geo-biased attention core / MHA / encoder layer |
| `nsys_gpu_idle.py` | GPU busy/idle and kernel count per step from the nsys SQLite export |
| `kernel_categories.py` | Buckets kernel time from a torch.profiler Chrome trace |
| `ncu_summarize.py` | One row per kernel launch from the `ncu --csv --page details` output |

Heavy outputs (Chrome traces, `.nsys-rep`, `.sqlite`, raw ncu CSV) go to
`/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/foreswe_gpu_profiling_v1/`. Keep them out of git.
Environment: NERSC `pytorch/2.6.0`. The historical `foreswe_repro_att` env was purged.
