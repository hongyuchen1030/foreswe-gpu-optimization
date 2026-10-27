# ForeSWE GPU Optimization

GPU performance engineering and numerical-accuracy benchmarks for **ForeSWE** (Thapa et al., AAAI 2026),
a 628 M-parameter spatio-temporal attention model for snow-water-equivalent forecasting at 512 SNOTEL
stations. The model and its forecasting targets are unchanged. This repo measures and explains where the
GPU time and the numerical error go, and it consolidates every historical ForeSWE experiment.

Upstream: https://github.com/Krishuthapa/SWE-Forecasting. Its history is preserved here (tag
`upstream-393de07`, branches `upstream-dev`, `upstream-main`), and its README is kept as `README_UPSTREAM.md`.

## Headline results

A100-SXM4-40GB, PyTorch 2.6.0, Plan B 24-week model, 1 issue date × 512 stations per step, median of 120
steps. Full tables, variability and method: `engineering_audit/optimization/PRECISION_PERFORMANCE_REPORT.md`.

| Config | Step ms | Speedup | Fwd / Bwd / Opt ms | Peak VRAM | Output err. vs FP32 | Grad err. vs FP32 |
|---|---:|---:|---|---:|---:|---:|
| E0 FP32 + AdamW foreach (production) | 148.2 | 1.00× | 39.7 / 72.7 / 27.0 | 9.11 GiB | – | – |
| E1 TF32 + foreach | 69.5 | 2.13× | 13.3 / 20.7 / 27.0 | 9.11 GiB | 3.9e-4 | 2.2e-4 |
| E2 BF16 autocast + foreach | 74.7 | 1.98× | 16.9 / 21.0 / 28.3 | 9.13 GiB | 5.7e-3 | 3.6e-3 |
| E3 FP32 + AdamW fused | 129.8 | 1.14× | 39.7 / 72.7 / 9.5 | 8.50 GiB | 0 | 0 |
| **E4 TF32 + fused** | **50.8** | **2.92×** | 13.3 / 20.6 / 9.5 | **8.50 GiB** | 3.9e-4 | 2.2e-4 |
| E5 BF16 + fused | 56.6 | 2.62× | 16.8 / 20.6 / 10.6 | 9.13 GiB | 5.7e-3 | 3.6e-3 |

- Tensor Cores confirmed by Nsight Compute: 82–85 % tensor-pipe activity on TF32 GEMMs, 80 % on BF16;
  0 % on FP32.
- Fused AdamW is 2.75× faster than foreach (28.0 → 10.2 ms, ~75 % of HBM bandwidth) and numerically
  equivalent (update rel-L2 1.8e-6).
- After the switch to Tensor Cores the step is **launch/host-bound** (GPU busy 63 % on E4, 50 % on E5).
- Short-run convergence on real data is unchanged in every configuration.

### torch.compile and CUDA Graphs on top of E4

Details: `engineering_audit/optimization/cuda_graphs/COMPILE_AND_CUDAGRAPH_REPORT.md`.

| Config (TF32 + fused AdamW) | Step ms | vs E4 eager | vs FP32 baseline | GPU busy | Compile / capture | Error vs E4 eager |
|---|---:|---:|---:|---:|---:|---|
| G0 E4 eager (re-measured) | 50.78 | 1.00× | 2.92× | 64 % | – | – |
| G3 manual full-step CUDA graph | 43.94 | 1.16× | 3.37× | 86 % | 0.5 s | bitwise identical |
| G2 `torch.compile(mode="reduce-overhead")` | 41.09 | 1.24× | 3.61× | 83 % | 47 s cold / 17 s warm | out 5e-5, grad 3e-4 |
| **G4 Inductor kernels inside a manual graph** | **40.05** | **1.27×** | **3.70×** | **87 %** | 46 s / 15 s | out 5e-5, grad 3e-4 |

- CUDA graphs cut host launches from 1,106 to 13 per step while executing the same 1,108 kernels;
  Inductor removes about 150 kernels by fusion.
- Fused geo-attention (Triton prototype, `foreswe_bench/geo_attention.py`, opt-in): FP32-accurate
  forward and 9× less memory, but 3.4× slower than eager, and the attention core is only 14–18 % of the step.
  See `engineering_audit/optimization/fused_attention/`.
- Station angularity uses Kahan's robust `2·atan2` formula (`pairwise_vector_angle` in
  `fore_swe/extended_dl_forecasting.py`).

## Layout

```
fore_swe/ raw_gp/ transformer_ar/ other_comparing_models/   upstream ForeSWE source (model logic unchanged)
data/                         upstream SNOTEL CSVs (632 MB, tracked upstream)
experiments/                  every historical ForeSWE experiment (verbatim copy + path-only fixes)
  original_daily/  original_weekly/  planA_stitching/  planB_24week/
  sierra_subset/ (incl. sources recovered from transcripts)  domain_sensitivity/  bias_diagnosis/
engineering_audit/            experiment history, rollout audit, first GPU profiling report
  optimization/               precision + fused-AdamW study: 3 reports, results (JSON/CSV), figures, scripts
foreswe_bench/                library: verbatim model extraction, data, precision modes, metrics, timing
benchmarks/                   CLIs: python -m benchmarks.<name>
tests/                        pytest: CPU unit + repo-hygiene tests, GPU regression tests (-m gpu)
scripts/                      migration, venv, run dirs, checksums; scripts/slurm/ allocation + node workloads
configs/external_paths.env    checkpoints, HF cache, external data (all overridable)
docs/migration/               provenance, file manifests, checksums, checkpoint preservation plan
docs/legacy/                  earlier ForeSWE design notes
```

## Setup on Perlmutter

Never run on a login node; everything executes inside an interactive allocation.

```bash
cd /global/u1/h/hyvchen/foreswe-gpu-optimization
JOB=$(bash scripts/slurm/alloc.sh fsw_node1)                          # 1x A100-40GB, <= 4 h
bash scripts/slurm/run.sh $JOB bash scripts/setup_venv.sh             # once: thin venv over pytorch/2.6.0
bash scripts/slurm/run.sh $JOB python -m benchmarks.build_data_cache  # once: real-data tensor cache
bash scripts/slurm/run.sh $JOB python -m pytest tests                 # CPU + GPU tests
scancel $JOB
```

`fsw_env.sh` (sourced by `run.sh`) loads `pytorch/2.6.0`, activates the venv (sentence-transformers,
gpytorch, hydroeval) and sets every path. New outputs go to `/pscratch/sd/h/hyvchen/foreswe-gpu-optimization`,
never into the repo or a historical folder.

## Reproducing the study

```bash
bash engineering_audit/optimization/scripts/reproduce_all.sh   # two allocations, all phases, figures
```

or piece by piece (inside an allocation):

| Command | Measures |
|---|---|
| `python -m benchmarks.bench_one --exp E4 --infer --out r.json` | one config: event-timed phases, memory, inference |
| `python -m benchmarks.bench_matrix --tag mytag --repeats 3` | E0–E5, fresh process per run, rotated order |
| `python -m benchmarks.numerics_precision` | FP32/TF32/BF16 output, loss and gradient error vs FP32 and FP64 |
| `python -m benchmarks.error_origin` | error per module and per op, split-GEMM emulation, optimizer absorption |
| `python -m benchmarks.fused_adamw_study --free-running` | foreach vs fused AdamW: correctness, drift, timing |
| `python -m benchmarks.convergence --exp E4 --steps 300` | matched-budget training on real data |
| `python -m benchmarks.profile_config --exp E4` | kernel categories, Tensor-Core share, launches |
| `python -m benchmarks.summarize_results && python -m benchmarks.make_figures` | final table + figures |
| `python -m benchmarks.graph_matrix --tag t --cfgs G0 G1 G2 G3 G4` | torch.compile / CUDA-graph matrix (both timing scopes) |
| `python -m benchmarks.graph_correctness` | graph/compile vs eager: outputs, grads, updates, graph-safety checks |
| `python -m benchmarks.attention_study` | geo-attention core: eager vs Inductor vs fused Triton, accuracy + latency |

## Rerunning the original ForeSWE pipelines

Historical launchers under `experiments/*/slurm` are kept for provenance; they reference the purged conda
env `foreswe_repro_att`. To rerun an upstream script, give it an isolated run directory:

```bash
source fsw_env.sh
RUN=$(bash scripts/make_run_dir.sh daily_rerun_01)
BASE_URL=$RUN python -u fore_swe/forecasting_att_daily_model.py
FSW_PLANB_ROOT=$RUN python -u experiments/planB_24week/planB_24week_att_model.py
```

## Data, checkpoints, large artifacts

None are in git. Locations, sizes, SHA-256 sums and a **pscratch preservation plan** (the six historical
checkpoints are purge-eligible from about 2026-12-21) are in `docs/migration/MIGRATION_PROVENANCE.md`.
