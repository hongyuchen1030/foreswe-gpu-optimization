# ForeSWE GPU Profiling Report (2026-10-26)

All numbers here are **newly measured [M]** unless marked **[V]** (historical, verified) or
**[D]** (derived). No historical forward/backward/optimizer/memory/utilisation measurements
existed (see `FORE_SWE_EXPERIMENT_HISTORY.md`). Optimisation ideas are **hypotheses**: none are
implemented.

## 1. Environment

| | |
|---|---|
| GPU | 1 × NVIDIA A100-SXM4-40GB, 108 SMs, max SM clock 1410 MHz (nid001185, job 60333179). ncu run on A100-SXM4-80GB (nid200505, job 60333598, `--perf=generic`). Attention microbench on A100-40GB |
| Driver / CUDA | 580.178.04 / CUDA runtime 12.4, cuDNN 9.5 |
| Software | NERSC module `pytorch/2.6.0` (Python 3.12.9, torch 2.6.0) |
| Why not the historical env | `$SCRATCH/conda_envs/foreswe_repro_att` (torch 2.4.1+cu121, Py 3.8) was partially deleted by the pscratch purge and no longer starts. Not modified |
| Precision | FP32, no autocast. `torch.backends.cuda.matmul.allow_tf32 = False` (default), so every GEMM runs as `ampere_sgemm_*` on CUDA cores and **tensor cores are unused** |
| Profilers | `torch.profiler` (CPU+CUDA), `torch.cuda.Event`, `torch.cuda.max_memory_{allocated,reserved}`, Nsight Systems 2026 (HPC SDK 26.5), Nsight Compute (SpeedOfLight, Occupancy, LaunchStats) |

## 2. Methodology

- **Harness** `profiling/profile_foreswe.py`. It takes `CrossVarTransformer`, `SWETransformer`
  and the `getInputs`/`getOutputs` helpers **verbatim from the production source files via
  `ast`**, so no production code is edited or imported. Variants: `planB24` (the long-horizon
  adaptation), `weekly4`, `daily10`.
- **Data**: synthetic tensors with the exact production shapes and residency. `model_inputs`
  (10305, 512, 19) fp32 and `model_outputs` live on the GPU; prompts are (1, 512, 1536);
  distance/angularity are float64 numpy (512, 512), converted and copied H2D every step as in
  production. The model has no data-dependent control flow, so values don't affect timing.
  Inference also loads the **real Plan B / daily checkpoint** (read-only).
- **Batch** = 1 issue date × 512 stations (production). Inputs: daily (30, 512, 19), weekly
  (30, 512, 19), yearly (5, 512, 19); daily model: (1, 512, 19), (120, 512, 19), (5, 512, 19).
- **Per-step regions** (each bracketed by `torch.cuda.Event`, with `cuda.synchronize()` before
  and after the step and `perf_counter` for wall time):
  - training: prep → forward + loss → `loss.item()` → `zero_grad` → `backward` → `AdamW.step`
    (same order and hyperparameters as production);
  - inference: prep → forward (`eval`, `no_grad`) → D2H `.cpu()`.
  Also a back-to-back forward loop with no host work, as the GPU-bound floor.
- **Counts**: 5 warm-up + 20 measured steps (planB24 main run); 3 + 10 for the variant and
  launch-blocking runs; 3 profiled steps for `torch.profiler`; 2 + 5 for nsys. Medians reported.
  Model init and checkpoint load are reported separately and excluded from step times.
- **Profiler overhead**: the step times in §3 come from plain runs, not profiled ones. The nsys
  run reproduced them within 1.5% (150.4 vs 148.4 ms train step).
- **Runner**: `profiling/run_profiling.sh`, launched from the login node with
  `srun --jobid=<alloc> --overlap … bash -l run_profiling.sh`. It refuses to run on a non-`nid`
  host.

## 3. Results

### 3.1 Training step (Plan B, 24 heads; launch-blocking off, as Plan B ran)

| Region | Median ms | % of step |
|---|---:|---:|
| Data prep (`get_example`: 3 full-dataset gathers, numpy 512×512 fancy-index + H2D, `torch.tensor` copies) | 8.0–8.5 | 5.6 |
| Forward + MSE | 39.6 | 26.7 |
| `loss.item()` (host sync) | 0.03 | 0.0 |
| `zero_grad` | 0.45 | 0.3 |
| **Backward** | **72.5** | 48.9 |
| **AdamW step** (foreach, 426.0 M params with state) | **27.1** | 18.2 |
| **End-to-end step (wall)** | **148.4** | 100 |
| Same step with `CUDA_LAUNCH_BLOCKING=1` (as hard-coded in the original scripts) | 182.2 (fwd 52.1 / bwd 90.4 / opt 31.0) | +23% |

Variants are indistinguishable (median wall ms, off / `CLB=1`): weekly4 147.3 / 182.8,
daily10 148.3 / 183.5, planB24 148.4 / 182.2.

### 3.2 Inference (Plan B forward, real checkpoint)

| Metric | Value |
|---|---|
| Forward per issue date (event-timed, inside step) | 39.2 ms |
| Forward back-to-back (no host work, GPU floor) | 38.3 ms (`CLB=1`: 48.4 ms) |
| Prep per issue date | 6.9–8.1 ms |
| D2H (representation + prediction + target, 112 KiB) | 0.085 ms |
| Step wall | 46.2–47.4 ms |
| **Long-horizon "rollout" for one year** (Plan B = 1 forward from Feb 1 → 24 weeks) | **≈ 0.047 s/yr**; 5 test years ≈ 0.24 s |
| Original weekly eval for one year (181 daily-issue forwards + 7 stitched) | ≈ 8.6 s/yr [D] (188 × 46 ms) |
| Original daily eval for one year (181 forwards) | ≈ 8.4 s/yr [D]; ≈ 10.6 s with `CLB=1` |
| Representation export over the training set (Plan B, 3297 forwards) | ≈ 2.6 min [D]. Historical: ≤ 4 min [V] |
| Checkpoint read (5.9 GB, pscratch) + H2D | 5.7 s + 0.27 s (one-off, excluded) |
| Model construction | 2.8 s (one-off, excluded) |

### 3.3 Memory

| | Allocated | Reserved |
|---|---:|---:|
| Parameters (627.6 M fp32; daily: 426.8 M) | 2.34 GiB (daily 1.59) | |
| ↳ of which **dead** `CrossVarTransformer.encoder_layers` (built, never called; weekly/Plan B only) | **0.75 GiB (201.6 M params, 32%)** | |
| Inference static (params + 400 MB dataset + prompts) | 2.74 GiB | |
| **Inference peak** | **3.13 GiB** | **3.50 GiB** |
| Training after first step (params + grads + AdamW m, v) | 7.52 GiB | |
| **Training peak** (Plan B / weekly4) | **9.11 GiB** | **9.54 GiB** |
| Training peak (daily10) | 8.37 GiB | 8.80 GiB |
| Attention core transient per call (16 MiB score tensors, several live) | 56 MiB extra | |

On a 40 GB A100 this is ~23% of capacity: the attention model is **latency-bound, not
memory-bound**. The historical OOMs were all in the GPyTorch GP head.

### 3.4 Transfers, launches, idle time (Nsight Systems, Plan B)

| Metric | Train step | Inference step |
|---|---:|---:|
| Kernels launched | 1246 | 376 |
| GPU busy (union of kernels + copies) | 139.3 ms | 40.7 ms |
| **GPU idle inside the step** | **11.4 ms (7.6%)** | **7.1 ms (14.9%)** |
| CPU-side launch time (NVTX) for forward / backward | 17.7 / 22.7 ms (vs 39.6 / 72.5 ms GPU) | 14.5 ms (vs 39 ms GPU) |
| H2D per step | 2 MiB (distance + angularity, recomputed from numpy every step) | 2 MiB |
| D2H per step | `loss.item()` scalar | 112 KiB |
| D2D per step | `torch.tensor(t)` copies of the gathered windows | same |

The CPU stays ahead of the GPU during forward/backward, so **the steady state is GPU-bound**.
Kernel-launch overhead is hidden except during the host-side prep phase. The idle time is the
host data prep plus the synchronisations at `loss.item()` and `.cpu()`.
`cudaLaunchKernel`: median 5.9 µs, mean 15.7 µs.

### 3.5 GPU time by kernel category (torch.profiler, per step)

| Category | Train ms | Train % | Train launches | Infer ms | Infer % | Infer launches |
|---|---:|---:|---:|---:|---:|---:|
| FP32 GEMM (`ampere_sgemm_*`, gemv; includes the attention bmm) | 98.9 | **71.0** | 348 | 35.1 | **86.1** | 131 |
| AdamW `multi_tensor_apply` (foreach) | 27.0 | **19.3** | 175 | – | – | – |
| Other elementwise (adds, muls, GELU, ReLU, attention bias math) | 4.4 | 3.2 | 284 | 1.8 | 4.5 | 139 |
| Reductions | 2.6 | 1.9 | 196 | 0.02 | 0.1 | 4 |
| **Index gather** `model_inputs[:, loc_indices, :]` | 2.55 | 1.8 | 8 | 2.55 | **6.3** | 8 |
| copy / contiguous / `.to()` | 1.7 | 1.2 | 119 | 0.55 | 1.4 | 55 |
| softmax fwd/bwd | 0.9 | 0.6 | 38 | 0.29 | 0.7 | 19 |
| LayerNorm fwd/bwd | 0.44 | 0.3 | 32 | 0.19 | 0.5 | 16 |
| memcpy / memset | 0.35 | 0.2 | 100 | 0.11 | 0.3 | 17 |
| **Total GPU kernel time** | **139.3** | | **1346** | **40.8** | | **393** |

Top individual kernels per training step: `ampere_sgemm_128x64_tn` 26.7 ms (43 calls; the
3072×3072 Q/K/V/O projections), `ampere_sgemm_128x128_nn` 15.8 ms, `ampere_sgemm_128x64_nt`
12.8 ms, `ampere_sgemm_128x64_nn` 10.0 ms, `ampere_sgemm_128x128_nt` 7.9 ms.

Achieved throughput [D]: the analytic forward cost is ≈ 537 GFLOP. That is the main encoder
(8 × 54.8 GFLOP) plus the three cross-variable MHAs, whose K/V projections run over
512 × 13 = 6656 variable tokens (≈ 84 GFLOP). At 38.3 ms per forward this is **≈ 14 TFLOP/s =
~72% of the A100's 19.5 TFLOP/s FP32 peak**. The FP32 GEMMs are already efficient; the headroom
is in tensor cores (156 TFLOP/s TF32, 312 TFLOP/s BF16 dense).

### 3.6 Nsight Compute (selected forward kernels, A100-80GB, ncu-locked clocks)

| Kernel | Duration | SM % | Memory % | DRAM % | Achieved / theor. occupancy |
|---|---:|---:|---:|---:|---|
| `ampere_sgemm_128x64_tn` (grid 1664) | 1.70 ms | **93.6** | 46.9 | 2.8 | 23.1 / 25 |
| `ampere_sgemm_32x32_sliced1x4_tn` (grid 512) | 61–165 µs | 54–56 | 58–60 | ~4 | ~20 / 25 |
| attention-bias elementwise (grid 26,624 / 53,248) | 26–78 µs | 44–55 | 36–76 | 18–76 | 40–87 / 100 |
| per-variable token-embedding GEMM `ampere_sgemm_128x32_tn` (grid 128) | ~10.4 µs | 16–20 | 18–23 | ~6 | 12–19 / 50 |
| per-variable copy `elementwise_kernel` (grid 60) | ~5.3 µs | **0.75** | 9–11 | 9–11 | **~6** / 100 |
| `softmax_warp_forward` (grid 512) | 4.2 µs | 13.6 | 9.7 | 5.1 | 27 / 100 |

The big GEMMs are **compute-bound on the FP32 pipe**. The token-embedding loop
(13 variables × 3 branches, each a `Linear` plus a copy) is a train of tiny launch-bound kernels
at <1–20% SM utilisation.

### 3.7 Attention core in isolation (`profiling/bench_attention_core.py`)

Upstream `MultiHeadAttention.scaled_dot_product_attention` at [1, 16, 512, 192]:
QKᵀ → softmax → recompute `1/(1+dist)`, `1/(1+ang)` → `α·S + β·D + γ·A` → softmax → ·V.

| | ms per layer | × 8 layers | share |
|---|---:|---:|---|
| core forward | 0.48 | 3.9 ms | 12.7% of an encoder layer, ≈10% of the model forward |
| core forward + backward | 1.54 | 12.3 ms | 13.5% of a layer's fwd+bwd, ≈8% of the train step |
| full EncoderLayer forward | 3.79 | 30.3 ms | the main encoder is ≈79% of the forward |
| full EncoderLayer forward + backward | 11.37 | 91.0 ms | |

## 4. Bottleneck ranking (Phase 6)

**Measured** = numbers above. **Hypothesis** = expected effect of an optimisation that has not
been tried.

| # | Operation | CUDA time | % GPU time | Memory impact | Possible optimisation (hypothesis) |
|---|---|---|---|---|---|
| 1 | FP32 GEMMs: linear fwd/bwd + attention bmm | 98.9 ms/train step; 35.1 ms/forward | 71% train, 86% infer | weights 2.34 GiB fp32 | Tensor cores: `allow_tf32=True` (no code change), then BF16 autocast. GEMMs are compute-bound at ~72% of the FP32 peak, so tensor cores are the main lever |
| 2 | AdamW foreach (`multi_tensor_apply`, ~12 passes over 426 M params) | 27.0 ms/step | 19% train | m + v = 3.17 GiB | `AdamW(fused=True)`: one pass, memory-bound. Optionally BF16 or 8-bit states |
| 3 | Host data prep + syncs: GPU idle inside the step | 11.4 ms idle/train step; 7.1 ms/infer step | 7.6% / 14.9% of wall | – | Distance/angularity are constant: convert and upload once instead of every step. Prefetch the next window; drop the per-step `.item()` |
| 4 | Full-dataset gather `model_inputs[:, loc_indices, :]` (gathers all 10305 days before taking the 30-day window; `loc_indices` is the identity) | 2.55 ms/step, ~0.83 ms per 400 MB gather | 1.8% train, 6.3% infer | +400 MB transient per call, 3×/step | Index days first or skip the identity station gather: ~300× fewer bytes |
| 5 | Geo-biased attention core (QKᵀ, 2 softmax, bias recompute, mix, AV) | 3.9 ms/fwd; 12.3 ms/train step | ≈10% fwd, ≈8% train | 3+ × [16, 512, 512] fp32 per layer (56 MiB peak); saved for backward | **Fused Triton kernel**, fwd + bwd. SDPA/FlashAttention can't express softmax → biased re-mix → softmax. A 512-key row fits in SRAM; precompute the static bias once per model, not per layer per step |
| 6 | Small-kernel / launch overhead (1246 launches/step; token-embedding loop at 0.75–20% SM; "other elementwise" 4.4 ms) | ~5–7 ms/step | 3–5% | – | Batch the 13 per-variable `Linear`s into one einsum/grouped GEMM; CUDA Graphs (all shapes are static: always 512 stations); `torch.compile` |
| 7 | `CUDA_LAUNCH_BLOCKING=1` in the original daily/weekly scripts (config, not a kernel) | +34 ms/train step, +10 ms/forward | +23% / +26% wall | – | Remove the line. Plan B already does |
| 8 | Dead `CrossVarTransformer.encoder_layers` | 0 ms (never called; no grads, so AdamW skips them) | 0 | 0.75 GiB GPU, ~0.8 GB of every 5.9 GB checkpoint | Drop them from the module (requires a state-dict migration) |

**Ranking after the first fix (hypothesis).** If tensor cores cut the GEMMs ~4–8×, the step
falls from ~148 ms to roughly 55–75 ms. AdamW (27 ms), host idle (11 ms) and the attention core
(12 ms) then make up most of what remains. That is when the fused-attention and fused-optimizer
work becomes the critical path.

## 5. Ranked CUDA optimisation opportunities

1. **Tensor-core enablement (TF32 → BF16 autocast).** Largest expected gain, smallest code
   change. A configuration project, not a CUDA one, but it is the baseline that every later
   kernel speedup must be measured against.
2. **Fused optimizer.** `fused=True` AdamW; or a custom fused AdamW + gradient-norm Triton kernel
   if you want to own the kernel. Memory-bound, so it can be judged against a bandwidth roofline.
3. **Fused geo-biased attention (Triton, fwd + bwd)**. **Best custom-kernel candidate**:
   - it is this model's own operator, and no stock kernel implements it;
   - it is memory-bound with a clean roofline story: it materialises several 16 MiB
     [16, 512, 512] tensors per layer per direction;
   - its share grows once GEMMs move to tensor cores (Amdahl);
   - the eager implementation is a ready-made correctness oracle (`torch.allclose`).
4. **Input-pipeline fixes**: static bias tensors on the GPU, an index-first gather, async
   prefetch. Recovers 7–15% of wall time; no kernel writing needed.
5. **Launch-overhead reduction**: a grouped token-embedding GEMM and CUDA Graph capture of the
   static-shape inference step. Small absolute gain, good demonstration of Nsight Systems
   timeline reading.

## 6. Measured vs missing

| Item | Status |
|---|---|
| Forward / backward / optimizer / step timing (all 3 variants, both launch modes) | **Measured** |
| Inference per-step and per-season latency | **Measured** (Plan B real checkpoint); season latency for the original eval loops **derived** |
| Peak allocated / reserved memory, train + infer | **Measured** |
| Operator and kernel breakdown, launches, idle, transfers | **Measured** (torch.profiler + nsys) |
| Kernel-level SOL / occupancy | **Measured** for 30 forward kernels (ncu); backward / optimizer kernels **not** profiled with ncu |
| Results under the exact historical stack (torch 2.4.1+cu121) | **Not measurable**: env purged. The daily and weekly `CLB=1` steps agree with the historical epoch times within 4–5% |
| Historical Plan B 233 ms/step vs measured 148 ms | **Unexplained**. Per-step `flush()` to the home filesystem is an unverified hypothesis |
| Full-season, full-epoch or multi-GPU runs | Deliberately **not run** (bounded profiling only) |
| GP head (GPyTorch) GPU profile | **Not profiled** (outside this attention-model scope); its memory is documented in `gp_oom_diagnosis.md` |

## 7. Artifacts

- Scripts: `engineering_audit/profiling/{profile_foreswe.py, run_profiling.sh, bench_attention_core.py, nsys_gpu_idle.py, kernel_categories.py, ncu_summarize.py}`.
- Small results (git-safe): `engineering_audit/profiling/results/` (bench JSONs, kernel-category JSONs, nsys CSV summaries, ncu summary, operator tables).
- Heavy outputs (not in git): `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/foreswe_gpu_profiling_v1/`
  (`torch_trace_*.json` Chrome traces, `nsys_planB24.nsys-rep` / `.sqlite`, `ncu_planB24_infer.csv`, logs),
  mirrored at `artifacts/foreswe_gpu_profiling_v1`.
