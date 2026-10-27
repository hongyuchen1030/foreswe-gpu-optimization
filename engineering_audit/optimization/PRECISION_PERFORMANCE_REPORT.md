# Precision & optimizer study: FP32 vs TF32 vs BF16 × AdamW foreach vs fused

**Model:** Plan B 24-week ForeSWE attention model (627.6 M params, 426.0 M live), production batch
(1 issue date × 512 stations), production AdamW (lr 5e-6, wd 1e-4), MSE loss.
**Everything below is newly measured on 2026-10-26** unless labelled *historical* or *derived*.

## Environment

| | Node 1 (primary) | Node 2 (replicate + Nsight) |
|---|---|---|
| SLURM job / host | 60335641 / nid003640 | 60335651 / nid003700 (`--perf=generic` for ncu) |
| GPU | A100-SXM4-40GB (UUID GPU-4eee3451…), 108 SMs, max SM clock 1410 MHz | A100-SXM4-40GB |
| Software | NERSC `pytorch/2.6.0` (Python 3.12.9, torch 2.6.0, CUDA 12.4, cuDNN 9.5), driver 580.178.04, repo commit at run time in each JSON | same |
| Launch mode | `CUDA_LAUNCH_BLOCKING` unset (asserted by the harness) | same |

The two nodes have the same GPU model and software. Each node has its own FP32 baseline. The headline table
uses node 1; node 2 reproduces every step time within 2 % (E4: 50.8 vs 52.1 ms) and is reported separately
below. No speedup mixes the two nodes.

## Method

- **Matrix:** E0–E5 = {FP32, TF32, BF16-autocast} × {foreach, fused}. Identical seed-0 initial weights
  (built on CPU), identical synthetic production-shaped inputs (`model_inputs` 10305×512×19 resident on GPU,
  as in production), identical training-date sequence, lr, wd, loss.
- **TF32:** `allow_tf32=True` (+ `float32_matmul_precision("high")`). **BF16:** `torch.autocast(bf16)` over
  forward + loss, TF32 off, FP32 master weights, gradients and AdamW state, no loss scaling (not needed for bf16).
- **Timing:** each (experiment, repeat) runs in a **fresh process**; 3 repeats with rotated order;
  10 warm-up + 40 measured steps each, so 120 steps per configuration. Every phase (prep, forward + loss,
  `loss.item()`, `zero_grad`, backward, optimizer) is bracketed by `torch.cuda.Event`s, with
  `cuda.synchronize()` around the step and `perf_counter` for wall time. Median and IQR reported.
  Model construction is excluded.
- **Profiling:** `torch.profiler` (3 steady-state steps per config), Nsight Systems (8 steps,
  NVTX per phase, SQLite analysis of busy/idle), Nsight Compute (12 GEMM launches per precision:
  SM / DRAM / **tensor-pipe** / FMA-pipe utilisation).
- **Numerics and convergence:** real ForeSWE data. See `NUMERICAL_ERROR_ANALYSIS.md` and Phase 3 below.

## Final table (node 1; step = median of 120 steps)

| Experiment | Step ms (IQR) | Speedup | Forward ms | Backward ms | Optimizer ms | Peak VRAM alloc / reserved | Output error vs FP32 | Gradient error vs FP32 | Loss after 300 steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **E0** FP32 + foreach (production) | 148.19 (1.14) | 1.00× | 39.66 | 72.67 | 27.00 | 9.11 / 9.88 GiB | – | – | 187,807 |
| E1 TF32 + foreach | 69.53 (0.42) | **2.13×** | 13.26 | 20.67 | 27.03 | 9.11 / 9.88 GiB | 3.9e-4 | 2.2e-4 | 187,807 |
| E2 BF16 + foreach | 74.72 (0.36) | **1.98×** | 16.89 | 21.02 | 28.29 | 9.13 / 10.46 GiB | 5.7e-3 | 3.6e-3 | 187,800 |
| E3 FP32 + fused | 129.80 (0.45) | **1.14×** | 39.71 | 72.69 | 9.45 | 8.50 / 9.03 GiB | 0 (same math) | 0 | 187,807 |
| **E4 TF32 + fused** | **50.75** (1.46) | **2.92×** | 13.29 | 20.63 | 9.45 | **8.50 / 9.03 GiB** | 3.9e-4 | 2.2e-4 | 187,806 |
| E5 BF16 + fused | 56.59 (0.54) | **2.62×** | 16.77 | 20.64 | 10.55 | 9.13 / 9.53 GiB | 5.7e-3 | 3.6e-3 | 187,800 |

Error columns: relative L2 vs FP32 (TF32 off), median over 9 real issue dates, trained Plan B weights, eval
mode, before any update. Speedup range over the 3 repeats: E1 2.11–2.14×, E2 1.97–1.99×, E3 1.14×,
E4 2.84–2.94×, E5 2.61–2.63×. Data prep (3 full-dataset gathers + H2D) is 6.8–8.3 ms in every
configuration and is included in the step.

**Throughput** (issue dates / s; ×512 station samples): E0 6.75, E1 14.38, E2 13.38, E3 7.70, **E4 19.70**,
E5 17.67.

**Inference (forward only, Feb-1-2017 issue):** FP32 39.2 ms, **TF32 12.1 ms (3.25×)**, BF16 14.2 ms (2.77×).
The inference peak memory in the matrix (7.9–8.3 GiB) was measured after training in the same process, so
it includes resident optimizer state. A clean inference-only peak is 3.13 GiB (FP32, `GPU_PROFILING_REPORT.md`).

**Node 2 replicate (own FP32 baseline):** E0 147.64, E1 69.31 (2.13×), E2 74.53 (1.98×), E3 130.64 (1.13×),
E4 52.12 (2.83×), E5 56.45 (2.62×) ms.

**Historical reference (not re-measured conditions):** 148.4 ms/step (FP32, foreach) from
`GPU_PROFILING_REPORT.md`; new E0 = 148.19 ms on the same GPU model and stack.

## Where the time went (torch.profiler, ms of GPU kernel time per step)

| | E0 | E1 | E2 | E3 | E4 | E5 |
|---|---:|---:|---:|---:|---:|---:|
| GEMM on **SIMT FP32** (`ampere_sgemm_*`) | 98.7 | 1.1 | 0.4 | 98.7 | 1.1 | 0.4 |
| GEMM on **Tensor Cores** | 0 | 16.8 | 9.7 | 0 | 16.8 | 9.7 |
| AdamW (`multi_tensor_apply` kernels; foreach or fused) | 26.9 | 26.9 | 27.0 | 9.4 | 9.4 | 9.5 |
| dtype casts / copies | 1.7 | 1.7 | **8.1** | 1.8 | 1.7 | **8.1** |
| other elementwise + reductions | 7.2 | 7.2 | 6.1 | 7.2 | 7.2 | 6.1 |
| index gather (`model_inputs[:, loc, :]`) | 2.5 | 2.5 | 2.5 | 2.6 | 2.6 | 2.6 |
| softmax + LayerNorm | 1.3 | 1.3 | 1.4 | 1.3 | 1.3 | 1.4 |
| **total GPU kernel time** | 139.3 | 58.1 | 55.6 | 121.9 | 40.6 | 38.2 |
| kernel launches / step | 1,346 | 1,311 | 1,832 | 1,197 | 1,162 | 1,683 |

**Nsight Systems (per training step, median of 5 steps after warm-up):**

| | E0 | E1 | E2 | E3 | E4 | E5 |
|---|---:|---:|---:|---:|---:|---:|
| GPU busy | 92.5 % | 72.2 % | 58.0 % | 91.1 % | 63.4 % | 49.6 % |
| GPU idle within step | 11.2 ms | 22.4 ms | 40.4 ms | 12.0 ms | 23.5 ms | 38.7 ms |
| Tensor-Core share of GEMM time | 0.3 % | 93.9 % | 95.7 % | 0.3 % | 93.9 % | 95.7 % |
| H2D per step | 2.11 MB / 0.09 ms | same | same | same | same | same |
| D2H per step | 4 B (`loss.item()`) | same | same | same | same | same |

**Tensor-Core evidence (Nsight Compute, hardware counters, A100-40GB):**

| Precision | Largest encoder GEMM kernel | SM throughput | **Tensor pipe active** | FMA pipe |
|---|---|---:|---:|---:|
| FP32 | `ampere_sgemm_128x64_tn` | 84 % | **0 %** | 93 % |
| TF32 | `cutlass_80_tensorop_s1688gemm_256x128_32x3_tn` | 78 % | **82 %** (85 % on 128×128 tiles) | 24 % |
| BF16 | `ampere_bf16_s16816gemm_bf16_128x128_…` | 68 % | **80 %** | 0.9 % |

So yes: TF32 and BF16 run the GEMMs on Tensor Cores (82–85 % / 80 % tensor-pipe utilisation on the large
GEMMs), FP32 does not (0 %, FMA-pipe bound at 93 %).

## Interpretation

1. **Tensor Cores remove the GEMM bottleneck.** GEMM time drops 98.7 → 16.8 ms (TF32, 5.9×) and → 9.7 ms
   (BF16, 10.2×). Forward 3.0×, backward 3.5×.
2. **BF16 is slower end-to-end than TF32 here (56.6 vs 50.8 ms with fused AdamW)** even though its GEMMs
   are 1.7× faster. Autocast re-casts the FP32 master weights and activations every step: +8.1 ms of
   cast kernels and +520 launches per step. The GPU is then idle 39–40 ms per step (busy 50 %), waiting on
   the host to launch ~1,700 small kernels. At batch 1 × 512 tokens the model is no longer GEMM-bound
   after the precision change; it is **launch/host-bound**.
3. **Fused AdamW** saves 17–18 ms in every precision (27 → 9.5 ms) and 0.61 GiB peak memory; it is
   numerically equivalent to foreach (`FUSED_ADAMW_REPORT.md`).
4. **Combined:** E4 (TF32 + fused) is the fastest, **2.92× over production**, with the smallest memory and
   4e-4 output error. Remaining time per step: ≈ 40.6 ms GPU kernels + ≈ 23 ms idle.

## Phase 3: convergence (real data, matched budget)

Same seed-0 weights, production data order (first Plan B training dates, chronological), dropout seed
1234, train mode (dropout on), production AdamW. `benchmark_results/convergence/` has per-step loss,
gradient norm and update norm. Figures: `figures/convergence_300steps.png`, `figures/convergence_3297steps.png`.

| | 300 steps: final loss | mean of last 50 | NaN/Inf steps | final ‖g‖ |
|---|---:|---:|---:|---:|
| E0 | 187,807 | 273,288 | 0 | 29,124 |
| E1 | 187,807 | 273,287 | 0 | 29,125 |
| E2 | 187,800 | 273,284 | 0 | 29,090 |
| E3 | 187,807 | 273,288 | 0 | 29,125 |
| E4 | 187,806 | 273,287 | 0 | 29,125 |
| E5 | 187,800 | 273,284 | 0 | 29,089 |

The loss curves overlay to within line width. The loss is not monotone over 300 steps because the data
order is chronological: consecutive issue dates walk through a water year, and the per-date MSE follows
the seasonal SWE magnitude. Relative agreement between configurations is the quantity of interest.

**One full epoch (3,297 steps) for E0, E1, E4, E5.** E2 and E3 were not extended: their 300-step behaviour
matched E1 and E0, and the allocation time went to the promising configurations.

| | final loss | mean of last 50 | NaN/Inf steps | final ‖g‖ | epoch wall time | test-prediction rel-L2 vs E0 | test RMSE (mm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| E0 FP32 + foreach | 148,552 | 132,046 | 0 | 70,636 | 521 s | 0 | 370.41 |
| E1 TF32 + foreach | 148,420 | 132,014 | 0 | 70,062 | 265 s | 4.8e-2 | 370.53 |
| E4 TF32 + fused | 148,491 | 132,035 | 0 | 70,242 | 204 s | 2.8e-2 | 370.56 |
| E5 BF16 + fused | 148,475 | 131,988 | 0 | 70,479 | 223 s | 2.3e-2 | 370.43 |

Test predictions are on the five Feb-1 issue dates × 512 stations × 24 leads. After 300 steps they still
match E0 closely (rel-L2: E3 2.4e-5, E1 1.5e-4, E4 1.8e-4, E2 3.2e-3, E5 3.3e-3). After a full epoch the
**weights** have diverged (2–5 % prediction difference, up to 32 mm at one station): small per-step
differences compound through 3,297 noisy single-sample updates. The divergence is trajectory chaos, not
precision bias:
- E1 (TF32) diverges more than E5 (BF16).
- Every run lands on the same loss (last-50 mean within 0.05 %) and the same test RMSE (370.4–370.6 mm, within 0.04 %).
- Even E3 vs E0, which differ only in AdamW evaluation order, differ by 2.4e-5 after just 300 steps.

**Measured wall-clock training speedup over a real epoch** (including data prep, logging and gradient-norm
bookkeeping): E0 521 s → E4 204 s (**2.55×**), E5 223 s (2.34×), E1 265 s (1.97×).

**Conclusion (Phase 3):** within this matched budget (300 steps; one epoch for E0/E1/E4/E5), TF32 and BF16 (and fused vs foreach) produce no
measurable change in training behaviour: same losses (300 steps: 4–5 significant figures; one epoch: last-50 mean within 0.05 %), same gradient norms,
no instabilities. This is a sanity check, not proof of equivalence over a full multi-epoch training run.

## Best observed trade-off

**E4 = TF32 + fused AdamW:** 2.92× faster training step (148.2 → 50.8 ms), 3.25× faster inference
forward, −0.61 GiB peak memory, output error 3.9e-4 and gradient error 2.2e-4 relative to FP32, identical
short-run convergence. It needs no code change to the model: two flags (`allow_tf32`, `fused=True`).

## Next bottlenecks (measured, in priority order)

1. **Host / launch overhead:** 23 ms idle per E4 step, 39 ms per E5 step, 1,160–1,680 launches. Candidates:
   CUDA Graphs (all shapes are static: 512 stations, fixed windows) or `torch.compile`.
2. **BF16 cast traffic:** 8.1 ms per step. Keep a BF16 weight copy updated by the optimizer, or fuse the
   cast into the GEMM prologue.
3. **AdamW:** 9.5 ms at ~75 % of HBM bandwidth. Little headroom left.
4. **Data path:** 6.8–8.3 ms prep per step (full 400 MB dataset gather ×3 + numpy distance matrices),
   unchanged by precision. Now ~15 % of the E4 step.
