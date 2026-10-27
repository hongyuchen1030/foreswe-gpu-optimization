# AdamW: foreach vs fused (Plan B ForeSWE, A100-SXM4-40GB)

All numbers are **newly measured** on 2026-10-26 (PyTorch 2.6.0, CUDA 12.4, driver 580.178.04) unless
marked otherwise. Source data: `benchmark_results/fused_adamw/fused_adamw_study.json` (node 2, job 60335651,
nid003700) and the E0–E5 matrices in `benchmark_results/node{1,2}_synthetic/`.

## Setup

- Production hyper-parameters: `AdamW(lr=5e-6, weight_decay=1e-4)`, betas/eps default, FP32 params.
- foreach = `foreach=True, fused=False` (production default on CUDA); fused = `foreach=False, fused=True`.
- 627.6 M parameters, of which **426.0 M are live** (the 201.6 M in the never-called
  `CrossVarTransformer.encoder_layers` receive no gradient, so both implementations skip them).
- **Correctness isolation:** models A (foreach) and B (fused) start bit-identical. Each step the gradient is
  computed once on A (real Plan B training dates, eval mode so the graph is deterministic) and copied
  bit-for-bit into B. Any A/B difference therefore comes only from the optimizer kernels.
- **Timing:** `optimizer.step()` alone on fixed gradients, 10 warm-up + 50 measured, CUDA events; kernel
  launches from `torch.profiler`; memory from the CUDA caching allocator.

## Performance

| Implementation | step() median | IQR | kernel launches / step | optimizer state | transient peak |
|---|---:|---:|---:|---:|---:|
| single-tensor (for-loop) | 37.59 ms | 0.21 | 2,168 | 3.17 GiB | 0.07 GiB |
| **foreach (production)** | **27.97 ms** | 0.04 | 175 | 3.17 GiB | **1.59 GiB** |
| **fused** | **10.19 ms** | 0.06 | **26** | 3.17 GiB | **0.00 GiB** |

- Fused is **2.75× faster** than foreach in isolation (27.97 → 10.19 ms), with **6.7× fewer launches**,
  and it needs no 1.59 GiB of foreach temporaries (one gradient-sized intermediate).
- **Bandwidth check [derived]:** a fused AdamW step must read the param, grad, `exp_avg` and `exp_avg_sq`
  and write the param and both moments: 7 × 4 B × 426 M ≈ 11.9 GB. At 10.19 ms that is **≈ 1.17 TB/s,
  about 75 % of the A100-40GB's 1.555 TB/s HBM2 peak**. The fused kernel is near the memory roofline.
  foreach does about 12 passes (≈ 22 GB) at about 0.8 TB/s.
- **Inside the full training step** (matched matrices, median of 120 steps):

| | foreach | fused | Δ step |
|---|---:|---:|---:|
| FP32 (E0 → E3) | 147.6 ms | 130.6 ms | −17.0 ms (1.13×) |
| TF32 (E1 → E4) | 69.3 ms | 52.1 ms | −17.2 ms (1.33×) |
| BF16 (E2 → E5) | 74.5 ms | 56.5 ms | −18.1 ms (1.32×) |

  Optimizer phase: 27.1 → 9.5 ms. Peak allocated memory: 9.11 → 8.50 GiB (−0.61 GiB) for FP32/TF32.
  The saving matters more once mixed precision has shrunk the GEMM time (Amdahl): AdamW is 39 % of the
  E1 step but only 18 % of E4.

## Correctness

| Check | Result |
|---|---|
| One step, parameters: max \|Δ\| | 1.49e-08 (≤ 1 ulp at the parameter scale) |
| One step, parameters bit-identical | **99.89 %** |
| One step, update vector rel-L2 (B vs A) | **1.83e-06**, cosine 1 − 1.7e-12 |
| One step, `exp_avg` rel-L2 / max \|Δ\| | 3.8e-08 / 2.4e-07 |
| One step, `exp_avg_sq` rel-L2 / max \|Δ\| | 5.0e-08 / 1.2e-07 |
| 50 steps, shared gradients: param rel-L2 | **5.0e-09** (max \|Δ\| 1.2e-07, 97.3 % bit-identical) |
| 50 steps, shared gradients: moments rel-L2 | `exp_avg` 5.3e-08, `exp_avg_sq` 3.0e-07 |
| 50 steps, free-running (each model computes its own gradients) | param rel-L2 2.4e-06; losses identical to all printed digits (212,682.1 at step 50) |

The differences come from floating-point evaluation order (the fused kernel evaluates
`p − lr·m̂/(√v̂+ε)` with a different operation sequence and FMA contraction). They do **not** grow
systematically: under shared gradients drift stays at ≈ 1 ulp-scale, and the 2.4e-06 free-running drift is
chaotic amplification through the network rather than optimizer bias. Fused AdamW is **numerically
equivalent** to foreach for this model.

**Algorithm comparison (separate, not an implementation change):** `Adam(fused, weight_decay=1e-4)` vs
`AdamW(fused, weight_decay=1e-4)` gives an update rel-L2 of **0.89** and cosine 0.71. Coupled L2
regularisation changes the algorithm by orders of magnitude more than the implementation swap does.

## Unexpected finding: weight decay is a no-op in FP32

`error_origin.json → optimizer`: with the same gradients and fresh state, running AdamW with
`weight_decay=1e-4` and with `0` gives **bit-identical parameters in all 271 live tensors**, for both foreach
and fused. Decoupled decay multiplies each weight by `1 − lr·wd = 1 − 5e-10`, which is below half an FP32 ulp
(2⁻²⁴ ≈ 6e-8 relative), so `p·(1 − 5e-10)` rounds back to `p`. The production configuration has therefore
never applied weight decay. Not changed here (that would alter the training algorithm), but it is a concrete
case where **compensated (Kahan-style) master-weight accumulation**, or applying decay as a separate scaled
term, would change results (see `NUMERICAL_ERROR_ANALYSIS.md`).

Also measured: on the first update, **28.0 %** of the per-element exact (FP64) updates are below half an ulp
of their FP32 weight and are fully absorbed. Not separated in this run: this count includes elements whose
gradient is zero or ≲ ε, for which only the 5e-10 decay term remains. The applied update differs from an
FP64-exact update by rel-L2 1.3e-4.

## Conclusion

Fused AdamW is a free, exact-to-rounding replacement for this model. It saves 17–18 ms per step (≈ 1.13× at
FP32, ≈ 1.33× once Tensor Cores are on), removes 1.59 GiB of transient memory and runs within ≈ 75 % of HBM
bandwidth. A custom optimizer kernel could gain at most ≈ 25 % more on that phase. Larger wins now lie
elsewhere (see `PRECISION_PERFORMANCE_REPORT.md`).
