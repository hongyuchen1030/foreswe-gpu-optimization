# Kernel Tiling Optimization: D = 192 without padding

Task 1 of the fused geo-attention follow-up. The original prototype (`foreswe_bench/geo_attention.py`, kept
**unchanged** as the reference) is compared with the chunked-D kernel in `foreswe_bench/geo_attention_v2.py`.

All timings: one A100-SXM4-40GB (Perlmutter), PyTorch 2.6.0, Triton 3.2. Production shape Q/K/V
[1, 16, 512, 192]. Inputs are the real layer-0 activations captured from the trained Plan B checkpoint, using the
same cached geometry (distance/angularity) matrices for every implementation. Timings are CUDA-event medians
from `benchmarks.attention_tiling.timings`: forward alone, backward alone (`retain_graph`), and forward+backward.

## 1. The problem: padding 192 to 256

The v1 kernel uses one `BLOCK_D = next_power_of_2(D) = 256` tile for the head dimension, because `tl.arange`
needs a power-of-two length. For D = 192, one third of every Q/K/V/dO tile is masked padding:

- every `tl.dot` over the head dimension does 256/192 = 1.33× the MMA work that is needed;
- the padded tiles take 33 % more shared memory and registers, which pushed v1 to small tiles
  (BLOCK_M = 16, BLOCK_N = 32, one pipeline stage) to fit at all.

## 2. The change: three 64-wide chunks

`geo_attention_v2.py` splits D into `NC = cdiv(D, 64)` chunks of `DC = 64` (NC = 3 for D = 192; up to 3
are supported). Each chunk is a power of two, so there is no padding when D is a multiple of 64. For other D
values the last chunk is masked, as the shape tests do (D = 48, 64).

- Scores: `S = (Σ_c Q_c K_cᵀ) / √d`. Up to three chunk dots are summed (`_dot3`).
- Outputs and gradients that have a D dimension (O, dQ, dK, dV) keep one accumulator per chunk
  (`acc0..acc2`, `dk0..dk2`, `dv0..dv2`), each [BLOCK, 64].
- Loads go through `_ld_rows` / `_ld_cols`, which mask the chunk against D.

The algorithm and the math are the same as v1: two-pass forward with online softmax #1 statistics, the
`c = |α|+|β|+|γ|` bound for softmax #2, a row kernel for dQ (and δ₁/scalar partial sums), and a column kernel for
dK/dV. The number of passes is unchanged. The only structural change is per-kernel launch configuration: v1
used one (BLOCK_M, BLOCK_N) for all kernels, and v2 tunes the forward, row and column kernels separately
(`TUNED_TILES`).

## 3. Bounded sweep

`python -m benchmarks.attention_tiling` sweeps each kernel separately (coordinate descent: forward first, then
row kernel with the best forward fixed, then column kernel), with BLOCK_M ∈ {16, 32, 64} (row kernel {16, 32}),
BLOCK_N ∈ {32, 64} (column kernel {16, 32}), num_warps ∈ {4, 8} and num_stages ∈ {1, 2}: 64 configurations in
total. Every configuration is first checked for correctness against FP64 on random inputs (gate), then timed
on real layer 0 in TF32. The sweep ran with the numerics that existed at the time (before the accuracy fix in
`SATURATED_SOFTMAX_BACKWARD_DEBUG.md`). The fix does not change tile shapes or how many loads each tile does.

| Kernel | Configs | Out of shared memory | Best (BM, BN, warps, stages) | Best fwd+bwd ms | Worst fwd+bwd ms |
|---|---|---|---|---|---|
| forward | 24 | 2 | (32, 64, 4, 2) by fwd ms 0.441 | 1.648 | 1.926 (32, 32, 8, 1) |
| row (dQ, δ₁, dα/dβ/dγ) | 16 | 4 | (32, 32, 4, 1) by bwd ms 1.355 | 1.561 | 2.022 (16, 32, 8, 1) |
| column (dK, dV) | 24 | 8 | (32, 32, 8, 2) by bwd ms 1.269 | 1.481 | 1.871 (16, 16, 8, 1) |

Every config that compiled passed the correctness gate. The out-of-resources failures are shared memory: for
example, BLOCK_M = 64 in the column kernel needs 208–229 KB, and the A100 limit is 163 KB.

The spread between correct configurations is at most about 30 % (1.48 vs 1.87 ms in the column sweep). Most
configurations land within about 15 % of the best. So the main gain comes from removing the padding, not from
tile search. With the v2 forward kernel at v1's tile shape (16, 32, 4, 1) and the backward kernels at their pre-sweep
defaults, fwd+bwd is already 1.87 ms, against 3.10 ms for v1. Removing padding is worth about 1.66×, and tuning adds about 1.2×.

Full sweep: `benchmark_results/tiling/tiling_sweep.csv`. Per-config launch info:
`benchmark_results/tiling/tiling_results.json`.

## 4. Result (TF32, real layer 0, final run)

From `benchmark_results/final/final_comparison.csv`, measured in the same process back-to-back:

| Implementation | Fwd ms | Bwd ms | Fwd+Bwd ms | Kernels (fwd+bwd) | Extra peak memory |
|---|---|---|---|---|---|
| Original Triton (v1, padded, 16×32) | 0.863 | 2.359 | 3.007 | 17 | 8.2 MiB |
| Tuned tiling (v2 `fast_experimental`) | 0.445 | 1.311 | 1.528 | 19 | 8.2 MiB |
| **speed-up** | **1.94×** | **1.80×** | **1.97×** | | |

The tiling sweep's own final measurement gave 0.900 / 2.459 / 3.101 ms (v1) and 0.434 / 1.332 / 1.546 ms (v2):
2.01× fwd+bwd. The v2 fast preset has 2 more kernels than v1 because it zero-fills the separate scalar-gradient
partial buffer.

## 5. Resources (TF32 build, `benchmark_results/final/launch_resources.json`)

| Kernel (accurate preset) | Tile | Regs/thread | Spills | Shared memory | Blocks/SM | Theor. occupancy | Limiter |
|---|---|---|---|---|---|---|---|
| forward | 32×64, 4 warps, 2 stages | 255 | 10 | 144 KB | 1 | 6.25 % | shared memory |
| row B0 (δ₂) | 32×32, 4 w, 1 st | 255 | 8 | 56 KB | 2 | 12.5 % | registers |
| row B1 (δ₁, dα/dβ/dγ) | 32×32, 4 w, 1 st | 255 | 8 | 56 KB | 2 | 12.5 % | registers |
| row B2 (dQ) | 32×32, 4 w, 1 st | 255 | 28 | 56 KB | 2 | 12.5 % | registers |
| column (dK, dV) | 32×32, 8 w, 2 st | 255 | 66 | 64 KB | 1 | 12.5 % | registers |

- **Tensor cores:** every Q·Kᵀ, P·V, dO·Vᵀ, Pᵀ·dO, dS·K and dSᵀ·Q product is a `tl.dot`, which Triton lowers to
  `mma.sync` TF32 (HMMA) when `allow_tf32` is set. In FP32 (`input_precision="ieee"`) it lowers to FFMA, and
  tensor cores are not used. That explains the FP32 timings: the v2 backward takes 16.8 ms in FP32, against 1.1 ms
  for eager FP32 (cuBLAS's FP32 SGEMM is far better tuned than a Triton FMA loop).
- **Occupancy:** every kernel is at the 255-register cap, at 6–12.5 % theoretical occupancy. Three D-chunks of
  accumulators per row block (3 × [32, 64] FP32 = 96 registers per thread at 4 warps) plus the recomputed score,
  P₁, P₂, dZ and dS tiles do not fit without spills. That is the limit of this algorithm in Triton. The next step
  would need warp specialisation or splitting the backward by D-chunk, which this project's scope excludes
  (no CUDA C++ rewrite, no large autotune).
- **Memory:** extra peak memory over inputs and outputs is 8.2 MiB for every Triton variant: per-row statistics
  M₁, L₁, L₂, δ₁, δ₂ and the scalar partial sums. Eager uses 76.0 MiB and Inductor 58.1 MiB, because both
  materialise the [16, 512, 512] S, P₁, Z and P₂ tensors.

## 6. Reproduce

```bash
JOB=$(bash scripts/slurm/alloc.sh fsw_kern)   # 4 h interactive A100; scancel $JOB when done
bash scripts/slurm/run.sh $JOB python -m benchmarks.attention_tiling
bash scripts/slurm/run.sh $JOB python -m benchmarks.attention_final
bash scripts/slurm/run.sh $JOB python -m benchmarks.attention_resources
```
