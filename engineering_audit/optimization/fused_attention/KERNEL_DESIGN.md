# Fused geo-biased attention: kernel design (CUDA C++ / CUTLASS track)

The design for a production-quality fused kernel, informed by the measured Triton prototype
(`FUSION_FEASIBILITY_REPORT.md`). Operator, shapes and gradients: `ATTENTION_MATH_AND_GRADIENTS.md`.
Everything below is a design; performance figures marked *estimate* are not measurements.

## Targets and constraints

- Shape: B = 1, H = 16, N = 512, d = 192 (8 layers); FP32 activations with TF32 Tensor-Core math (E4),
  optional BF16.
- Must match eager within eager's own FP32/TF32 error vs FP64 (oracle: `eager_geo_attention`), including
  dα, dβ, dγ and the saturated-softmax rows observed in the trained model.
- Graph-capturable: no host syncs; α, β, γ read from device memory.
- Budget per SM (A100): 164 KB shared memory, 64K 32-bit registers, 4 tensor-core sub-partitions.

## Forward kernel (one CTA per (head, 64-row query block): 16 × 8 = 128 CTAs ≈ 1 wave on 108 SMs)

1. Load the Q block [64 × 192] once into shared memory with `cp.async`. With d = 192 there is **no
   padding**: 192 = 24 × k8 TF32 MMA steps (or 12 × k16 for BF16).
2. **Pass 1 (row stats of S):** stream K in [64 × 192] tiles (double-buffered, 2 × 48 KB). Compute S with
   `mma.sync.m16n8k8.tf32`; each warp owns 16 rows; keep the row max and sum (m₁, l₁) in registers, using
   warp shuffles for the reduction across the 4 lanes that share a row.
3. **Pass 2:** re-stream K and V tiles.
   - Recompute S.
   - P₁ = exp(S − m₁)/l₁ in registers.
   - Load dist/ang tiles [64 × 64] through L2 (these 1 MB matrices are shared by all 16 heads, so L2 hit
     rate is high), compute the priors in registers, Z = αP₁ + βD + γA, and e = exp(Z − c) with
     c = |α|+|β|+|γ|. **No second online max is needed** (exact bound, see the math doc).
   - Accumulate O += e·V with MMA and l₂ += rowsum(e).
4. Epilogue: O /= l₂; store O plus (m₁, l₁, l₂) as 3 × 4 B per row.

HBM traffic: Q, O once; K twice, V once per CTA row-block (8 row blocks per head share K/V through L2).
About 27 MB of compulsory traffic; FLOPs ≈ 4.8 GFLOP (S computed twice).
*Estimate:* 0.1–0.2 ms on A100 TF32, versus 0.33 ms eager and 0.21 ms Inductor.

## Backward (the harder half)

δ₁ needs a full pass over keys before dS is known (unlike FlashAttention), so:

| Kernel | Parallelism | Work | Output |
|---|---|---|---|
| B1 | (head, query block) | recompute S, P₁, P₂, dP₂ = dO·Vᵀ; δ₂ = rowsum(dO∘O) precomputed | δ₁ = α·Σ P₁∘dZ, per-CTA partials of dα, dβ, dγ |
| B2 | (head, key block) | recompute tiles; dS = P₁∘(α dZ − δ₁) | dK, dV in registers (no atomics) |
| — | — | dQ accumulated in B2 with fp32 `atomicAdd` into a [N × d] buffer, **or** a separate row-parallel B3 (deterministic) | dQ |

The Triton prototype used 3 kernels with a separate dQ pass. The CUDA version should merge dQ into B2.
Use atomics, or a 2-stage reduction for bitwise determinism: one dQ partial per key block, then a reduce.

dα/dβ/dγ: block-level reduction, then one `atomicAdd` per CTA (or a deterministic second pass over the
128 partials).

## Numerical design (driven by the measured defect)

- **Saturated rows:** in layer 0, P₁ is one-hot with score ranges up to 3,800, so the true dS is O(ε)
  and must be formed exactly like eager to keep the residual small:
  - Compute dS from the same P₁ values that formed δ₁, in the same summation order within a row tile.
  - Keep m₁/l₁ in FP32, and recompute P₁ with `expf` (not `__expf`) for parity with eager.
  - Add a regression test on the captured real layer-0/7 activations
    (`benchmarks/attention_study.capture_real_qkv`) with an *absolute* bound tied to eager's error.
- **TF32 operand rounding:** convert operands with `cvt.rna.tf32.f32` (round-to-nearest) to match cuBLAS
  accuracy. The Triton prototype's TF32 error is about 2.5× eager's.
- Optional `3xTF32` split for the two S GEMMs if FP32-class scores are required. Cost: 3× MMA on a small
  GEMM.

## Integration and test plan

1. `torch.utils.cpp_extension` module (`foreswe_geo_attn`) built against the `pytorch/2.6.0` module's
   CUDA 12.4; a `torch.library.custom_op` wrapper so `torch.compile` treats it as an opaque op and CUDA
   graphs capture it.
2. Reuse the existing tests: FP64 `gradcheck` of the reference; fused vs FP64 within 10× eager error over
   4 shapes; real-activation regression; full-model gradients vs eager (TF32); G3 capture/replay
   bitwise-stable.
3. Benchmark with `benchmarks/attention_study.py` (isolated) and `graph_bench --attn fused`
   (end-to-end, G0 and G3/G4).
4. Profile with Nsight Compute: tensor-pipe utilisation, shared-memory bank conflicts, achieved occupancy,
   DRAM vs L2 throughput.

## Expected payoff (honest)

Measured ceiling: the attention core is 14 % of the E4 step and 18 % of G4. A kernel at the *estimated*
0.25–0.35 ms fwd+bwd would save about 4.5–5 ms per step, so G4 would go from 40.0 to about 35 ms
(≈ 1.13–1.15×). This is a meaningful demonstration of CUDA kernel engineering, but not the largest
remaining speedup for this model. The host data path (19 % of G4) is larger and simpler.
