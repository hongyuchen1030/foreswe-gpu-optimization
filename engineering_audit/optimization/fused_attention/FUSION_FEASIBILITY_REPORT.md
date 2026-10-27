# Fused geo-biased attention: feasibility, prototype results and recommendation

All numbers are **newly measured on 2026-10-26** on an A100-SXM4-40GB (nodes nid001556, nid001409,
nid002177) with PyTorch 2.6.0 and Triton 3.2.0, unless labelled as estimates. Data:
`benchmark_results/attention_study.json`; end-to-end runs under
`$FSW_SCRATCH_ROOT/artifacts/fused_attn/`. Mathematics: `ATTENTION_MATH_AND_GRADIENTS.md`.

## 1. How much time does the attention core take now? (re-profiled)

Isolated core, production shape Q/K/V [1, 16, 512, 192], **real** layer-0 activations from the trained
Plan B checkpoint, 10 warm-up + 50 measured calls, CUDA events:

| Implementation | TF32 forward | TF32 fwd+bwd | FP32 forward | FP32 fwd+bwd | Saved for backward | Peak extra (fwd+bwd) | Kernels fwd / fwd+bwd |
|---|---:|---:|---:|---:|---:|---:|---:|
| eager (production) | 0.331 ms | 0.887 ms | 0.577 ms | 1.655 ms | 34.0 MiB | 76.0 MiB | 16 / 40 |
| torch.compile (Inductor) | **0.212 ms** | 0.901 ms | 0.448 ms | 1.362 ms | 32.1 MiB | 58.1 MiB | 5 / 21 |
| fused Triton prototype | 0.866 ms | 3.043 ms | 2.644 ms | 26.45 ms | **2.1 MiB** | **8.2 MiB** | 7 / 17 |

**Share of the training step (Amdahl).** Eight layers × fwd+bwd:

| Step configuration | Core time per step | Step time | Core share | Ceiling if the core took 0 ms |
|---|---:|---:|---:|---:|
| G0 E4 eager | 8 × 0.887 = 7.1 ms | 50.78 ms | 14.0 % | 1.16× |
| G4 compile + graph | 8 × 0.901 = 7.2 ms | 40.05 ms | 18.0 % | 1.22× |
| historical FP32 (`GPU_PROFILING_REPORT.md`) | 12.3 ms | 148.2 ms | 8.3 % | 1.09× |

The isolated times approximate the in-model cost; the bounds are **theoretical ceilings, not measurements**.
Even an infinitely fast attention kernel would make G4 at most 1.22× faster. A realistic, well-engineered
kernel at about 2× eager speed (**estimate**) would save about 3.6 ms per step, so G4 would drop from
40 to about 36.5 ms (1.10×).

## 2. What torch.compile already does (Phase 7)

Generated kernels for the compiled core (TF32, forward):
`cutlass_80_tensorop_s1688gemm_256x64` (QKᵀ), `triton_red_fused__softmax_2` (row stats of S),
`triton_per_fused__softmax_0`, `triton_poi_fused__softmax_add_mul_reciprocal_1` (softmax #1 normalisation
fused with **both prior transforms 1/(1+x), the α/β/γ mix and softmax #2**), and
`cutlass_80_tensorop_s1688gemm_64x64` (P₂V). The backward fuses both softmax-backwards with the
dα/dβ/dγ reductions (`triton_per_fused__softmax__softmax_backward_data_div_mul_sum_0`, …).

| | eager | Inductor |
|---|---:|---:|
| forward kernels | 16 | 5 |
| fwd+bwd kernels | 40 | 21 |
| forward time | 0.331 ms | 0.212 ms (−36 %) |
| fwd+bwd time | 0.887 ms | 0.901 ms (no gain) |
| saved for backward | 34 MiB | 32 MiB |

**What remains unfused:** Inductor still materialises the [16, 512, 512] score and probability tensors in
HBM (≈ 32 MiB saved), keeps the two GEMMs as separate library calls, and gains nothing in the backward,
which is dominated by four GEMM-shaped products (dP₂ = dO·Vᵀ, dV, dQ, dK). A custom kernel's only real
advantage is avoiding the N×N materialisation and the extra HBM passes.

## 3. Prototype (Phase 8): fused two-pass forward + recomputing backward in Triton

`foreswe_bench/geo_attention.py` (`FusedGeoAttention`, autograd-integrated; eager operator kept as fallback
and oracle; `patch_model(model, "fused" | "eager")`; `graph_bench --attn fused`).

- **Forward:** grid (N/BM, B·H).
  - Pass 1 over key blocks gives the row max and sum (m₁, l₁) of S.
  - Pass 2 recomputes S, forms P₁, computes the priors **on the fly** from dist/ang tiles, mixes α/β/γ,
    applies exp(Z − c) with the exact bound c = |α|+|β|+|γ| (no second online max), and accumulates P₂V
    and l₂.
  - Saves only O and three row statistics. No N×N tensor ever touches HBM.
- **Backward:** three kernels that recompute tiles from (m₁, l₁, l₂):
  - B1 (rows): δ₁ and dα/dβ/dγ partials.
  - B2 (rows): dQ.
  - B3 (key columns): dK, dV.
  - δ₂ = rowsum(dO∘O).
- α, β, γ are read from device memory (no host sync), so the kernel is CUDA-graph capturable. Verified:
  G3 + fused captures and replays.

### Correctness (Phase 9)

| Check | Result |
|---|---|
| `torch.autograd.gradcheck` (float64) of the reference operator incl. α, β, γ | **pass** |
| Random inputs, 4 shapes (N = 64, 100, 129, 512; D = 64, 192, 48), FP32 vs FP64 | output 1.7–4.1e-7 (eager 1.5–4.0e-7); dQ 3.1–6.7e-7 (eager 3.2–6.7e-7); dK/dV and dα/dβ/dγ at the same level as eager |
| Same, TF32 vs FP64 | output 7.1–8.2e-4 (eager TF32 2.6–3.2e-4): **about 2.5× eager's TF32 error**, consistent with Triton's TF32 operand conversion (truncation vs cuBLAS rounding; not verified at instruction level) |
| Real layer 0 / 7 activations, FP32 output | 3.4e-7 / 3.4e-7 (eager 3.3e-7 / 3.3e-7) |
| Full Plan B model, TF32, fused vs eager attention | output rel-L2 7.9e-5, gradient rel-L2 9.8e-4 (cosine 0.9999995), the same magnitude as TF32's own error vs FP32 |
| GPU tests (`tests/test_gpu_graphs.py`) | fused FP32 within 10× of eager FP32 error vs FP64 on 3 shapes: pass |

**Open accuracy defect (real data, backward).** On the trained model, softmax #1 is **degenerate**:
- in layer 0 the per-row score range is 900–3,800, so P₁ is one-hot in 99.8 % of rows;
- in layer 7 all scores in a row are equal, so P₁ is exactly uniform.

The true dQ and dK are therefore almost zero: ‖dQ‖ = 0.089 (layer 0) and about 1e-12 (layer 7), against
‖dV‖ ≈ 57. On layer 0 the fused kernel's dQ error is 2.9e-4 absolute (rel 0.135), where eager's is 1e-6
(rel 7.5e-5). dV and the outputs are accurate. In layer 7 even eager FP32 has relative dQ error 0.24, because
there is no gradient left to resolve.

I tested the hypothesis that the δ₂ shortcut causes this: an extra pass computing δ₂ = ΣP₂·dP₂ from the
recomputed tiles did **not** change the layer-0 error (`CFG["delta2"]="recompute"`), so it is falsified.
The remaining suspects are the online-softmax statistics (m₁, l₁) on extreme-range rows and the recomputed
P₁ at saturation; they are not yet isolated. The defect is real but numerically small against the
gradients that matter; it must be fixed before any production use.

> **Update (resolved).** The defect was isolated and fixed in `foreswe_bench/geo_attention_v2.py`. The first
> divergence is the backward's recomputed P₁: FMA contraction of `acc/√d − m₁` left the peak at 1 + 2.4e-4 on
> saturated rows. The δ₂ shortcut explains the scalar-gradient error, not dQ, which is consistent with the test
> above. FP32 dQ/dK error is now at parity with eager on real layers 0 and 7. See
> [SATURATED_SOFTMAX_BACKWARD_DEBUG.md](SATURATED_SOFTMAX_BACKWARD_DEBUG.md) and
> [KERNEL_TUNING_FINAL_REPORT.md](KERNEL_TUNING_FINAL_REPORT.md). The original prototype is unchanged.

### Performance

The tile sweep at the production shape (TF32, fwd / fwd+bwd ms) shows the design is shared-memory bound:

| BM × BN, warps | 32×32, 4 | 32×32, 8 | 16×32, 4 | 16×16, 4 | 32×16, 4 | 64×32, 8 | 64×64, 8 |
|---|---|---|---|---|---|---|---|
| fwd / fwd+bwd | 1.04 / 11.25 | 1.54 / 11.84 | **0.77 / 2.98** | 0.95 / 3.02 | 1.11 / 22.30 | out of shared memory | out of shared memory |

The best config (16×32, 4 warps, 1 stage at D=192) reaches 0.87 ms forward. That is 5.6 TFLOP/s and
31 GB/s on the minimum traffic of 27 MB, far from the A100's 156 TFLOP/s (TF32) and 1.5 TB/s.

**End-to-end training step (TF32):**

| | eager attention | fused prototype | Δ |
|---|---:|---:|---:|
| G0 (E4 eager) | 50.85 ms | 66.92 ms | **+16.1 ms (slower)** |
| G3 (manual graph) | 44.99 ms | 61.86 ms | +16.9 ms |
| G0 peak allocated | 8.51 GiB | 8.31 GiB | −0.2 GiB |

The prototype is correct in FP32 on the forward and on dV, uses 9× less memory, and is graph-capturable.
It is **3.4× slower** than eager in fwd+bwd and makes the training step 32 % slower, so it stays opt-in.
Production uses the eager operator.

Why it is slow:
- **D = 192 pads to a 256-wide tile.** That wastes 25 % of every tile and pushes BM×BN×D tiles past the 163 KB
  shared-memory limit, forcing 16-row tiles and a single pipeline stage.
- **The backward recomputes S and dP₂ three times** (B1, B2, B3), on top of the two forward passes.
- **N = 512 is small:** the work fits comfortably in L2, so avoiding HBM materialisation is worth less
  than at long sequence lengths.

## 4. Options (Phase 8 decision)

| Option | Expected outcome on A100 | Engineering cost | Verdict |
|---|---|---|---|
| **C. Inductor** (already in G4) | forward −36 %, fwd+bwd ≈ eager; zero code | none | **keep now** |
| **A. Triton** (this prototype, tuned) | needs: split D = 192 into 3 × 64 slices, a single fused backward with atomic dQ, separate tile configs per kernel, autotuning. **Estimated** best case ≈ 0.45 ms fwd+bwd (2× eager) → 1.10× on G4 | medium (days) | worthwhile only as a learning/portfolio exercise; small model-level payoff |
| **B. CUDA C++ / CUTLASS** | same algorithm with explicit `mma.sync` TF32/BF16 fragments, warp-specialised loads (`cp.async`), D = 192 as 12 × 16 k-steps (no padding), register-resident row statistics; could approach the **memory-traffic bound** (~0.1–0.2 ms fwd, **estimate**) | high (weeks): forward + 3-stage backward + autograd binding + tests | the strongest *skill-building* project, but the payoff is capped by Amdahl at ≤ 1.22× for this model |

**Recommendation.** Do not ship a custom attention kernel for ForeSWE yet: its ceiling is ≤ 1.22× and
the measured Inductor/graph path already captures the cheap wins. The larger untouched target is the
**host data path** (6.9–8.1 ms, 19 % of G4), which needs no kernel engineering. If the goal is to build
CUDA/C++ skill on a real operator, the geo-attention kernel is a good, well-specified exercise; see
`KERNEL_DESIGN.md`. Use the eager operator and these tests as the oracle, and fix the saturated-softmax
backward defect first.
