# Fused Geo-Attention Kernel Tuning: Final Report

Scope: the existing Triton geo-biased attention operator only. The model, the attention math, the geometry
(including the Kahan angle formula), the training and the checkpoints are unchanged. Details:
[KERNEL_TILING_OPTIMIZATION.md](KERNEL_TILING_OPTIMIZATION.md) (task 1) and
[SATURATED_SOFTMAX_BACKWARD_DEBUG.md](SATURATED_SOFTMAX_BACKWARD_DEBUG.md) (task 2).

Setup: one A100-SXM4-40GB (Perlmutter), PyTorch 2.6.0, Triton 3.2. Production shape Q/K/V [1, 16, 512, 192],
real activations of the trained Plan B checkpoint. One cached geometry matrix set is used for every
implementation; the cache predates the Kahan angle change, and baseline and candidates read the same file.
Timings are CUDA-event medians; all isolated-core numbers come from one process
(`benchmark_results/final/`).

## Code

| File | Status |
|---|---|
| `foreswe_bench/geo_attention.py` | original v1 prototype, **unchanged** (baseline) |
| `foreswe_bench/geo_attention_v2.py` | chunked-D kernel with per-kernel tiles and numerical presets `accurate_experimental` (default), `accurate_x3_experimental` and `fast_experimental` |
| `foreswe_bench/attn_check.py` | the fixed acceptance rule and synthetic regimes |
| `tests/test_gpu_geo_attention_v2.py` | regression tests (56 passed, 1 xfail with a recorded reason) |
| `benchmarks/attention_tiling.py`, `attention_backward_debug.py`, `attention_final.py`, `attention_resources.py` | sweep, intermediate diagnosis, final comparison, per-kernel resources |
| `benchmarks/graph_bench.py --attn fused_v2_accurate/fused_v2_fast` | full-model integration |

## Final comparison: isolated attention core, real layer 0

Errors are max-abs against FP64 with the same dO. "Peak memory" is extra peak allocation for forward+backward,
above inputs and outputs.

### TF32 (production precision, E4)

| Implementation | Fwd ms | Bwd ms | Fwd+Bwd ms | dQ error | dK error | Peak memory | Passes L0/L7 rule |
|---|---|---|---|---|---|---|---|
| PyTorch eager | 0.332 | 0.564 | 0.857 | 3.55e-4 | 1.81e-4 | 76.0 MiB | reference |
| Inductor | **0.209** | **0.489** | **0.779** | 3.55e-4 | 1.81e-4 | 58.1 MiB | yes / yes |
| Original Triton (v1) | 0.863 | 2.359 | 3.007 | 5.32e-4 | 4.05e-4 | 8.2 MiB | **no / no** (output 15.8×, dγ 1068×) |
| Tuned tiling (v2 fast) | 0.445 | 1.311 | 1.528 | 5.32e-4 | 4.05e-4 | 8.2 MiB | **no / no** |
| **Tuned + accuracy fix (v2 accurate)** | 0.491 | 1.930 | 2.191 | 3.54e-4 | 1.80e-4 | 8.2 MiB | yes / yes |
| Tuned + fix + 3×TF32 scores | 0.692 | 2.535 | 2.995 | **6.93e-5** | **3.60e-5** | 8.2 MiB | yes / yes |

### FP32 (`allow_tf32=False`)

| Implementation | Fwd ms | Bwd ms | Fwd+Bwd ms | dQ error | dK error | Peak memory | Passes L0/L7 rule |
|---|---|---|---|---|---|---|---|
| PyTorch eager | 0.582 | 1.144 | 1.662 | 2.39e-6 | 1.26e-6 | 76.0 MiB | reference |
| Inductor | 0.450 | 0.985 | 1.345 | 2.32e-6 | 1.32e-6 | 58.1 MiB | yes / yes |
| Original Triton (v1) | 2.630 | 23.880 | 26.224 | 3.14e-4 | 7.32e-4 | 8.2 MiB | **no** / yes |
| Tuned tiling (v2 fast) | 1.459 | 16.902 | 17.988 | 3.14e-4 | 7.32e-4 | 8.2 MiB | **no** / yes |
| **Tuned + accuracy fix (v2 accurate)** | 1.458 | 16.814 | 17.958 | 2.39e-6 | 1.35e-6 | 8.2 MiB | yes / yes |

Triton's FP32 `tl.dot` ("ieee") runs on FFMA, not tensor cores. The FP32 timings are shown for the numerical
comparison, not as a target.

Kernels launched per forward+backward: eager 40, Inductor 20, v1 17, v2 fast 19, v2 accurate 17.

## CUDA Graph integration (full model, E4 TF32, real data)

The accurate variant passes the correctness requirements and is the integration candidate. The fast variant
was run for reference only.

- **Capture/replay:** `test_cuda_graph_replay_matches_eager_launch_and_refreshes_inputs` captures the accurate
  forward+backward and copies two different inputs into the static buffers. Replay matches direct launches
  **bitwise** for the output and all six gradients, so gradients are fresh on every replay.
- **Full step:** `graph_bench` captures the whole training step (G3 manual graph, G4 Inductor + graph). The
  kernels capture without error, the loss stays finite, and the loss trajectory matches eager attention
  (first/last loss 208831.7 / 211162.1 for both in G0; G3 and G4 agree to within 1e-5 relative).

Bounded benchmark: 10 warm-up and 40 timed steps; A = GPU region on staged inputs, B = end-to-end step.

| Config | Attention | A: GPU region ms | B: end-to-end ms | Inference ms | Peak GiB |
|---|---|---|---|---|---|
| G0 (E4 eager) | eager | 41.04 | 51.43 | 11.92 | 8.51 |
| G0 | v2 accurate | 51.22 | 61.10 | 12.02 | 8.31 |
| G0 | v2 fast (reference) | 46.07 | 56.12 | 11.71 | 8.31 |
| G3 (CUDA graph) | eager | 36.93 | 44.86 | 8.86 | 7.92 |
| G3 | v2 accurate | 47.47 | 55.80 | 9.02 | 7.92 |
| G3 | v2 fast (reference) | 41.73 | 49.30 | 8.62 | 7.92 |
| **G4 (compile + graph)** | **eager** | **32.77** | **41.14** | 8.37 | 7.92 |
| G4 | v2 accurate | 45.91 | 54.04 | 11.28 | 7.93 |
| G4 | v2 fast (reference) | 40.13 | 48.30 | 12.67 | 7.93 |
| *(earlier run)* G3 | v1 original | 53.73 | 61.86 | — | 7.92 |

In G0, the phase split is forward 11.7 → 11.9 ms, **backward 19.7 → 29.9 ms**, optimizer 9.5 ms
(`benchmark_results/graphs/`). The 10 ms gap is consistent with 8 attention layers × the isolated backward gap
(1.93 vs 0.56 ms). In G4 the gap is larger, because Inductor also fuses eager attention's surrounding ops,
while the Triton op is opaque to it. The 0.2 GiB peak-memory saving in G0 does not appear in G3 or G4
(7.92 GiB with either attention).

Decision: **the production attention stays eager**. The accurate fused kernel is correct and graph-safe, but it
makes the G4 step 1.40× slower (45.9 vs 32.8 ms).

## Answers

1. **Did reducing D = 192 padding improve performance?** Yes. Fwd+bwd went from 3.007 ms (v1) to 1.528 ms (v2
   with the same numerics) in TF32: 1.97× (fwd 1.94×, bwd 1.80×). About 1.66× of that comes from chunking alone
   (1.87 ms at v1's tile shape). Tile tuning adds about 1.2×.
2. **Which tile configuration is now fastest?** Forward (BM 32, BN 64, 4 warps, 2 stages). Row backward
   (32, 32, 4 warps, 1 stage). Column backward (32, 32, 8 warps, 2 stages). Most of the 64-config sweep lies
   within about 15 % of the best. 14 configurations exceeded the 163 KB shared-memory limit.
3. **What was the first numerical divergence in the backward?** The backward's recomputed P₁. On saturated rows
   its peak was 1 + 2.38e-4 instead of 1 (Σ P₁ − 1 = 2.38e-4, against 7.9e-8 for eager). S, m₁, l₁ and O all
   matched eager.
4. **Why was dQ/dK less accurate under saturated attention?** On a one-hot row the true dS = α dZ P₁(1 − P₁) ≈ 0
   is a difference of two O(α dZ) terms. The compiler contracted `acc/√d − m₁` into one FMA, so the recomputed
   peak score was not the forward's rounded S; it was off by up to half an ulp of |S| ≈ 4,878. That left P₁ = 1 +
   2.4e-4, and the cancellation failed by 2.4e-4 · α · dZ ≈ 7.6e-5 per row, 20,000× eager's dS error. dQ and dK
   are dS·K and dSᵀ·Q, so they inherit it. Their true values are tiny (‖dQ‖ = 0.18), which makes the error
   visible.
5. **What exact code change improved the accuracy?** (a) `_scores3`: `tl.math.div_rn(acc, sqrt_dk)` (correctly
   rounded, not contractible), used in forward and backward, so the recomputed S is bitwise the forward's.
   (b) A B0 row pass δ₂ = Σ P₂·dP₂ in place of rowsum(dO∘O). (c) FP64 accumulation of the dα/dβ/dγ partials.
   (d) TF32 only: `_rn_tf32` round-to-nearest operands before every `tl.dot`. All four are in
   `PRESETS["accurate_experimental"]`.
6. **How much did absolute dQ/dK error decrease?** FP32, real layer 0: dQ 3.14e-4 → 2.39e-6 (131×) and dK
   7.32e-4 → 1.35e-6 (542×), now 1.00× and 1.08× eager. TF32: dQ 5.32e-4 → 3.54e-4 and dK 4.05e-4 → 1.80e-4
   (1.5× and 2.2× smaller, now equal to eager TF32). With 3×TF32 scores they reach 6.9e-5 and 3.6e-5, 5× better
   than eager TF32. The TF32 output (15.8× → 0.93×) and dγ (1068× → 0.55×) improved the most.
7. **Did the accuracy correction affect kernel latency?** Yes, in TF32: fwd+bwd 1.528 → 2.191 ms (+43 %). About
   +0.27 ms is the B0 pass and the rounded division; about +0.39 ms is TF32 operand rounding. In FP32 there is no
   measurable cost (17.99 vs 17.96 ms). Accurate v2 is still 1.37× faster than v1 (3.007 ms).
8. **How much memory does the optimized kernel use?** 8.2 MiB extra peak for forward+backward (row statistics
   and partial sums), against 76.0 MiB for eager and 58.1 MiB for Inductor: 9.3× and 7.1× less. Per-kernel
   resources: 255 registers/thread (spills 8–66), 56–144 KB shared memory, 6.25–12.5 % theoretical occupancy.
9. **Does it pass the real-model regression tests?** Yes. Real layers 0 and 7 pass in FP32 and TF32 under the
   unchanged 10× rule, and all 42/42 synthetic checks pass in each precision. The full suite gives 56 passed and
   1 recorded xfail (a seed where eager's dβ error is a low outlier; a six-seed sweep on that shape passes).
10. **Is it faster than eager or Inductor?** No. Isolated TF32 fwd+bwd: 2.19 ms, against 0.86 ms eager and
    0.78 ms Inductor. The forward is within 1.5× of eager (0.49 vs 0.33 ms); the backward is 3.4× slower. In
    the full model, the G4 step is 45.9 ms against 32.8 ms with eager attention. The kernel wins only on memory.
11. **Is it safe to integrate into training, or must it remain experimental?** It is numerically safe:
    accuracy matches eager on real data and graph replay is correct. It is not beneficial, being slower in every
    configuration. It stays experimental and opt-in (`--attn fused_v2_accurate`), and production attention
    remains eager. The fast preset must not be used for training, because it fails the real-data rule.
12. **What measured GPU kernel engineering result can be stated accurately on a resume?** For example:
    *"Optimized a fused Triton geo-biased two-softmax attention kernel (forward + 3-pass backward) on A100:
    removing head-dimension padding and tuning per-kernel tiles gave a 1.97× forward+backward speed-up over my
    prototype (3.01 → 1.53 ms) at 9× lower activation memory than PyTorch eager (8 MiB vs 76 MiB). Traced a
    saturated-softmax backward error to FMA contraction in score recomputation and to biased TF32 operand
    truncation; fixed both, cutting FP32 dQ/dK error 131×/542× to parity with eager, with FP64 regression tests
    on real model activations and CUDA-graph-safe replay."* Do not claim it beats cuBLAS/eager or Inductor
    attention, or that it sped up ForeSWE training: it did not.

## Reproduce

```bash
JOB=$(bash scripts/slurm/alloc.sh fsw_kern)
bash scripts/slurm/run.sh $JOB python -m pytest -q tests
bash scripts/slurm/run.sh $JOB python -m benchmarks.attention_tiling
bash scripts/slurm/run.sh $JOB python -m benchmarks.attention_final
bash scripts/slurm/run.sh $JOB python -m benchmarks.attention_resources
bash scripts/slurm/run.sh $JOB python -m benchmarks.attention_backward_debug --preset accurate_experimental --prec fp32
for cfg in G0 G3 G4; do for at in eager fused_v2_accurate fused_v2_fast; do
  bash scripts/slurm/run.sh $JOB python -m benchmarks.graph_bench --cfg $cfg --attn $at --data real \
    --out engineering_audit/optimization/fused_attention/benchmark_results/graphs/${cfg}_${at}.json
done; done
scancel $JOB
```
