# Numerical error analysis: FP32 vs TF32 vs BF16 (Plan B ForeSWE)

Newly measured 2026-10-26 on A100-SXM4-40GB (node 1, job 60335641, nid003640), PyTorch 2.6.0.
Data: `benchmark_results/numerics_precision/` and `benchmark_results/error_origin/`.

## Method

- **Inputs:** 9 real ForeSWE issue dates from the cached production loader (`build_data_cache`, checked
  against the historical Plan B run: tensor shapes, 3297 training dates, Feb-1 indices). 4 training dates
  spread over 1991–2014 plus the 5 Feb-1 test dates (2015–2019), 512 stations each.
- **Weights:** (a) seed-0 initialisation, (b) the trained Plan B checkpoint (epoch 7).
- **Graph:** `model.eval()` (dropout off) so every precision computes the identical graph. Full backward
  pass, no optimizer update.
- **Precisions:** FP32 = TF32 off and `float32_matmul_precision("highest")` (the main baseline). TF32 =
  `allow_tf32=True`. BF16 = `torch.autocast(bfloat16)` around forward + loss, with TF32 off, FP32 master
  weights, FP32 gradients.
- **Exact reference:** an FP64 copy of the same weights and inputs. One production line,
  `torch.stack(embeds).to(dtype=torch.float32)` in `CrossVarTransformer.forward`, forces one FP32 rounding
  inside the FP64 model, so the FP64 reference is FP64 except at that cast.
- Gradient metrics use only the 426.0 M **live** parameters (the 201.6 M never-called
  `CrossVarTransformer.encoder_layers` parameters are excluded). All reductions are in float64.

## Phase 1 results (median over 9 dates; max in brackets)

| Weights | Comparison | Output rel-L2 | Output max \|Δ\| (mm SWE) | Loss rel. Δ | Gradient rel-L2 | Grad. cosine (min) | NaN/Inf |
|---|---|---:|---:|---:|---:|---:|---:|
| trained | TF32 vs FP32 | **3.9e-4** [4.6e-4] | 0.70 | 1.1e-5 | **2.2e-4** [5.3e-4] | 0.99999987 | 0 |
| trained | BF16 vs FP32 | **5.7e-3** [6.4e-3] | 10.2 | 2.9e-4 | **3.6e-3** [4.7e-3] | 0.9999891 | 0 |
| trained | FP32 vs FP64 | 3.0e-7 | 0.0006 | 2.0e-8 | 2.0e-7 | 1.0 | 0 |
| init | TF32 vs FP32 | 7.2e-5 | 0.00005 | ≈0 | 8.1e-4 | 0.99999962 | 0 |
| init | BF16 vs FP32 | 2.4e-3 | 0.0019 | 9.6e-8 | 6.4e-3 | 0.9999795 | 0 |
| init | FP32 vs FP64 | 1.3e-7 | 1e-7 | 4.5e-8 | 1.5e-5 | 1.0 | 0 |

Reading the table:
- **FP32 is a good stand-in for exact arithmetic.** Its error vs FP64 (1e-7 to 3e-7) is 3–4 orders of
  magnitude below TF32/BF16, so "error vs FP32" and "error vs FP64" agree to all printed digits.
- **TF32 ≈ 4e-4 and BF16 ≈ 6e-3 relative output error** on trained weights. For context, SWE outputs reach
  ~2,000 mm: the worst BF16 deviation is 10 mm at one station, while the model's test RMSE is ~110–220 mm
  (`experiments/planB_24week/planB_pred_vs_actual_24weeks_full512_summary.csv`). The precision error is
  ~20× smaller than the forecast error.
- **No NaN/Inf in any output or gradient**; gradient cosines are ≥ 0.99998.
- Error is higher on trained than on initial weights for outputs (larger, more structured activations) and
  lower for gradients.

**Layer-wise gradient error (BF16 vs FP32, trained weights, one date):** it grows toward the input as
backprop accumulates rounding: encoder layer 7 ≈ 0.2–0.5 %, layer 0 ≈ 1.3–1.4 %; output MLP 0.6 %;
cross-variable branches 0.3–0.4 %.

**Activation dtypes under BF16 autocast:** Linear, GELU, Dropout, `nn.MultiheadAttention`, the custom
geo-attention, ReLU and the feed-forward run in **bf16**. LayerNorm, the EncoderLayer residual output and the
MSE loss run in **fp32**. The model output is **bf16**.

## Phase 5: where the error originates

### Along the forward pass (error vs FP64; trained weights, Feb-1-2017)

| Point | FP32 | TF32 | BF16 |
|---|---:|---:|---:|
| cross-variable branch outputs | 2–3e-7 | 2e-4 – 1.0e-3 | 3.6e-3 – 8.7e-3 |
| encoder L0 `W_q` | 7e-7 | 3.5e-4 | 4.2e-3 |
| encoder L0 output | 5.7e-7 | 1.7e-4 | 2.4e-3 |
| encoder L7 output | 3.8e-7 | 6.2e-5 | 1.8e-3 |
| output MLP | 2.8e-7 | 3.5e-4 | 4.8e-3 |
| **model output** | 2.9e-7 | **3.6e-4** | **5.9e-3** |
| loss | 8e-8 | 1.1e-5 | 4.5e-4 |

Error enters at the first GEMMs and **does not compound** through the 8 encoder layers (L7 is no worse than
L0: LayerNorm and residuals keep it bounded). The largest single contributions are the cross-variable
input branches and the narrow output MLP (3072 → 1024 → 128 → 8 → 24), where few accumulation terms
average the rounding less.

### Operation-level isolation (encoder L0 `W_q`, K = 3072, on captured FP64 activations)

| Computation of the same GEMM | rel-L2 vs FP64 |
|---|---:|
| FP32 SIMT (TF32 off) | 6.6e-7 |
| TF32 Tensor Cores (native) | 1.74e-4 |
| inputs rounded to TF32, exact FP64 accumulation | **1.74e-4** |
| BF16 autocast (native: bf16 in, fp32 accumulate, **bf16 out**) | 2.17e-3 |
| inputs rounded to BF16, exact FP64 accumulation | **1.40e-3** |
| bf16 inputs, fp32 accumulate, **fp32 out** (emulated) | 1.40e-3 |
| **bf16x3 split** (hi·hi + hi·lo + lo·hi), fp32 out (emulated) | **2.6e-6** |
| **tf32x3 split** (3 TF32 GEMMs, unfused) | **5.7e-6** |

FC2 (K = 2048) gives the same picture (TF32 1.6e-4, BF16 2.1e-3, tf32x3 5.4e-6). Conclusions:

1. **The error is almost entirely input (operand) rounding, not accumulation.** Native TF32 equals
   "TF32-rounded inputs, exact accumulation" to 3 digits. For BF16, ~65 % of the error (1.40e-3 of 2.17e-3)
   is operand rounding; the rest is the **bf16 rounding of the GEMM output** (PyTorch returns bf16 from a
   bf16 matmul even though the Tensor Core accumulated in fp32).
2. **Compensated accumulation (Kahan / TwoSum on the accumulator) cannot help here.** The accumulator is
   already FP32 and contributes ~1e-6. EFT-style accumulation would fix an error that is not there.
3. **Operand splitting does recover the information**, because the low parts are representable: a
   two-term split (x = x_hi + x_lo, each in BF16 or TF32) brings the GEMM error from 1e-3 / 1e-4 down to
   **3–6e-6**, close to FP32 SIMT, using 3 Tensor-Core GEMMs. This is the "3xTF32" / "BF16x3" family of
   Ozaki/Dekker-style decompositions. It is the GPU analogue of SIMD double-word arithmetic: the precision
   is recovered by **splitting operands before rounding**, not by compensating a rounding that already
   happened. Emulated cost (unfused): 0.28 ms for tf32x3 vs 0.10 ms TF32 vs 0.60 ms FP32 SIMT, so
   **FP32-class accuracy at ~2× the speed of SIMT FP32 even before fusion**.
4. **Attention core:** QKᵀ scores in BF16 have 2.3e-3 error; the double softmax and α/β/γ geo-bias mix
   bring the core output to 1.6e-3 (TF32: 1.6e-5 in layer 0, 2.1e-4 in layer 7). Softmax itself is
   benign (1e-7 on exact scores); its error is inherited from the BF16 scores (4.8e-3 in L0, where
   attention is sharp; 1.6e-6 in L7).
5. **LayerNorm** in FP32 adds 7e-8; given BF16-rounded input it reports 1.6e-3, i.e. it transmits but does
   not create error.
6. **Reductions:** the FP32 MSE sum over 12,288 terms has 1.1e-7 relative error. Not a concern.
7. **Optimizer updates:** see `FUSED_ADAMW_REPORT.md`. Decoupled weight decay (`lr·wd = 5e-10`) is
   **entirely lost** in FP32 master weights (bit-identical with and without decay), and 28 % of first-step
   element updates fall below half an ulp. This is the one place in the pipeline where **compensated
   summation (Kahan master weights, or a separate FP32 error-carry buffer)** would change the computed
   result.

## Would selectively keeping FP32 help?

Measured discrepancies are small (TF32 4e-4, BF16 6e-3) and have no visible effect on short training (see
`PRECISION_PERFORMANCE_REPORT.md`), so no FP32 islands are required for correctness. If BF16 error had to
be cut, the data says to target, in order: (1) the cross-variable input branches and the output MLP (largest
per-module error, tiny share of FLOPs, so FP32/TF32 there is nearly free); (2) the bf16 output rounding of
GEMMs (request fp32 output via cuBLASLt); and (3) operand splitting on the encoder GEMMs.

## Ranked candidates for a compensated / custom kernel (identified, not implemented)

| # | Candidate | Measured error it addresses | Measured runtime context | Verdict |
|---|---|---|---|---|
| 1 | **Split-operand GEMM** (tf32x3 / bf16x3, fused in one CUTLASS/Triton kernel, fp32 output) | 1.7e-4 (TF32) / 1.4e-3 (BF16) → 3–6e-6 | GEMMs = 16.8 ms/step on TF32; unfused emulation already 2.1× faster than FP32 SIMT | **Strongest**: real accuracy recovery, Tensor-Core speed, classic EFT/splitting idea |
| 2 | **Compensated AdamW** (Kahan master weights / error-carry, fused) | lost weight decay (100 %), 28 % absorbed updates | optimizer = 9.5 ms fused, ~75 % of HBM roofline; +1 FP32 buffer (1.6 GiB) | Strong *numerical* case; costs bandwidth; changes training semantics, so needs a convergence study |
| 3 | Fused geo-biased attention (QKᵀ → softmax → α/β/γ mix → softmax → ·V) with FP32 scores | 1.6e-3 core error (BF16) | attention core ≈ 10 % of forward (FP32 era, `GPU_PROFILING_REPORT.md`) | Performance-driven Triton target; accuracy gain secondary |
| 4 | Compensated reductions (loss, LayerNorm stats) | ≤ 1e-7 | negligible | Not worth it |

**EFT caveat, measured:** applying TwoSum/Kahan *after* a BF16 rounding cannot restore the 1.4e-3 lost to
operand rounding (row "inputs rounded, exact accumulation" already includes perfect accumulation). The
information has to be kept by splitting the operand into hi + lo before it is rounded.
