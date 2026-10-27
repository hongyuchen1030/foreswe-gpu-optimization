# torch.compile and CUDA Graphs on the E4 training step

Plan B ForeSWE (627.6 M params, 426 M live), one issue date × 512 stations per step, TF32 + fused AdamW
(lr 5e-6, wd 1e-4), MSE loss. All numbers are **newly measured on 2026-10-26**. Hardware:
A100-SXM4-40GB on node A (job 60347489, nid001409) and node B (job 60347830, nid001556). Software:
NERSC `pytorch/2.6.0` (torch 2.6.0, CUDA 12.4, Triton 3.2.0), driver 580.178.04.
Historical references: E4 = 50.8 ms and original FP32 + foreach = 148.2 ms
(`../PRECISION_PERFORMANCE_REPORT.md`).

## 1. Graph compatibility of the E4 path (Phase 1)

| Item | Finding | Consequence |
|---|---|---|
| Control flow | `SWETransformer.forward` and `CrossVarTransformer.forward` have no data-dependent branches; the loop over 13 variables and over 8 encoder layers has fixed trip counts | the whole forward traces into one graph (verified: dynamo `unique_graphs = 1`, no graph breaks) |
| Shapes | static: daily (30, 512, 19), weekly (30, 512, 19), yearly (5, 512, 19), prompts (1, 512, 1536), dist/ang (512, 512), target (512, 24) | no dynamic-shape recompiles (verified: 0 recompiles) |
| Data prep (`get_example`) | Python-list indexing of the GPU-resident dataset (gathers the full 10305×512×19 tensor 3×), NumPy fancy indexing of the 512×512 distance and angularity matrices, H2D copies, `torch.tensor(t)` copies | **not capturable**; stays outside the graph and remains in every end-to-end number (scope B) |
| Changing inputs / addresses | every step allocates new input tensors | graph variants copy them into persistent `StaticBatch` buffers (`copy_`, 7 tensors ≈ 8 MB D2D) |
| Host syncs | `loss.item()` once per step (production logs it); no `.cpu()`, no NumPy inside the model | kept outside the graph: read from the static loss after replay (same value; production reads it before backward) |
| Gradients | `zero_grad()` defaults to `set_to_none=True` (free + reallocate every step) | grads set to None **before capture**, so backward allocates them from the graph pool; every replay overwrites them, so there is no accumulation and no stale gradient |
| AdamW | fused, `capturable=False` keeps `step` on the host | in-graph optimizer needs `capturable=True` (step and bias correction on device). The ungraphed control **G0c** shows this setting changes nothing: bitwise identical to G0 |
| LR schedule | production StepLR changes lr once per epoch; the benchmark runs within one epoch | a graphed step with a schedule would need a tensor lr (supported for fused/capturable AdamW in 2.6); not needed here |
| Dropout (p = 0.05, 0.1) | stochastic | PyTorch's graph-safe Philox advances the offset per replay. Verified: train-mode replays on the same input give different losses, eval-mode replays identical; G3 even reproduces G0's 300-step loss sequence to all printed digits |
| Dead parameters | 201.6 M `CrossVarTransformer.encoder_layers` params never receive grads | skipped by AdamW in every mode; no effect on capture |

Capturable regions: the full forward + loss + backward + fused AdamW step (**G3**). The host-side data
preparation is not capturable without changing the data path, which this study does not do.

## 2. Configurations

| ID | What | Notes |
|---|---|---|
| G0 | E4 eager | re-measured on both nodes |
| G0c | E4 eager with `AdamW(fused, capturable=True)` | control for the optimizer setting G3/G4 need |
| G1 / G1L | `torch.compile(mode="default")` of the whole model / of each encoder layer | optimizer stays fused eager |
| G2 / G2L | `torch.compile(mode="reduce-overhead")` (Inductor CUDA-graph trees) | `cudagraph_mark_step_begin()` each step |
| G3f / G3 | manual `torch.cuda.CUDAGraph`: forward + backward captured / full step incl. fused AdamW | side-stream warm-up, then weights and AdamW state restored (same storage) |
| **G4** | Inductor-compiled model (default mode, no Inductor cudagraphs) captured inside the manual full-step graph | combines kernel fusion (G1) with one-launch replay (G3) |

All configurations start from identical seed-0 weights and fresh AdamW state, use the same synthetic
production-shaped inputs and the same training-date sequence, and run in a fresh process (3 repeats,
rotated order, 10 warm-up + 40 measured steps; repeat 0 with a cold Inductor cache).

## 3. Results (node A; median of 120 steps, variability = range of the 3 repeat medians)

**Scope B (end-to-end training step: data prep + input copy + step + loss sync) and scope A (accelerated region on staged inputs):**

| Config | B: step ms | Range | Speedup vs G0 | vs E4 hist. 50.8 | vs FP32 148.2 | A: GPU region ms | Inference fwd ms | Peak alloc / reserved | Compile/capture cold / warm |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **G0** E4 eager | **50.78** | 50.7–51.4 | 1.00× | 1.00× | 2.92× | 40.96 | 11.92 | 8.51 / 9.07 GiB | – |
| G0c capturable AdamW | 51.31 | 49.3–51.7 | 0.99× | 0.99× | 2.89× | 41.02 | 11.94 | 8.51 / 9.07 | – |
| G1 compile default | 43.61 | 43.2–44.0 | 1.16× | 1.16× | 3.40× | 34.90 | 10.08 | 8.50 / 11.11 | 45.8 s / 14–17 s |
| G1L compile layers | 46.94 | 46.9–47.3 | 1.08× | 1.08× | 3.16× | 37.55 | 10.71 | 8.50 / 10.98 | 14.4 s / 5 s |
| G2 reduce-overhead | 41.09 | 40.9–41.4 | 1.24× | 1.24× | 3.61× | 33.00 | 9.43 | **6.31** / 10.60 | 47.0 s / 17–21 s |
| G2L reduce-overhead layers | 46.45 | 46.3–46.8 | 1.09× | 1.09× | 3.19× | 37.22 | 10.64 | 6.41 / 10.86 | 16.8 s / 7–8 s |
| G3f graph fwd+bwd | 45.08 | 43.8–45.3 | 1.13× | 1.13× | 3.29× | 37.00 | 8.79 | 7.92 / 11.11 | 0.5 s |
| G3 graph full step | 43.94 | 43.9–44.9 | 1.16× | 1.16× | 3.37× | 36.89 | 8.80 | 7.92 / 11.11 | **0.5 s** |
| **G4 compile + graph** | **40.05** | 39.9–40.3 | **1.27×** | **1.27×** | **3.70×** | **32.55** | **8.28** | 7.92 / 12.09 | 46.4 s / 15 s |

Node B (independent baseline on another A100-40GB): G0 50.55 ms, G3 44.56 ms (1.13×), consistent with node A.

Within-step IQR was ≤ 0.2 ms for every configuration. Data prep took 6.9–8.1 ms in every configuration;
it is unchanged by construction and is now **19 % of the G4 step**.

**Nsight Systems (per end-to-end step, `--cuda-graph-trace=node`; times are under tracing, hence longer):**

| Config | GPU busy | GPU idle in step | GPU kernels executed | Triton kernels | Host kernel-launch calls | Graph launches | GEMM ms (Tensor-Core) |
|---|---:|---:|---:|---:|---:|---:|---:|
| G0 | 64.1 % | 22.7 ms | 1,106 | 0 | 1,106 | 0 | 17.8 (16.7) |
| G1 | 76.9 % | 10.8 ms | 956 | 516 | 956 | 0 | 17.8 (16.6) |
| G2 | 83.2 % | 7.1 ms | 962 | 516 | 48 | 2 | 17.5 (16.4) |
| G3 | 85.6 % | 6.6 ms | 1,108 | 0 | 12 | 1 | 17.6 (16.6) |
| G4 | 86.9 % | 5.4 ms | 958 | 516 | 12 | 1 | 17.6 (16.5) |

Figures: `figures/step_times.png`, `figures/launches_and_busy.png`, `figures/timeline_G0_G3_G4.png`
(one step's GPU activity and host launch calls, before/after), `figures/convergence_epoch.png`.

### What each technique actually changed

- **CUDA Graphs (G3) remove launch overhead, not GPU work.** G3 executes the *same* 1,108 kernels as G0,
  but the host submits 12 launch calls plus 1 graph launch instead of 1,106 kernel launches. GPU idle falls
  from 22.7 to 6.6 ms and busy rises from 64 % to 86 %. GEMM time is unchanged.
- **torch.compile (G1) reduces GPU work.** Inductor fuses ~150 small kernels: 956 kernels, 516 of them
  generated Triton kernels (pattern matcher: 918 rewrites); the GEMMs stay on cuBLAS/CUTLASS Tensor-Core
  kernels. Each kernel is still launched individually (956 launch calls), so idle remains 10.8 ms.
- **reduce-overhead (G2) does use CUDA graphs.** Verified from counters, not assumed: 2 graph launches per
  step (forward and backward cudagraph trees), host launches fall to 48 (fused AdamW plus input copies),
  8 non-static inputs are copied into the graph's static buffers, and there were 0 cudagraph skips. It also
  has the lowest allocated peak (6.31 GiB) because the trees reuse one memory pool for forward/backward
  activations.
- **G4 combines both:** fused Inductor kernels replayed from one manual full-step graph give the lowest
  step (40.05 ms), the lowest GPU region (32.55 ms) and the shortest inference (8.28 ms). Remaining idle
  (5.4 ms) is the host data prep.
- **Per-layer compilation (G1L/G2L)** compiles 3× faster (14 s cold) but recovers only about half the
  gain. The cross-variable branches, input embeddings and output MLP stay eager, and they are where most
  of the small kernels are.

## 4. Correctness (Phase 4)

Deterministic comparisons ran in eval mode (dropout off) on a real ForeSWE batch, every configuration from
identical seed-0 weights and fresh AdamW state (`benchmark_results/correctness/graph_correctness.json`).
Acceptance criteria were fixed before running:
- manual graphs must be bitwise identical to G0c;
- compiled variants must stay within 10 % of G0's error vs the FP32 reference, and their difference from G0
  must be below TF32's own error vs FP32 for the same weights (output 7.2e-5, gradient 8.4e-4).

| Config | Output vs G0 | Gradient vs G0 | One-step update vs G0 | Params after 20 steps vs G0 | Output / grad error vs FP32 (G0: 7.23e-5 / 8.35e-4) | Verdict |
|---|---:|---:|---:|---:|---:|---|
| G0c | 0 (bitwise) | 0 (bitwise) | 0 | 0 | 7.23e-5 / 8.35e-4 | pass |
| G1 | 5.2e-5 | 2.9e-4 | 1.8e-2 | 1.0e-5 | 7.60e-5 / 8.29e-4 | pass |
| G2 | 5.2e-5 | 2.9e-4 | 1.9e-2 | 9.6e-6 | 7.60e-5 / 8.14e-4 | pass |
| G3f | **0 (bitwise)** | **0** | **0** | **0** | 7.23e-5 / 8.35e-4 | pass |
| G3 | **0 (bitwise)** | **0** | **0** | **0** | 7.23e-5 / 8.35e-4 | pass |
| G4 | 5.2e-5 | 2.9e-4 | 1.9e-2 | 9.6e-6 | 7.59e-5 / 8.14e-4 | pass |

- **Why the update error is 1e-2:** AdamW's first update is ≈ lr·sign(g) for most elements, so elements
  with tiny gradients amplify any gradient difference. After 20 steps the parameters differ by only ~1e-5.
- **Graph-safety checks (G3f/G3/G4):**
  - Capture warm-up left the weights bitwise unchanged and the AdamW state zero.
  - Every run trained: parameter change ‖Δθ‖ = 1.029 after 21 steps.
  - Train-mode replays draw fresh dropout masks; eval-mode replays are identical.
  - Memory plateaus: over 800 steps, allocated memory stays constant (7.73 GiB) and reserved stays flat at
    11.77 GiB after step 1 (eager: 9.28 GiB). The graph pool costs about 2.5 GiB of extra *reserved* memory
    and does not grow.
- **No NaN/Inf** in any configuration.

**Convergence (real data, production order, train mode, dropout seed 1234):**

| Config | 300 steps: last-50 mean loss | NaN/Inf | 1 epoch (3,297 steps): last-50 mean loss | Epoch time (+ setup) |
|---|---:|---:|---:|---:|
| G0 | 273,287.8 | 0 | 132,035.9 | 205.9 s |
| G1 | 273,281.1 | 0 | – | – |
| G2 | 273,281.1 | 0 | – | – |
| G3 | **273,287.8 (identical to G0)** | 0 | – | – |
| G4 | 273,281.1 | 0 | 132,024.8 | 167.4 s (+15.9 s warm compile) |

Epoch times include per-step gradient-norm and update-norm bookkeeping, so they are longer than
steps × step time. G4 vs G0 over one epoch: **1.23× faster (1.12× including compile)**, with final losses
within 0.01 %.

## 5. Best configuration and when compilation pays off

| Config | Step ms | Speedup vs E4 (measured G0) | GPU busy | Peak VRAM alloc / reserved | Compile time (cold / warm) | Output error vs G0 | Gradient error vs G0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| G0 E4 eager | 50.78 | 1.00× | 64 % | 8.51 / 9.07 GiB | – | – | – |
| G1 | 43.61 | 1.16× | 77 % | 8.50 / 11.11 | 46 / 15 s | 5.2e-5 | 2.9e-4 |
| G2 | 41.09 | 1.24× | 83 % | 6.31 / 10.60 | 47 / 17 s | 5.2e-5 | 2.9e-4 |
| G3 | 43.94 | 1.16× | 86 % | 7.92 / 11.11 | 0.5 s | 0 (bitwise) | 0 (bitwise) |
| **G4** | **40.05** | **1.27×** | **87 %** | 7.92 / 12.09 | 46 / 15 s | 5.2e-5 | 2.9e-4 |

Against the original FP32 + foreach production setup (148.2 ms), G4 is **3.70×** faster.

Break-even, from the measured savings (G3 6.8 ms/step, G4 10.7 ms/step) and setup cost:
- **G3:** 0.5 s / 6.8 ms ≈ **75 steps**.
- **G4:** cold 46 s / 10.7 ms ≈ 4,300 steps (1.3 epochs); warm cache 15 s ≈ 1,400 steps (0.4 epoch).

| Use case | Recommendation |
|---|---|
| Short experiments (≤ a few hundred steps) | **G3**: 1.16× after 0.5 s, bitwise identical to eager, no compiler |
| One epoch (3,297 steps) | **G3**: saves ≈ 22 s; G4 saves ≈ 20 s with a warm cache and loses ≈ 11 s on a cold one |
| Repeated multi-epoch training | **G4**: 1.27×, compile amortised (FX-graph cache hit verified on warm runs) |
| Memory-constrained | G2: lowest allocated peak (6.31 GiB) |

**Selected:** G4 for production-length training, G3 as the zero-risk default. Both are numerically
validated. G4 costs about 3 GiB of extra reserved memory (12.1 vs 9.1 GiB), well within 40 GB.

## 6. What is left

After G4, the 40 ms step is about 17.6 ms Tensor-Core GEMMs + about 7.5 ms host data prep
(19 %, not addressed) + the remaining fused elementwise, attention and optimizer kernels, with about 5 ms
idle. The largest untouched item is the data path: a GPU-side index gather (no full-dataset gathers) and
static distance/angularity tensors would make the prep capturable. The geo-attention core is examined in
`../fused_attention/`.

## Reproduce

```bash
JOB1=$(bash scripts/slurm/alloc.sh fsw_node1); JOB2=$(bash scripts/slurm/alloc.sh fsw_node2)
bash scripts/slurm/run.sh $JOB1 bash scripts/slurm/graphs_node1.sh      # G-matrix
bash scripts/slurm/run.sh $JOB2 bash scripts/slurm/graphs_node2.sh      # correctness, baseline, nsys, convergence
bash scripts/slurm/run.sh $JOB2 env STAGES=epoch EPOCH_CFGS="G0 G4" bash scripts/slurm/graphs_node2.sh
python -m benchmarks.make_figures_graphs                                # inside an allocation
```
