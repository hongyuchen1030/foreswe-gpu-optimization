# Long-Horizon ("Continuous Rollout") Adaptation of ForeSWE: Implementation Audit

## 1. What actually exists

I found **no autoregressive rollout** of ForeSWE that feeds predictions back as inputs up to
April 1, in any form (code, logs, checkpoints, transcripts). The upstream code has the hook for it,
`getInputsWithForecastedSWE` (`fore_swe/forecasting_att_daily_model.py:531`,
`fore_swe/forecasting_att_weekly_model.py:537`). It overwrites the SWE input column with forecast
SWE, but it is **never called**: `grep` finds only the definitions.

Two long-horizon variants exist instead. Both were built in May 2026 on top of the weekly model:

| | Plan A: stitched windows | Plan B: 24-week direct head |
|---|---|---|
| Requested in | Codex session 2026-05-21 (Fig. 6 recreation) | same session, msg [474]: *"At one fixed issue date, Feb 1 … predict all future weekly targets directly in one forward pass … forecasting_window = 24"* |
| Model change | none (upstream 4-week weekly model + checkpoint) | output head 4 → 24, then retrained |
| Code | `experiments/fig6_weekly_stitching/plot_weekly_fig6_paper_style.py:26,59-68` (post-processing only) | `experiments/foreswe_planB_24week/planB_24week_att_model.py`, `planB_24week_gp_model.py` |
| Horizon covered | weeks 8–23 from Dec 1, as four independent 4-week issues (8, 12, 16, 20) | Feb 1 + 7…168 d (Feb 8 → Jul 18). **Apr 1 = lead 59 d**, between head 8 (56 d) and head 9 (63 d) |
| Category (task's A/B/C) | inference-only re-use of the original model, no new compute | **B: modified training procedure with a longer horizon** |

**Plan B is the "continuous rollout" adaptation.** Rest of this document = Plan B unless noted.

## 2. Execution paths side by side (Deliverable B)

| Stage | Original weekly (`fore_swe/forecasting_att_weekly_model.py`) | Plan B (`experiments/foreswe_planB_24week/planB_24week_att_model.py`) |
|---|---|---|
| Launch-blocking | `os.environ['CUDA_LAUNCH_BLOCKING']='1'` (:24) | removed: kernels launch asynchronously |
| Horizon config | `forecasting_window = 4`, `forecasting_step_size = 7` (:42-43) | `forecasting_window = 24`, `forecasting_step_size = 7` (:56-57) |
| Train issue days | `training_data_indices[:-(4*7)]` (:71) | `[: -(24*7)]` (:114) → 3297 days (168 fewer, so no target crosses into test years) |
| Head | `SWETransformer(..., n_output_heads=4)` (:240) | `n_output_heads=24` (:281) → `final_output = Linear(8, 24)`. Everything else is byte-identical (class bodies re-indented) |
| Target | `getOutputs`: `index + v*7, v=1..4` (:318-323) | `get_outputs`: `index + v*7, v=1..24` (:291-294) |
| Inputs | daily 30×1 d, weekly 30×7 d, yearly 5×365 d, ending at the issue day (:332-334) | unchanged (:297-313) |
| Training loop | `TrainModelSP` (:325-366): per issue day, forward → MSE over 4 leads → `loss.item()` → `zero_grad` → `backward` → `step`; logs every 100 steps; early stop (patience 3, max 6 epochs) | `train_epoch` (:316-335): identical step structure; MSE over **24** leads; logs and flushes **every step**; no early stop; 6 epochs; seeded (`PLANB_SEED`) |
| Checkpointing | save on improved epoch loss (:407-412) | same, plus `best_loss` key (:357-366) |
| Eval: representations | `getFinalInputsForGP` over all train days, `torch.no_grad()` (:438-475) | `collect_representations` (:379-396), same |
| Eval: test | `TestModelSP` = 1 forward per day Dec 1–May 30 (~181/yr) (:489-535) + `getDailyPredictions` stitched every 28 d (:555-602) | 1 forward per year, at the Feb 1 index only (:400-421) |
| Output | `(days, 512, 4)` per year | `(1, 512, 24)` per year (`planB_shape_check.txt`) |
| GP head | `forecasting_weekly_gp_model.py`, 4 latents | `planB_24week_gp_model.py`: inputs `[-1, 8]`, outputs `[-1, 24]`, 7000 inducing points; OOM on 40 GB, fixed per `gp_oom_diagnosis.md` |

## 3. The investigation checklist

| Question | Answer (Plan B) | Evidence |
|---|---|---|
| Autoregressive rollout mechanism | **None.** All 24 leads come out of one linear head in one forward pass | `:281`, `:316-335` |
| Forecast initialisation | Fixed issue index, Feb 1 of each test year (8511, 8876, 9241, 9606, 9971), matched by date string | `:74-80`, `:136-141` |
| Predictions fed back as inputs? | **No.** Inputs are observed history ending at the issue day | `get_example` `:297-313` |
| Sequence length / temporal resolution | Encoder sequence = 512 station tokens. Temporal context enters as feature width (30 daily, 30 weekly, 5 yearly samples) through per-variable `Linear(window → 1024)`. Output resolution weekly | `CrossVarTransformer.forward` |
| Fixed vs growing context | Fixed | windows constant |
| Teacher forcing vs AR feedback | Neither: direct multi-horizon regression | |
| Gradients across rollout? | No rollout. One graph per step; loss = mean MSE over 24 leads; the graph depth is the same as the 4-lead model | `:321-325` |
| Train or inference mode for the long horizon | Trained (`swe_model.train()`), evaluated with `eval()` + `no_grad()` | `:353`, `:376`, `:383` |
| Checkpoint loading | `torch.load(map_location=device)` restores model + optimizer. Resumes if the file exists, so the full run resumed from the smoke run's checkpoint (`epoch`=2) | `:343-349`, `:369-374` |
| CPU↔GPU transfers per step | H2D: distances + angularities, `np.ndarray[idx][:, idx]` → float32, 2 × 1 MiB. D2H: representation (16 KiB) + prediction + target (48 KiB each) via `.detach().cpu()`. The whole dataset `model_inputs` (10305×512×19 fp32, 400 MB) stays GPU-resident | `:302-303`, `:387-389`, `:149-150` |
| Intermediate tensor storage | `get_inputs` does `model_inputs[:, loc_indices, :]`, which gathers the **entire 400 MB dataset** before slicing the window, 3× per step; then `torch.tensor(t)` copies again | `:286-288`, `:305-307` |
| Memory management | No `empty_cache`, no AMP, no checkpointing. Train-time representations are accumulated on CPU as a Python list | `:384-390` |
| Result saving | `torch.save` of representation lists, `joblib.dump` of per-year pickles to `results/{actual,predict}` | `:394-416` |

## 4. Effect on computational workload

Measured on an A100-40GB, FP32 (`GPU_PROFILING_REPORT.md`).

- **Per training step: unchanged.** The only new compute is `Linear(8→24)` in place of
  `Linear(8→4)`: +180 params out of 627.6 M. Measured step time is 148.4 ms (Plan B) vs
  147.3 ms (weekly-4); peak memory is 9.11 GiB for both.
- **Per epoch: 4% fewer steps** (3297 vs 3437 issue days), because the 24-week target window
  removes 140 more days at the end of the training period.
- **Wall time: Plan B removed `CUDA_LAUNCH_BLOCKING=1`.** On its own this cuts the training step
  from 182 → 148 ms (−19%) and the inference forward from 48.4 → 38.4 ms. The historical Plan B
  run was still slower per step (233 ms). See the unexplained gap noted in the history doc.
- **Inference for a season: ~190× cheaper.** One forward per test year (~47 ms) instead of
  ~181 daily forwards plus 7 stitched forwards with the original weekly test loop (~8.6 s/yr).
  Total test-set inference for 5 years drops from ~45 s to 0.24 s. Representation export over
  3297 training days dominates evaluation (≈3297 × 47 ms ≈ 2.6 min).
- **Memory: unchanged** for the attention model. The GP head is where the horizon costs memory:
  24 outputs / 5 latents with 7000 float64 inducing points need 1.83 GiB per allocation, which
  OOMed on 40 GB and needed an 80 GB A100.

The adaptation is not a long-rollout workload. In GPU terms it has the same steady-state kernel
profile as the original, so the bottleneck analysis applies to all three variants.
