# ForeSWE Experiment History (audit, 2026-10-26)

Scope: every ForeSWE attention-model run I could identify on Perlmutter, for user `hyvchen`, from the
repository, pscratch, SLURM accounting (`sacct`), run logs and past agent transcripts
(`~/.codex/sessions/2026/05/*`). Nothing was re-trained or re-run to recover history.

Evidence labels used throughout:

- **[V]** verified: read directly from a log, file, `sacct` record or file timestamp (source given).
- **[D]** derived: arithmetic on verified numbers (formula given).
- **[M]** newly measured on 2026-10-26 (see `GPU_PROFILING_REPORT.md`).
- **NOT FOUND**: searched for and absent. Nothing is guessed.

---

## 0. Headline findings

1. **No autoregressive "continuous rollout through April 1" implementation exists** anywhere in the
   repo, the experiment folders, pscratch or the May 2026 transcripts. Searches covered
   `rollout|autoregress|april 1|apr1|0401`, the `getInputsWithForecastedSWE` helper (defined in
   upstream but never called), and the transcripts. Two long-horizon adaptations do exist:
   - **Plan A, stitched:** the unchanged 4-week weekly model, run at issue weeks 8/12/16/20 with the
     4-week chunks concatenated for plotting (Fig. 6 recreation). No model change.
   - **Plan B, 24-week direct:** retrained with a **24-output head** (t+7 … t+168 days) from a
     fixed **Feb 1** issue date, in **one forward pass**. Its window covers April 1 (lead 59 d
     falls between the week-8 and week-9 outputs) and runs to mid-July.
   This audit treats **Plan B as "the continuous-rollout adaptation"**, because it is the only
   long-horizon variant with its own code, checkpoint and training run. It is category **B**
   (modified training procedure, longer horizon). It is **not** an autoregressive rollout. See
   `CONTINUOUS_ROLLOUT_IMPLEMENTATION.md`.
2. **No forward/backward/optimizer timing, GPU utilisation, or attention-model GPU memory figures
   were ever recorded.** The only historical performance data are epoch wall times, SLURM elapsed
   times, and one GP-head memory diagnosis (Plan B GP OOM).
3. **The historical conda env (`$SCRATCH/conda_envs/foreswe_repro_att`, torch 2.4.1+cu121) is
   broken.** The pscratch purge removed its Python stdlib (`lib/python3.8/encodings/` holds only
   `__pycache__`; 1 `.py` file remains in `lib/python3.8`), so it fails at startup with
   `init_fs_encoding`. Checkpoints and outputs are still present. **Copy any checkpoint you want to
   keep off pscratch; the purge may take it next.**

---

## Deliverable A: Experiment inventory

Columns: **Original** = upstream ForeSWE daily model reproduced at full 512 stations (the README's
"ForeSWE model"). **Weekly** = upstream weekly model, which is also what Plan A plots. **Plan B** =
the long-horizon adaptation.

| Field | Original ForeSWE (daily, 512 stn) | Original weekly (512 stn) / Plan A | Continuous / long-horizon = Plan B (24-wk) |
|---|---|---|---|
| Source code path | `thirdparties/SWE-Forecasting/fore_swe/forecasting_att_daily_model.py` (upstream `Krishuthapa/SWE-Forecasting` @ `393de07`, 2026-05-04, branch `dev`; local diff = `results/actual` pickle dumps only) [V] | `experiments/foreswe_weekly_reproduction_full512/scripts/forecasting_att_weekly_model.py` (copy of upstream `fore_swe/forecasting_att_weekly_model.py` with path/log edits) [V]; Plan A plots: `experiments/fig6_weekly_stitching/plot_weekly_fig6_paper_style.py` [V] | `experiments/foreswe_planB_24week/planB_24week_att_model.py` (attention) + `planB_24week_gp_model.py` (GP head) [V] |
| Entry point | `python -u fore_swe/forecasting_att_daily_model.py` with `BASE_URL`, `PYTHONPATH` = repo [V: `experiments/foreswe_reproduction/slurm/run_att_daily_gpu.sbatch`] | `scripts/run_weekly_att_interactive.sh` [V] | `$SCRATCH/conda_envs/foreswe_repro_att/bin/python planB_24week_att_model.py` with `PLANB_SMOKE=0 PLANB_MAX_EPOCHS=6` [V: `planB_24week_summary.md`, `planB_shape_check.txt`] |
| Model architecture | `SWETransformer`: 3 × `CrossVarTransformer` (daily window 1, "monthly" 120×7 d, yearly 5×365 d; d=1024, MHA over 13 variables) → 8 × custom geo-biased encoder layers (d=3072, 16 heads, FF 2048, attention biased by station distance + angularity) → MLP 3072→1024→128→8 → Linear(8→10). **426.8 M params** [M] | Same, but daily/weekly windows 30×1 d and 30×7 d, plus 3 × 8 **dead** d=1024 encoder layers inside `CrossVarTransformer` (built but never called, 201.6 M params). Head Linear(8→4). **627.6 M params** [M] | Identical to weekly except head Linear(8→24). **627.6 M params**, of which 201.6 M are never used [M] |
| Training dataset | 512 SNOTEL stations, `data/*.csv` (`model_inputs` (10305, 512, 19)); train water years 1991–2014 (README); ~3455 issue days [D] | same; ~3437 issue days [D] | same; **3297** issue days (`train_indices_count`, `planB_shape_check.txt`) [V] |
| Evaluation dataset | test WY 2015–2019, every day Dec 1–May 30 (181/182/180/181/181 issue days) [V: `FIG4A_VALIDATION.md`] | same days; weekly windows | 5 issue dates only: Feb 1 of 2015–2019 (indices 8511…9971) [V] |
| Forecast horizon | 10 daily leads (t+1 … t+10 d) | 4 weekly leads (t+7 … t+28 d); Plan A stitches issues at weeks 8/12/16/20 | 24 weekly leads (t+7 … t+168 d) from Feb 1, which covers Apr 1 |
| Checkpoint path | `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/thirdparties/SWE-Forecasting/checkpoints/forecasting_modified_daily_att_climax_model_gp_act.pt` (5.12 GB, 2026-05-07 10:06, `epoch`=6) [V]. Older copy: `experiments/foreswe_reproduction/archive_pre_scratch_attention_20260506_144156/checkpoints/` (2026-05-04 21:34) [V]. Note: the repo symlink `checkpoints -> …/experiments/foreswe_reproduction/checkpoints` points at an **empty** dir [V] | `/pscratch/.../experiments/foreswe_weekly_reproduction_full512/checkpoints/forecasting_modified_weekly_att_climax_model_gp_act.pt` [V] | `/pscratch/.../experiments/foreswe_planB_24week/checkpoints/planB_24week_att_model.pt` (5.92 GB, 2026-05-21 23:43, `epoch`=7, `best_loss`) [V] |
| SLURM job ID | Training: **52634970** (`gpu_interactive`, nid001104, 2026-05-07 07:55–11:55, TIMEOUT at 4 h after outputs written 10:36–10:41) [V: sacct + file mtimes]. Eval-only re-run: **52509595** (`gpu_regular`, nid008425, 5 min 59 s) [V]. Epochs 1–2: probably 52458927 / 52463684 (2026-05-04), **not confirmed** | **53260521** (`gpu_interactive`, nid002601, 2026-05-21 13:00–14:07) [V: sacct; logs `weekly_att_interactive.log`] | Attention: **53283879** (`gpu_interactive`, nid001089, 2026-05-21 22:38–23:47) [V]. GP head: **53313853** (`gpu&hbm80g`, nid008549, 2026-05-22 13:07–14:43) [V] |
| Execution command | `srun -n1 -c32 --gpus=1 --gpu-bind=none conda run --no-capture-output -p $FSW_ENV python -u fore_swe/forecasting_att_daily_model.py` [V: sbatch]; training run 52634970 was interactive, exact command NOT FOUND | `run_weekly_att_interactive.sh` (tmux + interactive srun) [V] | see entry point; GP: `run_planb_gp_interactive_80g_verified.sh` (`PLANB_GP_EPOCHS=2 PLANB_GP_BATCH=4096 PLANB_GP_INDUCING=7000`) [V] |
| GPU type / count | 1 × A100-SXM4-40GB used (`cuda:0`); 4 allocated (`gres/gpu:a100=4`, nid001xxx are 40 GB nodes) [V] | 1 × A100 (nid002601, 40 GB class) [V] | attention: 1 × A100-40GB (nid001089) [V]; GP: 1 × A100-80GB [V: log `nvidia-smi`] |
| CPU / memory allocation | 128 CPUs, 229,902 MB (whole node, sacct AllocTRES) [V] | same [V] | same [V] |
| Batch size | 1 issue date × all 512 stations per step (stations are the attention "sequence"); no minibatching [V: code] | same | same |
| Precision | FP32, no autocast; `torch.backends.cuda.matmul.allow_tf32` default False [V: code + M] | FP32 | FP32 |
| Optimizer / LR | AdamW lr 5e-4, wd 1e-4; StepLR(2, γ=0.5); MSE loss; ≤8 epochs, patience 3 [V: lines 241–242, 388] | AdamW lr 5e-6, wd 1e-4; StepLR(4, 0.65); ≤6 epochs | AdamW lr 5e-6, wd 1e-4; StepLR(4, 0.65); 6 epochs, no early stop |
| `CUDA_LAUNCH_BLOCKING` | **'1'** hard-coded (line 24) [V] | **'1'** hard-coded [V] | **not set** (removed in the Plan B copy) [V] |
| PyTorch / CUDA | torch 2.4.1+cu121, Python 3.8, driver 580.105.08 [V: job 52509595 stdout] | same env [V] | same env [V] |
| Training runtime | epochs 3–8: **3679.4 s** (`Time Elapsed`, `outputs/forecasting_modified_daily_att_tr_act_loss.txt`) [V] ⇒ 613 s/epoch, **≈177 ms/step** (3679.4 / (6 × 3455)) [D]. Epochs 1–2: NOT FOUND | 1 epoch (resumed at epoch 6): **602.0 s** [V] ⇒ **≈175 ms/step** (602 / 3437) [D] | epochs 2–6: **3836.3 s** (`Elapsed_seconds`, `planB_24week_train_log.txt`) [V] ⇒ **232.7 ms/step** (3836.3 / 16485 logged batches) [D]. Epoch 1 was the 4-sample smoke run (checkpoint resumed at epoch 2) [V] |
| Evaluation runtime | Eval-only job 52509595: 5 min 59 s for ~3455 train-representation forwards + 905 test forwards + 94 stitched forwards (stdout timestamps 02:25:39 → 02:31:33) [V] | NOT FOUND separately | ≤ 4 min after training (checkpoint 23:43 → outputs 23:47) for 3297 + 5 forwards [V: mtimes] |
| GP output head | `forecasting_daily_gp_model.py`: CPU jobs 52715240 (6 h TIMEOUT) and 52825860 (12 h TIMEOUT); GPU-80GB job **52868684** completed in **1 h 22 m 38 s** [V] | `logs/weekly_gp_interactive_80g.log` [V] | `planB_24week_gp_model.py`: OOM on 40 GB (`gp_oom_diagnosis.md`), then ~34 min on A100-80GB (20:09:56Z → 13:43 PDT log end) [V] |
| Forecasting results | Daily GP-head pickles `results/predict/201{4..8}_daily_gp_*` (pscratch); Fig. 4a relative bias reproduced (`experiments/foreswe_reproduction/FIG4A_VALIDATION.md`) [V] | Station NSE bins `[0,103,282,82,34,11]` (`figures/reproduced_fig3bd_weekly_foreswe_only_NOTE.md`) [V] | Feb-1 issue, 512 stations × 5 yrs: week 1 RMSE 111.8 mm, r 0.887; week 8 RMSE 174.8, r 0.879; week 9 RMSE 186.2, r 0.875; week 24 r 0.11 (`planB_pred_vs_actual_24weeks_full512_summary.csv`) [V] |
| Profiling data available? | **None historically.** New: [M] | **None historically.** New: [M] | **None historically** (only the GP-head memory estimate in `gp_oom_diagnosis.md`). New: [M] |

Other ForeSWE runs found, not tabulated above: the Sierra 43-station subset
(`/pscratch/.../experiments/foreswe_sierra_station`, train 1918.6 s for 7 epochs at 43 stations,
job 52959384); the D2 Sierra+500 km domain (`experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km`,
job 53195212, 45 m 50 s); and the bias diagnosis (`experiments/foreswe_bias_diagnosis/`, analysis only).

---

## Phase 2: Original ForeSWE historical run, Q&A

1. **Exact version.** Upstream `https://github.com/Krishuthapa/SWE-Forecasting`, commit
   `393de0750160b257eab83495848a75c089068b76` ("Update citation format in README.md", 2026-05-04).
   Local changes (`git diff`): `att_daily_run.sh` `base_url`, and 5 `joblib.dump` lines appended to
   `forecasting_att_daily_model.py` to save `results/actual/*.pkl`. The model and training logic are
   unmodified. [V]
2. **Data and configuration.** Upstream `data/*.csv` (16 CSVs, 512 SNOTEL stations, 13 physical
   variables plus 6 space-time prompt features = 19 columns), sentence-transformer station prompts
   (1536-d) from the `foreswe_model_cache` HF cache. Config is hard-coded in the script, with no
   CLI overrides. [V]
3. **From scratch or pretrained?** From scratch. Before the from-scratch run the earlier checkpoint
   was moved to `archive_pre_scratch_attention_20260506_144156/` (sbatch echo: "Starting
   from-scratch ForeSWE attention training"). The run that produced the current checkpoint resumed
   once internally (its log starts at epoch 3). [V]
4. **Results.** Training MSE per epoch, epochs 3–8: 963.0, 914.8, **912.7** (best, saved as
   `epoch`=6), 1005.6, 1009.3, 1084.1. Test per-window MSE in `..._ts_act_loss.txt` (905 batches).
   GP-head predictions feed the Fig. 4 reproduction. [V]
5. **Time.** Training: 3679 s for 6 epochs ≈ 61 min [V], ≈177 ms/step [D]. Full-pipeline
   inference (all representations + test): ≈6 min [V]. GP head: 1 h 23 min on A100-80GB [V].
6. **GPU resources.** One A100-SXM4-40GB actually used (code pins `cuda:0`), on a 4-GPU node. [V]
7. **Hyperparameters.** Batch = 1 issue date × 512 stations. "Sequence length" = 512 station
   tokens in the main encoder and 13 variable tokens per station in the cross-variable attention.
   FP32. AdamW lr 5e-4, wd 1e-4, StepLR(2, 0.5). [V]
8. **Bottlenecks documented before?** Only for the GP head: CPU GP jobs hit 6 h and 12 h limits,
   and Plan B's GP OOMed on 40 GB, with a precise allocation analysis in `gp_oom_diagnosis.md`
   (1.83 GiB float64 inducing covariance across 5 latents). Nothing was documented for the
   attention model. [V]

### Which performance metrics existed before this audit?

| Metric | Original daily | Plan B |
|---|---|---|
| Forward-pass latency | NOT FOUND | NOT FOUND |
| Backward-pass latency | NOT FOUND | NOT FOUND |
| Optimizer-step latency | NOT FOUND | NOT FOUND |
| End-to-end training-step latency | only derivable from epoch wall time: ≈177 ms [D] | ≈233 ms [D] |
| Evaluation / inference latency | job-level only (6 min eval run) [V] | NOT FOUND |
| Peak GPU allocated / reserved memory | NOT FOUND | NOT FOUND (GP head only: OOM trace "19.82 GiB allocated", `gp_oom_diagnosis.md`) |
| GPU utilisation | NOT FOUND (no `nvidia-smi dmon`/DCGM logs; `sacct` has `energy=` only) | NOT FOUND |
| GPU throughput | NOT FOUND | NOT FOUND |
| Profiler traces | NOT FOUND | NOT FOUND |

All of these are now measured: see `GPU_PROFILING_REPORT.md`.

### Cross-check of new measurements against history

| | Historical [D] | Re-measured [M] (A100-40GB, torch 2.6.0, same launch-blocking setting) | Agreement |
|---|---|---|---|
| Daily, `CUDA_LAUNCH_BLOCKING=1` | 177 ms/step | 183.5 ms/step | within 4% |
| Weekly, `CUDA_LAUNCH_BLOCKING=1` | 175 ms/step | 182.8 ms/step | within 5% |
| Plan B, launch-blocking off | 233 ms/step | 148.4 ms/step | **+57% unexplained** |

The Plan B gap is not explained by GPU work. **Hypothesis, unverified:** Plan B writes and
`flush()`es one log line per step (16,485 flushes) to a `/global/homes` file
(`planB_24week_att_model.py:324–331`), where the upstream scripts print every 100 steps. Host
contention on the shared interactive node is another possibility. A one-epoch timing with the log
redirected to `$TMPDIR` would settle it.

---

## Evidence index

- Repo: `thirdparties/SWE-Forecasting` (git), `experiments/foreswe_*`, `experiments/fig6_weekly_stitching`.
- pscratch: `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/{thirdparties/SWE-Forecasting,experiments/foreswe_*}`.
- SLURM: `sacct -u hyvchen -S 2026-05-01 -E 2026-06-01 -X --format=JobID,JobName,QOS,State,Start,End,Elapsed,AllocTRES,NodeList`.
- Transcripts: `~/.codex/sessions/2026/05/21/rollout-2026-05-21T21-02-12-*.jsonl` (Plan B request, msg [474]),
  `…/05/22/rollout-2026-05-22T11-44-08-*.jsonl` (GP OOM).
