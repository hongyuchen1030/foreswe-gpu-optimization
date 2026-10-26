# Plan B 24-Week Weekly Forecast Summary

## Status Note

This file currently documents the earlier smoke/debug pass and should be treated as historical implementation notes only.

- The smoke-run outputs and smoke-run plot are **not** scientifically meaningful.
- The Plan B scripts have since been updated so they now default to a **real run configuration**:
  - `PLANB_SMOKE=0`
  - `PLANB_GP_SMOKE=0`
  - no zero-prompt fallback unless explicitly enabled with `PLANB_ALLOW_ZERO_PROMPTS=1`
  - local Hugging Face / SentenceTransformer cache configured from:
    - `/pscratch/sd/h/hyvchen/foreswe_model_cache`

Use this note for code provenance, but do not treat the smoke-run figures or smoke-run statistics below as final Plan B results.

## Goal

Implement a non-stitched weekly ForeSWE experiment where one fixed issue date, Feb 1, predicts all future weekly targets in one forward pass:

- `forecasting_window = 24`
- `forecasting_step_size = 7`
- targets: `t+7, t+14, ..., t+168`

This is intentionally different from the original weekly ForeSWE behavior, which uses `forecasting_window = 4` and later stitches multiple 4-week forecasts.

## Files Copied / Adapted

- [planB_24week_att_model.py](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week/planB_24week_att_model.py)
- [planB_24week_gp_model.py](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week/planB_24week_gp_model.py)
- [plot_planB_24week_feb1.py](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week/plot_planB_24week_feb1.py)

These were created under the isolated experiment directory. The original third-party ForeSWE source files were not modified.

## Exact Changed Variables

- original weekly setting:
  - `forecasting_window = 4`
  - `forecasting_step_size = 7`
- Plan B setting:
  - `forecasting_window = 24`
  - `forecasting_step_size = 7`

The target formula in the adapted attention script is:

`index + value * 7 for value in range(1, 25)`

## Input History Windows

The original weekly input-history design was kept unchanged:

- daily branch: `window = 30`, `step_size = 1`
- weekly branch: `window = 30`, `step_size = 7`
- yearly branch: `window = 5`, `step_size = 365`

No future inputs after the issue date are used as model inputs.

## Fixed Feb 1 Issue Dates

The fixed issue indices identified from the original loader date axis were:

- `2015-02-01 -> 8511`
- `2016-02-01 -> 8876`
- `2017-02-01 -> 9241`
- `2018-02-01 -> 9606`
- `2019-02-01 -> 9971`

These were verified and recorded in [planB_shape_check.txt](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week/planB_shape_check.txt).

## Output Shape

The adapted attention model produced fixed-issue test outputs with shape:

- actual: `(1, 512, 24)`
- predicted: `(1, 512, 24)`

for each of the five test seasons.

The adapted GP head produced:

- mean: `(1, 512, 24)`
- lower: `(1, 512, 24)`
- upper: `(1, 512, 24)`

for each of the five test seasons.

## What Completed

### Attention side

Completed as a smoke run.

- training smoke:
  - `PLANB_SMOKE=True`
  - `PLANB_MAX_TRAIN_INDICES=4`
  - `PLANB_MAX_EPOCHS=1`
- output head shape verified in the training log as:
  - `output_shape (1, 512, 24)`
- saved outputs:
  - [forecasting_modified_weekly_att_inputs_for_gp_act.pt](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week/outputs/forecasting_modified_weekly_att_inputs_for_gp_act.pt)
  - [forecasting_modified_weekly_att_outputs_for_gp_act.pt](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week/outputs/forecasting_modified_weekly_att_outputs_for_gp_act.pt)
  - [forecasting_modified_weekly_att_predicted_outputs_for_gp_act.pt](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week/outputs/forecasting_modified_weekly_att_predicted_outputs_for_gp_act.pt)
  - [weekly_test1_inputs_for_gp_act.pt](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week/outputs/weekly_test1_inputs_for_gp_act.pt)
  - [weekly_test2_inputs_for_gp_act.pt](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week/outputs/weekly_test2_inputs_for_gp_act.pt)
  - [weekly_test3_inputs_for_gp_act.pt](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week/outputs/weekly_test3_inputs_for_gp_act.pt)
  - [weekly_test4_inputs_for_gp_act.pt](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week/outputs/weekly_test4_inputs_for_gp_act.pt)
  - [weekly_test5_inputs_for_gp_act.pt](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week/outputs/weekly_test5_inputs_for_gp_act.pt)

### GP side

Completed as a smoke run.

- GP train inputs were reshaped to `[-1, 8]`
- GP train outputs were reshaped to `[-1, 24]`
- test outputs were reshaped to `[-1, 24]`

Saved Plan B outputs:

- [planB_feb1_24week_mean.pkl](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week/planB_feb1_24week_mean.pkl)
- [planB_feb1_24week_low.pkl](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week/planB_feb1_24week_low.pkl)
- [planB_feb1_24week_high.pkl](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week/planB_feb1_24week_high.pkl)
- [planB_feb1_24week_actual.pkl](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week/planB_feb1_24week_actual.pkl)

## Plot

Generated:

- [planB_feb1_24week_plot.png](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week/planB_feb1_24week_plot.png)
- [planB_feb1_24week_plot.pdf](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week/planB_feb1_24week_plot.pdf)

The plotting script automatically selected:

- year: `2017`
- station index: `239`
- station name: `Leavitt Lake`

This is recorded in [planB_plot_selection.txt](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week/planB_plot_selection.txt).

## Commands Used

### Attention smoke test

```bash
/pscratch/sd/h/hyvchen/conda_envs/foreswe_repro_att/bin/python \
  /global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week/planB_24week_att_model.py
```

### GP smoke test

```bash
/pscratch/sd/h/hyvchen/conda_envs/foreswe_repro_att/bin/python \
  /global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week/planB_24week_gp_model.py
```

### Plot generation

```bash
/pscratch/sd/h/hyvchen/conda_envs/foreswe_repro_att/bin/python \
  /global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week/plot_planB_24week_feb1.py
```

## Remaining Issues

- Full faithful 24-week training did **not** complete. What completed here was:
  - script creation
  - shape verification
  - one-epoch / small-sample smoke training
  - GP smoke run
  - plot generation
- Prompt embeddings could not be regenerated on this node because the SentenceTransformer model was not available in local cache and network access to Hugging Face was unavailable.
- To keep the experiment runnable offline, the attention smoke run used:
  - `prompt_mode=zero-fallback`
- That means the smoke-run outputs are suitable for verifying Plan B mechanics and shapes, but they should **not** be treated as a final scientific 24-week ForeSWE result.

## Clear Next Full-Run Commands

Once prompt embeddings are available locally, a fuller run would be:

```bash
PLANB_SMOKE=0 PLANB_MAX_EPOCHS=6 \
/pscratch/sd/h/hyvchen/conda_envs/foreswe_repro_att/bin/python \
  /global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week/planB_24week_att_model.py
```

```bash
PLANB_GP_SMOKE=0 PLANB_GP_EPOCHS=2 PLANB_GP_INDUCING=7000 PLANB_GP_BATCH=4096 \
/pscratch/sd/h/hyvchen/conda_envs/foreswe_repro_att/bin/python \
  /global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week/planB_24week_gp_model.py
```
