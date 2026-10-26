# ForeSWE Bias Diagnosis

## Files inspected
- `thirdparties/SWE-Forecasting/fore_swe/forecasting_att_weekly_model.py`
- `thirdparties/SWE-Forecasting/fore_swe/forecasting_weekly_gp_model.py`
- `thirdparties/SWE-Forecasting/fore_swe/extended_dl_forecasting.py`
- `experiments/foreswe_planB_24week/planB_24week_att_model.py`
- `experiments/foreswe_planB_24week/planB_24week_gp_model.py`
- `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_weekly_reproduction_full512/...` saved outputs
- `experiments/foreswe_planB_24week/...` saved outputs

## Code-level diagnosis

### Attention loss
- Original weekly attention loss: `nn.MSELoss()` at `forecasting_att_weekly_model.py:244`, applied at `forecasting_att_weekly_model.py:348-355` during training and again at `forecasting_att_weekly_model.py:517-524` during test logging.
- Plan B attention loss: `nn.MSELoss()` at `planB_24week_att_model.py:282`, applied at `planB_24week_att_model.py:320-324`.
- The loss is plain MSE on raw SWE targets. No high-SWE weighting, no lead weighting, and no station-group weighting were found in either attention script.

### Target normalization and inverse transform
- Inputs are globally min-max normalized across all station columns in `extended_dl_forecasting.py:20-29`.
- Input normalization is applied in `extended_dl_forecasting.py:418-435` to meteorology, partitioned rain/snow, input SWE, and day length.
- `extended_dl_forecasting.py:257-271` contains a likely feature bug: `self.snow_info` is overwritten from `self.rain_info` at line 270, and `self.rain_info` is then rebuilt from the already overwritten `self.snow_info` at line 271. That means the shared dataloader used by both the original weekly model and Plan B likely corrupts or duplicates the rain/snow feature channels.
- The target SWE is **not normalized** before the attention loss. `getModelInputsAndOutputs()` stores raw SWE targets from `self.swe_values` in `extended_dl_forecasting.py:123-131`.
- `getMaxMinSWE()` returns `(highest_swe + 500, 0)` in `extended_dl_forecasting.py:410-416`, but this is used only when replacing SWE inputs during recursive daily prediction, not to scale the weekly training targets.
- Original weekly GP and Plan B GP both define a no-op input normalizer (`return tensor`) in `forecasting_weekly_gp_model.py:66-68` and `planB_24week_gp_model.py:93-95`.
- No inverse transform is applied to the saved attention or GP outputs because the targets are already on the raw SWE scale.
- No clipping, sigmoid, tanh, ReLU, softplus, or clamp is applied to the final SWE outputs in either attention head.
- SWE zeros are kept as ordinary training targets; there is no special positive-only transform or zero inflation handling.

### Output head
- Original weekly attention head: `Linear(3*model_dim -> model_dim)`, `GELU`, `Dropout(0.05)`, `Linear(model_dim -> 128)`, `GELU`, `Linear(128 -> 8)`, `GELU`, final `Linear(8 -> n_output_heads)` in `forecasting_att_weekly_model.py:203-214`.
- Plan B attention head uses the same structure for `n_output_heads=24` in `planB_24week_att_model.py:255-264`.
- The final projection is linear, so the head can represent large SWE values in principle. There is no positivity constraint or explicit large-SWE amplification.

### GP model
- Original weekly GP sets `torch.float64` in `forecasting_weekly_gp_model.py:29`, reshapes training labels to `(-1, 4)` in `forecasting_weekly_gp_model.py:70-72`, and trains a multitask variational GP with `LMCVariationalStrategy` in `forecasting_weekly_gp_model.py:83-129`.
- Original weekly GP likelihood is `MultitaskGaussianLikelihood(num_tasks=4)` at `forecasting_weekly_gp_model.py:143-157` and the training objective is `-VariationalELBO` at `forecasting_weekly_gp_model.py:157-167`.
- Plan B GP also uses `torch.float64`, leaves the 8-dim attention features unnormalized, and trains `MultitaskGPModel(ApproximateGP)` with `LMCVariationalStrategy`, `ScaleKernel(RBF + Linear)`, and `MultitaskGaussianLikelihood(num_tasks=24)` in `planB_24week_gp_model.py:139-200`.
- The GP mean is the saved `preds.mean`, while the intervals come from `confidence_region()` in both weekly and Plan B GP scripts.

## Loaded tensor shapes

| name                            | shape           |
| ------------------------------- | --------------- |
| original_attention_train_actual | (3437, 512, 4)  |
| original_attention_train_pred   | (3437, 512, 4)  |
| original_gp_test_actual         | (905, 512, 4)   |
| original_gp_test_mean           | (905, 512, 4)   |
| planb_attention_train_actual    | (3297, 512, 24) |
| planb_attention_train_pred      | (3297, 512, 24) |
| planb_attention_test_actual     | (5, 512, 24)    |
| planb_attention_test_pred       | (5, 512, 24)    |
| planb_gp_test_actual            | (5, 512, 24)    |
| planb_gp_test_mean              | (5, 512, 24)    |

### Original weekly GP saved-output alignment

The original weekly GP reproduction artifacts are not perfectly self-consistent: `2016_weekly_gp_window_predict.pkl`, `2016_weekly_gp_low_window_predict.pkl`, and `2016_weekly_gp_high_window_predict.pkl` each have 182 issue rows, while `2016_weekly_window_actual.pkl` has 180 rows. The analysis aligned each year to the common saved row count and wrote the exact bookkeeping to `original_weekly_gp_year_alignment.csv`.

### Source caveat for original weekly attention

The expected original weekly attention test pickle files (`2014_weekly_window_predict.pkl` ... `2018_weekly_window_predict.pkl`) are not present in the saved reproduction directory, even though the script contains `joblib.dump` calls for them. The original weekly attention-vs-actual analysis therefore uses the saved attention outputs that were written for GP training (`forecasting_modified_weekly_att_predicted_outputs_for_gp_act.pt` and `forecasting_modified_weekly_att_outputs_for_gp_act.pt`), while the original weekly GP analysis uses the saved per-year GP test outputs.

## Generated plots
- `predicted_vs_actual_original_weekly.png`
- `predicted_vs_actual_planB_24week.png`
- `bias_by_actual_swe_bin.png`
- `bias_by_lead_week.png`
- `target_distribution_by_lead.png`
- `attention_vs_gp_bias_comparison.png`
- `original_weekly_attention_3panel.png`
- `planB_gp_3panel.png`

## Distribution imbalance

| dataset         | split                 | swe_bin  | count    | fraction |
| --------------- | --------------------- | -------- | -------- | -------- |
| original_weekly | train_attention_saved | SWE=0    | 1202939  | 0.1709   |
| original_weekly | train_attention_saved | 0-100    | 727370   | 0.1033   |
| original_weekly | train_attention_saved | 100-300  | 2324071  | 0.3302   |
| original_weekly | train_attention_saved | 300-600  | 1807203  | 0.2567   |
| original_weekly | train_attention_saved | 600-1000 | 702750   | 0.0998   |
| original_weekly | train_attention_saved | >1000    | 274643   | 0.039    |
| original_weekly | test_gp_saved         | SWE=0    | 385158   | 0.2078   |
| original_weekly | test_gp_saved         | 0-100    | 210066   | 0.1133   |
| original_weekly | test_gp_saved         | 100-300  | 596352   | 0.3218   |
| original_weekly | test_gp_saved         | 300-600  | 444384   | 0.2398   |
| original_weekly | test_gp_saved         | 600-1000 | 166561   | 0.0899   |
| original_weekly | test_gp_saved         | >1000    | 50919    | 0.0275   |
| planB_24week    | train_attention_saved | SWE=0    | 21054788 | 0.5197   |
| planB_24week    | train_attention_saved | 0-100    | 2603322  | 0.0643   |
| planB_24week    | train_attention_saved | 100-300  | 6495229  | 0.1603   |
| planB_24week    | train_attention_saved | 300-600  | 6379166  | 0.1575   |
| planB_24week    | train_attention_saved | 600-1000 | 2769255  | 0.0684   |
| planB_24week    | train_attention_saved | >1000    | 1211776  | 0.0299   |
| planB_24week    | test_gp_saved         | SWE=0    | 25778    | 0.4196   |
| planB_24week    | test_gp_saved         | 0-100    | 3798     | 0.0618   |
| planB_24week    | test_gp_saved         | 100-300  | 11878    | 0.1933   |
| planB_24week    | test_gp_saved         | 300-600  | 12648    | 0.2059   |
| planB_24week    | test_gp_saved         | 600-1000 | 5523     | 0.0899   |
| planB_24week    | test_gp_saved         | >1000    | 1815     | 0.0295   |

## Conditional bias by actual SWE bin

| dataset         | stage     | source                | swe_bin  | count   | mean_error | relative_bias | rmse     | mae     | corr  | slope |
| --------------- | --------- | --------------------- | -------- | ------- | ---------- | ------------- | -------- | ------- | ----- | ----- |
| original_weekly | attention | train_attention_saved | SWE=0    | 1202939 | 14.479     |               | 42.433   | 14.523  |       |       |
| original_weekly | attention | train_attention_saved | 0-100    | 727370  | 18.974     | 0.338         | 58.417   | 39.857  | 0.33  | 0.649 |
| original_weekly | attention | train_attention_saved | 100-300  | 2324071 | -9.519     | -0.048        | 73.742   | 55.744  | 0.55  | 0.847 |
| original_weekly | attention | train_attention_saved | 300-600  | 1807203 | -66.405    | -0.157        | 128.444  | 96.756  | 0.479 | 0.698 |
| original_weekly | attention | train_attention_saved | 600-1000 | 702750  | -163.494   | -0.216        | 212.697  | 173.252 | 0.514 | 0.714 |
| original_weekly | attention | train_attention_saved | >1000    | 274643  | -307.162   | -0.233        | 351.222  | 310.596 | 0.842 | 0.837 |
| original_weekly | gp_mean   | test_gp_saved         | SWE=0    | 385158  | 14.337     |               | 44.012   | 14.424  |       |       |
| original_weekly | gp_mean   | test_gp_saved         | 0-100    | 210066  | 27.267     | 0.501         | 67.199   | 45.813  | 0.358 | 0.794 |
| original_weekly | gp_mean   | test_gp_saved         | 100-300  | 596352  | 11.806     | 0.06          | 85.106   | 66.766  | 0.525 | 0.924 |
| original_weekly | gp_mean   | test_gp_saved         | 300-600  | 444384  | -29.79     | -0.07         | 135.124  | 102.812 | 0.437 | 0.759 |
| original_weekly | gp_mean   | test_gp_saved         | 600-1000 | 166561  | -116.95    | -0.155        | 222.001  | 161.874 | 0.377 | 0.692 |
| original_weekly | gp_mean   | test_gp_saved         | >1000    | 50919   | -338.596   | -0.262        | 496.93   | 359.315 | 0.34  | 0.402 |
| planB_24week    | attention | test_attention_saved  | SWE=0    | 25778   | 79.232     |               | 187.09   | 118.451 |       |       |
| planB_24week    | attention | test_attention_saved  | 0-100    | 3798    | 45.971     | 0.978         | 140.259  | 88.472  | 0.019 | 0.081 |
| planB_24week    | attention | test_attention_saved  | 100-300  | 11878   | -99.982    | -0.485        | 168.988  | 142.46  | 0.078 | 0.18  |
| planB_24week    | attention | test_attention_saved  | 300-600  | 12648   | -274.737   | -0.639        | 330.365  | 293.234 | 0.107 | 0.22  |
| planB_24week    | attention | test_attention_saved  | 600-1000 | 5523    | -543.211   | -0.718        | 595.907  | 552.572 | 0.074 | 0.153 |
| planB_24week    | attention | test_attention_saved  | >1000    | 1815    | -984.983   | -0.76         | 1053.085 | 990.312 | 0.201 | 0.207 |
| planB_24week    | gp_mean   | test_gp_saved         | SWE=0    | 25778   | 102.242    |               | 153.925  | 109.853 |       |       |
| planB_24week    | gp_mean   | test_gp_saved         | 0-100    | 3798    | 140.349    | 2.987         | 193.024  | 146.159 | 0.074 | 0.317 |
| planB_24week    | gp_mean   | test_gp_saved         | 100-300  | 11878   | 53.256     | 0.258         | 129.182  | 95.452  | 0.294 | 0.641 |
| planB_24week    | gp_mean   | test_gp_saved         | 300-600  | 12648   | -20.512    | -0.048        | 136.979  | 105.476 | 0.369 | 0.622 |
| planB_24week    | gp_mean   | test_gp_saved         | 600-1000 | 5523    | -176.801   | -0.234        | 250.997  | 211.054 | 0.306 | 0.498 |
| planB_24week    | gp_mean   | test_gp_saved         | >1000    | 1815    | -447.891   | -0.346        | 538.748  | 469.101 | 0.457 | 0.447 |

## Lead-dependent bias

| dataset         | stage     | source                | lead | mean_actual | mean_pred | mean_error | relative_bias | rmse    | mae     | slope  | intercept | r2    |
| --------------- | --------- | --------------------- | ---- | ----------- | --------- | ---------- | ------------- | ------- | ------- | ------ | --------- | ----- |
| original_weekly | attention | train_attention_saved | 1    | 310.989     | 275.404   | -35.586    | -0.114        | 131.68  | 83.883  | 0.768  | 36.588    | 0.835 |
| original_weekly | attention | train_attention_saved | 2    | 309.738     | 274.102   | -35.636    | -0.115        | 117.255 | 73.158  | 0.79   | 29.269    | 0.88  |
| original_weekly | attention | train_attention_saved | 3    | 306.329     | 238.086   | -68.243    | -0.223        | 136.016 | 84.326  | 0.712  | 19.936    | 0.899 |
| original_weekly | attention | train_attention_saved | 4    | 301.107     | 264.315   | -36.792    | -0.122        | 120.399 | 75.683  | 0.748  | 39.017    | 0.894 |
| original_weekly | gp_mean   | test_gp_saved         | 1    | 279.146     | 260.368   | -18.778    | -0.067        | 151.416 | 90.695  | 0.728  | 57.264    | 0.722 |
| original_weekly | gp_mean   | test_gp_saved         | 2    | 277.387     | 255.941   | -21.447    | -0.077        | 138.714 | 78.989  | 0.775  | 41.041    | 0.773 |
| original_weekly | gp_mean   | test_gp_saved         | 3    | 273.725     | 260.897   | -12.828    | -0.047        | 127.9   | 72.191  | 0.781  | 47.053    | 0.809 |
| original_weekly | gp_mean   | test_gp_saved         | 4    | 268.144     | 252.851   | -15.293    | -0.057        | 129.674 | 73.088  | 0.78   | 43.684    | 0.809 |
| planB_24week    | attention | test_attention_saved  | 1    | 305.808     | 381.039   | 75.231     | 0.246         | 139.957 | 105.19  | 0.972  | 83.938    | 0.755 |
| planB_24week    | attention | test_attention_saved  | 2    | 327.084     | 245.847   | -81.236    | -0.248        | 148.249 | 103.759 | 0.561  | 62.335    | 0.772 |
| planB_24week    | attention | test_attention_saved  | 3    | 352.229     | 95.597    | -256.631   | -0.729        | 323.06  | 257.852 | 0.258  | 4.842     | 0.595 |
| planB_24week    | attention | test_attention_saved  | 4    | 377.401     | -29.007   | -406.408   | -1.077        | 498.396 | 406.483 | -0.078 | 0.299     | 0.607 |
| planB_24week    | attention | test_attention_saved  | 5    | 399.974     | -42.541   | -442.515   | -1.106        | 553.278 | 443.949 | -0.148 | 16.627    | 0.458 |
| planB_24week    | attention | test_attention_saved  | 6    | 410.665     | -17.473   | -428.139   | -1.043        | 527.779 | 428.139 | -0.016 | -10.847   | 0.36  |
| planB_24week    | attention | test_attention_saved  | 7    | 406.148     | 125.551   | -280.597   | -0.691        | 367.57  | 284.547 | 0.259  | 20.225    | 0.677 |
| planB_24week    | attention | test_attention_saved  | 8    | 418.511     | 130.012   | -288.498   | -0.689        | 385.973 | 295.304 | 0.236  | 31.434    | 0.728 |
| planB_24week    | attention | test_attention_saved  | 9    | 412.479     | 26.062    | -386.418   | -0.937        | 500.972 | 387.279 | 0.063  | 0.271     | 0.637 |
| planB_24week    | attention | test_attention_saved  | 10   | 399.989     | 214.173   | -185.816   | -0.465        | 300.868 | 209.695 | 0.38   | 62.032    | 0.728 |
| planB_24week    | attention | test_attention_saved  | 11   | 386.866     | 195.546   | -191.32    | -0.495        | 336.458 | 232.207 | 0.272  | 90.373    | 0.755 |
| planB_24week    | attention | test_attention_saved  | 12   | 340.41      | 102.265   | -238.145   | -0.7          | 387.744 | 257.408 | 0.18   | 40.919    | 0.704 |
| planB_24week    | attention | test_attention_saved  | 13   | 312.261     | 127.066   | -185.195   | -0.593        | 343.512 | 222.424 | 0.229  | 55.511    | 0.691 |
| planB_24week    | attention | test_attention_saved  | 14   | 245.17      | 204.689   | -40.481    | -0.165        | 242.773 | 161.551 | 0.343  | 120.685   | 0.644 |
| planB_24week    | attention | test_attention_saved  | 15   | 191.308     | 262.622   | 71.314     | 0.373         | 211.93  | 160.473 | 0.489  | 169.054   | 0.614 |
| planB_24week    | attention | test_attention_saved  | 16   | 163.571     | 142.761   | -20.81     | -0.127        | 217.251 | 136.213 | 0.308  | 92.421    | 0.534 |
| planB_24week    | attention | test_attention_saved  | 17   | 118.985     | -8.989    | -127.974   | -1.076        | 275.974 | 128.15  | 0.016  | -10.858   | 0.176 |
| planB_24week    | attention | test_attention_saved  | 18   | 65.866      | 229.129   | 163.262    | 2.479         | 234.066 | 186.484 | 0.513  | 195.345   | 0.318 |
| planB_24week    | attention | test_attention_saved  | 19   | 35.363      | 16.538    | -18.824    | -0.532        | 148.81  | 51.967  | -0.012 | 16.98     | 0.006 |
| planB_24week    | attention | test_attention_saved  | 20   | 17.695      | 226.649   | 208.954    | 11.809        | 261.545 | 213.099 | 0.617  | 215.733   | 0.153 |
| planB_24week    | attention | test_attention_saved  | 21   | 7.931       | -183.73   | -191.66    | -24.167       | 268.408 | 192.965 | -0.599 | -178.979  | 0.081 |
| planB_24week    | attention | test_attention_saved  | 22   | 2.964       | 117.886   | 114.922    | 38.777        | 135.955 | 117.827 | 0.271  | 117.083   | 0.045 |
| planB_24week    | attention | test_attention_saved  | 23   | 1.488       | 329.741   | 328.253    | 220.558       | 394.765 | 328.253 | 1.118  | 328.078   | 0.028 |
| planB_24week    | attention | test_attention_saved  | 24   | 0.571       | -16.212   | -16.783    | -29.417       | 26.429  | 16.783  | -0.046 | -16.186   | 0.013 |
| planB_24week    | gp_mean   | test_gp_saved         | 1    | 305.808     | 356.794   | 50.986     | 0.167         | 111.771 | 88.551  | 0.723  | 135.645   | 0.787 |
| planB_24week    | gp_mean   | test_gp_saved         | 2    | 327.084     | 430.691   | 103.607    | 0.317         | 163.561 | 127.571 | 1.005  | 101.973   | 0.773 |
| planB_24week    | gp_mean   | test_gp_saved         | 3    | 352.229     | 302.329   | -49.899    | -0.142        | 145.725 | 103.269 | 0.717  | 49.649    | 0.71  |
| planB_24week    | gp_mean   | test_gp_saved         | 4    | 377.401     | 392.328   | 14.926     | 0.04          | 130.762 | 94.673  | 0.693  | 130.772   | 0.772 |
| planB_24week    | gp_mean   | test_gp_saved         | 5    | 399.974     | 276.922   | -123.051   | -0.308        | 220.903 | 148.728 | 0.405  | 115.079   | 0.747 |
| planB_24week    | gp_mean   | test_gp_saved         | 6    | 410.665     | 383.77    | -26.895    | -0.065        | 159.467 | 110.07  | 0.679  | 105.098   | 0.737 |
| planB_24week    | gp_mean   | test_gp_saved         | 7    | 406.148     | 324.467   | -81.681    | -0.201        | 198.585 | 143.129 | 0.683  | 47.021    | 0.663 |
| planB_24week    | gp_mean   | test_gp_saved         | 8    | 418.511     | 386.553   | -31.958    | -0.076        | 174.785 | 120.06  | 0.587  | 140.911   | 0.773 |

## Station-group bias files
- `station_group_bias_by_lead.csv`
- `station_group_bias_by_swe_bin.csv`
- `station_peak_groups.csv`

## Bias before vs after GP

| dataset         | stage     | source                | low_bin_mean_error_mm | high_bin_mean_error_mm | mean_lead_slope | max_lead_bias_mm | min_lead_bias_mm |
| --------------- | --------- | --------------------- | --------------------- | ---------------------- | --------------- | ---------------- | ---------------- |
| original_weekly | attention | train_attention_saved | 18.974                | -307.162               | 0.755           | -35.586          | -68.243          |
| original_weekly | gp_mean   | test_gp_saved         | 27.267                | -338.596               | 0.766           | -12.828          | -21.447          |
| planB_24week    | attention | test_attention_saved  | 45.971                | -984.983               | 0.258           | 328.253          | -442.515         |
| planB_24week    | gp_mean   | test_gp_saved         | 140.349               | -447.891               | 0.56            | 146.495          | -123.051         |

Interpretation from the saved outputs:
- In both datasets, the attention head already shows positive mean error in low-SWE bins and negative mean error in high-SWE bins, so the regression-to-the-mean behavior starts before the GP.
- The GP mean usually keeps the same sign pattern and often tightens the slope further toward 1 only weakly, which is consistent with the GP acting as a smooth multitask shrinkage layer rather than fixing the conditional bias.
- Because the attention loss is unweighted raw-scale MSE, the model is optimized for average squared error over a highly imbalanced target distribution with many zeros and many low-SWE samples. That setup gives much more statistical leverage to the abundant small-to-medium targets than to the rare large-SWE cases.
- The output head is linear and unconstrained, so the evidence points more strongly to loss/distribution effects than to a hard architectural saturation.

## Likely causes ranked by evidence
1. Unweighted raw-scale MSE on a heavily imbalanced SWE target distribution.
2. Shared single head over all stations and leads encourages average behavior across many low-SWE and zero-SWE cases.
3. Multitask variational GP smoothing preserves and can slightly reinforce the attention head's mean-regression pattern.
4. Lack of target transform or anomaly decomposition means high-end SWE is learned on the same linear scale as the dominant low-SWE mass.
5. The shared dataloader's rain/snow overwrite bug likely reduces the quality of accumulation-related predictors and can amplify conservative predictions, although the saved outputs alone do not isolate how much of the conditional bias it explains.

## Recommended next model change

Most justified next change from the code and saved outputs: **Option 1 plus Option 2, starting with a weighted loss on the attention head**.
- Weighted loss is the smallest code change because it only touches the attention training objective and leaves the data interface and GP stage unchanged.
- The saved outputs already show the sign pattern expected from underweighting rare large-SWE targets and overweighting abundant low-SWE targets under plain MSE.
- A second step worth testing immediately after weighted MSE is a `log1p(SWE)` or anomaly-style target, because the current training never rescales the raw target distribution.

## Exact next experiment command

```bash
cd /global/homes/h/hyvchen/Snow-Predication-at-Sierra
# Recommended next change: weighted attention loss before the GP
python experiments/foreswe_planB_24week/planB_24week_att_model.py
```

Suggested implementation detail for that next experiment: replace `nn.MSELoss()` with unreduced MSE and multiply by a weight such as `1 + alpha * clamp(actual / 300, 0, cap)` before averaging.
