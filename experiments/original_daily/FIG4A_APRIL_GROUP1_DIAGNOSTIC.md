# Figure 4(a) April Group 1 Diagnostic

## Scope

This diagnostic investigates why the reproduced ForeSWE Figure 4(a) shows April Group 1 relative bias as negative or near zero at short horizons, while the paper figure appears mostly positive.

No new final figure was generated in this step.

## Grouping Verification

- Current diagnostic grouping uses averaged yearly max SWE across training water years 1991-2014.
- Thresholds are the 30th, 60th, and 90th percentiles of station-wise averaged yearly max SWE.
- Group counts: {1: 154, 2: 153, 3: 153, 4: 52}.
- Percentile cuts (mm): [332.031, 511.493, 912.336].
- Group 1: count 154, averaged max SWE range 46.355-331.047 mm.
- Group 2: count 153, averaged max SWE range 334.328-511.175 mm.
- Group 3: count 153, averaged max SWE range 511.704-896.620 mm.
- Group 4: count 52, averaged max SWE range 914.082-2062.903 mm.

This matches the paper-style setup much better than equal-size quartile grouping, so the current mismatch is unlikely to be caused by the station grouping itself.

## Test-Year Labeling Evidence

- The released code names daily outputs by the December start year, not the ending water year.
- Therefore `2014_daily_gp_window_predict.pkl` corresponds to the test season from `2014-12-01` through `2015-05-30`, which is effectively the paper's WY2015 test season.
- Evidence:
  - `thirdparties/SWE-Forecasting/fore_swe/extended_dl_forecasting.py:154-158`: Each test split spans year_i-12-01 through year_i+1-05-30.
  - `thirdparties/SWE-Forecasting/fore_swe/extended_dl_forecasting.py:449-453`: The five released daily test seasons are labeled 2014, 2015, 2016, 2017, 2018.
  - `thirdparties/SWE-Forecasting/fore_swe/forecasting_att_daily_model.py:618-622`: Daily window actual files are dumped using 2014-2018 names.
  - `thirdparties/SWE-Forecasting/fore_swe/forecasting_daily_gp_model.py:244-248`: Daily GP window prediction files are dumped using 2014-2018 names.
  - `thirdparties/SWE-Forecasting/transformer_ar/helper_functions.py:49-54`: Future outputs use indices index+1 through index+window, so horizon h targets origin_date + h days.
  - `thirdparties/SWE-Forecasting/fore_swe/forecasting_att_daily_model.py:560-565`: The selected daily path advances origins in steps of forecasting_window=10, creating a non-overlapping daily-alignment variant.
  - `thirdparties/SWE-Forecasting/fore_swe/forecasting_att_daily_model.py:316-321`: getOutputs also uses index+1 through index+window, consistent with target_date = origin_date + horizon_days.

## Date Alignment

- Exact origin dates were recovered from the released `DataLoader` rather than guessed from array length.
- The recovered daily test spans are:
  - `2014`: 2014-12-01 to 2015-05-30 with 181 origins (Dec 31, Jan 31, Feb 28, Mar 31, Apr 30, May 30).
  - `2015`: 2015-12-01 to 2016-05-30 with 182 origins (Dec 31, Jan 31, Feb 29, Mar 31, Apr 30, May 30).
  - `2016`: 2016-12-01 to 2017-05-30 with 180 origins (Dec 30, Jan 31, Feb 28, Mar 31, Apr 30, May 30).
  - `2017`: 2017-12-01 to 2018-05-30 with 181 origins (Dec 31, Jan 31, Feb 28, Mar 31, Apr 30, May 30).
  - `2018`: 2018-12-01 to 2019-05-30 with 181 origins (Dec 31, Jan 31, Feb 28, Mar 31, Apr 30, May 30).
- `getFutureOutputs` and `getOutputs` both use `index+1` through `index+window`, so horizon `h` maps to `origin_date + h days`, not `origin_date + h - 1`.

The four April conventions tested here are:
- A: old naive month slicing by raw window index (`0:28`, `28:59`, `59:89`, `89:`).
- B: April defined by target date, `origin_date + horizon_days` in April.
- C: April defined by prediction origin date in April.
- D: target-date April using only non-overlapping origins `0, 10, 20, ...`, matching the released selected-daily stepping logic.

## April Group 1: Month-Slicing Comparison

Mean and median values below are averages across the five released test seasons for each horizon.

| Convention | Horizon (days) | Avg pooled mean bias (%) | Avg pooled median bias (%) |
| --- | ---: | ---: | ---: |
| A_old_naive_window_index | 1 | 4.468 | -2.349 |
| A_old_naive_window_index | 3 | 3.890 | -3.608 |
| A_old_naive_window_index | 5 | 2.446 | -5.799 |
| A_old_naive_window_index | 7 | 2.512 | -6.992 |
| A_old_naive_window_index | 10 | 1.727 | -9.447 |
| B_target_date_april | 1 | 0.830 | -7.391 |
| B_target_date_april | 3 | 19.894 | -1.042 |
| B_target_date_april | 5 | 32.034 | -1.213 |
| B_target_date_april | 7 | 57.678 | 4.293 |
| B_target_date_april | 10 | 76.481 | 6.117 |
| C_origin_date_april | 1 | 1.028 | -7.530 |
| C_origin_date_april | 3 | 20.853 | -1.082 |
| C_origin_date_april | 5 | 35.217 | -0.454 |
| C_origin_date_april | 7 | 66.982 | 7.391 |
| C_origin_date_april | 10 | 107.498 | 14.240 |
| D_selected_nonoverlap_target_april | 1 | -3.632 | -7.495 |
| D_selected_nonoverlap_target_april | 3 | 21.261 | -0.675 |
| D_selected_nonoverlap_target_april | 5 | 49.651 | -0.050 |
| D_selected_nonoverlap_target_april | 7 | 55.309 | 4.347 |
| D_selected_nonoverlap_target_april | 10 | 66.376 | 5.263 |

Interpretation:
- Convention A is strongly misaligned with the actual calendar because the arrays start in December; it should not be used for paper-style month panels.
- Convention B and C are the most relevant choices for Figure 4(a).
- For target-date April, the average pooled median bias at 1/3/5 days is -7.391, -1.042, and -1.213.
- For origin-date April, the corresponding values are -7.530, -1.082, and -0.454.
- This changes the magnitude somewhat, especially at 5+ days, but does not flip the short-horizon sign from negative to clearly positive.
- Convention D tests the non-overlapping selected-daily stepping logic. It stays close to the target-date result and does not resolve the early-horizon mismatch.

## Aggregation Diagnostic

For target-date April only, the script compares pooled sample-level bias against station-level aggregation.
- Pooled sample-level bias uses every valid station-day-horizon sample.
- Station-level aggregation first summarizes each station's April bias across days, then aggregates the station summaries.

This matters because pooled aggregation can overweight stations with more valid April days, while station-level aggregation can shift the median if low-SWE stations have sparse but extreme values.
- At 3 days, the average pooled median bias is -1.042, while the average median across station-level mean biases is 9.600.
- At 5 days, the pooled median is -1.213, while the station-level mean-bias median is 16.944 and the station-level median-bias median is 0.341.
- This is the strongest signal in the diagnostics that a paper-side aggregation or boxplot-construction difference could explain the more positive-looking April Group 1 panel.

## Low-SWE Threshold Diagnostic

Threshold sensitivity below is computed for target-date April.

| Threshold (mm) | Horizon (days) | Avg pooled median bias (%) |
| ---: | ---: | ---: |
| 0 | 1 | -7.391 |
| 0 | 3 | -1.042 |
| 0 | 5 | -1.213 |
| 0 | 7 | 4.293 |
| 0 | 10 | 6.117 |
| 1 | 1 | -7.391 |
| 1 | 3 | -1.042 |
| 1 | 5 | -1.213 |
| 1 | 7 | 4.293 |
| 1 | 10 | 6.117 |
| 10 | 1 | -7.777 |
| 10 | 3 | -1.423 |
| 10 | 5 | -1.628 |
| 10 | 7 | 3.476 |
| 10 | 10 | 4.504 |
| 25 | 1 | -8.268 |
| 25 | 3 | -1.722 |
| 25 | 5 | -2.122 |
| 25 | 7 | 2.472 |
| 25 | 10 | 3.044 |
| 50 | 1 | -8.262 |
| 50 | 3 | -2.062 |
| 50 | 5 | -2.658 |
| 50 | 7 | 1.485 |
| 50 | 10 | 1.693 |

- Raising the threshold from 1 mm to 50 mm moves the average 1-day pooled median bias from -7.391 to -8.262.
- At 5 days, the same threshold change moves the average pooled median from -1.213 to -2.658.
- Higher thresholds make April Group 1 slightly more negative at short horizons, so low-SWE filtering does not explain the paper's positive-looking panel.

## Main Takeaway

The station grouping is already consistent with the paper, and the 2014-2018 file names are just December-start labels for the paper's 2015-2019 test seasons.
Target-date April versus origin-date April changes the short-horizon April Group 1 medians only modestly, and the non-overlapping selected-daily alignment also stays in the same regime.
Changing the low-SWE threshold does not help; it pushes the short-horizon medians slightly more negative.
The strongest remaining explanation is an aggregation mismatch: if the paper's Figure 4(a) boxes are based on station-level April bias summaries rather than pooled station-day samples, April Group 1 can shift from slightly negative to around zero or positive at 3-5 day horizons.
That still does not fully explain the 1-day sign, so there may be an additional convention difference in the paper, but aggregation choice is the leading explanation from the evidence collected here.

## Output Files

- CSV: `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_reproduction/figures/fig4a_april_group1_diagnostic.csv`
- This report: `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_reproduction/FIG4A_APRIL_GROUP1_DIAGNOSTIC.md`

For full per-year values, see the CSV.
