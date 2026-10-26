# ForeSWE Figure 4 validation with average-SWE grouping

## Motivation

We needed an apple-to-apple grouping because the ForeSWE paper defines the four location groups by **average SWE**, from low to high. The previous local reproduction used **average annual max SWE** quartiles instead, which was not fully aligned with the paper and may have contributed to the Figure 4(a) mismatch, especially for April Group 1.

## Corrected grouping method

- Grouping file used:
  `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/thirdparties/SWE-Forecasting/data/swe_values.csv`
- Parsing method:
  - `Date` treated as the time column
  - all other columns treated as station SWE columns
  - wide format confirmed
- Missing-value handling:
  - numeric coercion applied
  - no missing values found
  - no obvious fill values found
  - zeros retained as valid SWE observations
- Average-SWE definition used:
  - station-wise arithmetic mean over all valid SWE observations in `swe_values.csv`
- Quartile cut points:
  - 25th percentile: `82.037 mm`
  - 50th percentile: `127.987 mm`
  - 75th percentile: `214.518 mm`
- Group counts:
  - Group 1: `128`
  - Group 2: `128`
  - Group 3: `128`
  - Group 4: `128`

Saved corrected grouping metadata:

- `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_reproduction/figures/average_swe_station_groups.csv`

Saved old-vs-new grouping comparison:

- `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_reproduction/figures/grouping_old_vs_average_swe_confusion.csv`

## Figure 4(a) corrected result

Outputs:

- Plot PNG:
  `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_reproduction/figures/reproduced_fig4a_daily_relative_bias_average_swe_groups.png`
- Plot PDF:
  `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_reproduction/figures/reproduced_fig4a_daily_relative_bias_average_swe_groups.pdf`
- Summary CSV:
  `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_reproduction/figures/reproduced_fig4a_bias_summary_average_swe_groups.csv`

Method notes:

- Exact test dates were recovered through the repository dataloader, so exact February-March-April-May month masks were used instead of the approximate slice fallback.
- Relative bias used:
  `100 * (predicted SWE - actual SWE) / actual SWE`
- To avoid unstable blow-ups, only samples with `actual SWE > 1 mm` were used for the bias boxplots and summary statistics.

### Qualitative comparison to the paper

- February and March:
  - bias remains relatively small overall
  - shorter horizons look tighter than longer horizons
- March Group 1:
  - still shows mild negative bias rather than a strong positive shift
- April Groups 1 and 2:
  - spread increases clearly relative to winter months
  - Group 2 shows stronger positive drift at longer horizons
- April Groups 3 and 4:
  - median relative bias is close to zero at shorter horizons
  - bias grows gradually with horizon
- May Groups 3 and 4:
  - relative bias increases strongly with horizon
  - this matches the paper’s qualitative “later horizons worse during melt season” pattern
- Group 1 and Group 2 May panels:
  - hatched/blocked in the paper style

### April Group 1 diagnosis

This was the main mismatch of interest.

With the corrected average-SWE grouping, April Group 1 median bias by horizon remained:

- 1 day: mostly negative across years
- 3 days: near zero to slightly negative
- 5 days: near zero, mixed sign
- 7 days and 10 days: clearly positive

Examples from the rerun:

- `2014`: median bias `-7.67`, `-0.92`, `-1.62`, `10.55`, `18.88`
- `2018`: median bias `-9.89`, `-0.08`, `1.73`, `11.01`, `16.44`

So the corrected grouping is more paper-faithful in definition, but it **does not materially improve the April Group 1 mismatch**. The early-horizon April Group 1 medians are still negative or near zero rather than clearly positive.

## Figure 4(b) result

Outputs:

- Main 1-day plot PNG:
  `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_reproduction/figures/reproduced_fig4b_daily_swe_availability_average_swe_groups.png`
- Main 1-day plot PDF:
  `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_reproduction/figures/reproduced_fig4b_daily_swe_availability_average_swe_groups.pdf`
- All-horizon sensitivity PNG:
  `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_reproduction/figures/reproduced_fig4b_daily_swe_availability_average_swe_groups_all_horizons.png`
- All-horizon sensitivity PDF:
  `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_reproduction/figures/reproduced_fig4b_daily_swe_availability_average_swe_groups_all_horizons.pdf`
- Summary CSV:
  `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_reproduction/figures/reproduced_fig4b_swe_summary_average_swe_groups.csv`

### Qualitative comparison

- SWE scale increases cleanly from Group 1 to Group 4.
- Actual and predicted SWE distributions are generally close in February and March.
- April actual and predicted medians are also fairly close for Groups 3 and 4:
  - Group 3 April medians: actual `406.91 mm`, predicted `398.60 mm`
  - Group 4 April medians: actual `741.17 mm`, predicted `745.65 mm`
- April Group 1 medians are also close in absolute SWE:
  - actual `30.99 mm`
  - predicted `31.10 mm`
- May melt-out behavior is visible:
  - Group 1 actual median is `0 mm`
  - Group 2 actual median is low (`12.19 mm`)
  - Groups 3 and 4 retain much larger SWE

### May Groups 3 and 4 horizon behavior

Using the horizon-by-horizon diagnostic:

- Group 3 predicted median SWE exceeds actual median SWE more strongly as horizon increases
- Group 4 shows the same pattern even more clearly
- For Group 4 in May, mean median difference grows from about `12.8 mm` at 1 day to about `171.3 mm` at 10 days

That horizon-dependent inflation is qualitatively consistent with the Figure 4(a) bias growth in late season.

## Limitations

- The corrected grouping is more faithful to the paper wording, but the exact averaging period used by the paper is still not fully specified.
- If the paper used an active-season-only or training-period-only average SWE, some residual grouping mismatch may remain.
- Relative bias remains sensitive when actual SWE is small, even with the `actual SWE > 1 mm` threshold.
- These plots are intended to validate the **trend structure** and grouping logic, not to provide a pixel-perfect reproduction of the published figure.
