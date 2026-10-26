# ForeSWE Figure 4 validation with paper-style grouping

## Motivation

The final paper-style grouping for Figure 4 is based on **averaged yearly maximum SWE**, not plain station-wise mean SWE.

The earlier plain-average-SWE workflow was useful as a diagnostic, but it was not fully aligned with the Appendix / Table 2 setup. That diagnostic workflow has now been archived, and the final validation below uses the paper-style grouping:

- per-station yearly max SWE across training years
- average of those yearly maxima
- 30th / 60th / 90th percentile splits

## Corrected grouping method

- Source file:
  `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/thirdparties/SWE-Forecasting/data/swe_values.csv`
- Format:
  wide table with one `Date` column and 512 station SWE columns
- Parsing:
  numeric coercion on all station columns
- Missing-value handling:
  - no missing values found
  - no negative values found
  - no obvious fill values found
  - zeros retained as valid SWE values
- Training years used:
  water years `1991-2014`
- Metric:
  for each station, compute yearly max SWE in each training water year, then average those yearly maxima
- Percentiles used:
  `30th`, `60th`, `90th`

### Final cut points

- p30: `332.031 mm`
- p60: `511.493 mm`
- p90: `912.336 mm`

### Final group counts

- Group 1: `154`
- Group 2: `153`
- Group 3: `153`
- Group 4: `52`

### Final group ranges

- Group 1: `46.355` to `331.047 mm`
- Group 2: `334.328` to `511.175 mm`
- Group 3: `511.704` to `896.620 mm`
- Group 4: `914.082` to `2062.903 mm`

These match the paper’s Table 2 counts and ranges closely.

## Grouping artifacts

- Paper-style station groups:
  `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_reproduction/figures/paper_style_station_groups.csv`
- Confusion versus the archived plain-average-SWE grouping:
  `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_reproduction/figures/grouping_plain_average_swe_vs_paper_style_confusion.csv`

## Figure 4(a) final result

Outputs:

- PNG:
  `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_reproduction/figures/reproduced_fig4a_daily_relative_bias_paper_groups.png`
- PDF:
  `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_reproduction/figures/reproduced_fig4a_daily_relative_bias_paper_groups.pdf`
- Summary CSV:
  `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_reproduction/figures/reproduced_fig4a_bias_summary_paper_groups.csv`

Method notes:

- Exact test dates were recovered from the repository dataloader, so exact month masks were used.
- Relative bias formula:
  `100 * (predicted SWE - actual SWE) / actual SWE`
- Stability threshold:
  `actual SWE > 1 mm`

### Qualitative comparison

- February and March remain relatively tight, especially at shorter horizons.
- Spread grows in April and especially May.
- Longer horizons have larger spread almost everywhere.
- Groups 1 and 2 in May are still blocked in the paper style.
- Groups 3 and 4 in May show strong positive bias growth with horizon, which matches the paper’s late-season deterioration pattern.

### April Group 1 check

This was the key mismatch to revisit.

With the final paper-style grouping, April Group 1 median bias remains:

- 1 day: negative
- 3 days: near zero
- 5 days: near zero to weakly positive
- 7 days and 10 days: clearly positive

Examples:

- `2014`: `-5.10`, `0.16`, `1.32`, `11.30`, `23.81`
- `2018`: `-8.38`, `-0.90`, `-0.05`, `7.24`, `11.48`

So the paper-style grouping is clearly the correct grouping method, but it still **does not fully fix the April Group 1 mismatch** at the earliest horizons. Agreement improves mainly in methodological correctness, not in a dramatic visual flip of April Group 1.

## Figure 4(b) final result

Outputs:

- PNG:
  `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_reproduction/figures/reproduced_fig4b_daily_swe_availability_paper_groups.png`
- PDF:
  `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_reproduction/figures/reproduced_fig4b_daily_swe_availability_paper_groups.pdf`
- Summary CSV:
  `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_reproduction/figures/reproduced_fig4b_swe_summary_paper_groups.csv`

This final Figure 4(b) uses the 1-day prediction horizon as the paper-style main availability view. No extra “final” all-horizon variant is kept in the main figure outputs.

### Qualitative comparison

- SWE scale increases clearly from Group 1 to Group 4.
- Actual and predicted SWE distributions are generally close in February and March.
- April Groups 3 and 4 are very close in median SWE:
  - Group 3: actual `541.53 mm`, predicted `540.10 mm`
  - Group 4: actual `980.69 mm`, predicted `993.40 mm`
- May melt-out behavior appears clearly:
  - Group 1 actual median is `0 mm`
  - Group 2 has low remaining SWE
  - Groups 3 and 4 retain substantial SWE
- May Groups 3 and 4 show predicted SWE increasingly higher than actual as horizon increases, consistent with the Figure 4(a) late-horizon bias growth.

### April Group 1 in SWE space

For the final Figure 4(b):

- April Group 1 median actual SWE: `61.98 mm`
- April Group 1 median predicted SWE: `57.27 mm`
- Difference: `-4.71 mm`

So in absolute SWE, the April Group 1 distributions are reasonably close even though the relative-bias plot still looks pessimistic at short lead times.

## Archived plain-average-SWE workflow

The plain-average-SWE grouping workflow has been archived here:

- `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_reproduction/archive/plain_average_swe_grouping`

That archive includes:

- the non-paper-style figures
- the old CSV summaries
- the old logs
- the earlier average-SWE validation note

## Diagnostics

- May Group 3/4 horizon diagnostic CSV:
  `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_reproduction/diagnostics/fig4b_may_group34_horizon_diagnostic_paper_groups.csv`

## Limitations

- The paper-style grouping now matches the Appendix/Table 2 interpretation very closely, but some residual figure mismatch can still come from other differences in evaluation details.
- Relative bias remains sensitive when actual SWE is small, even with the `actual SWE > 1 mm` filter.
- The final goal here is validation of trend structure and grouping consistency, not pixel-perfect reproduction.
