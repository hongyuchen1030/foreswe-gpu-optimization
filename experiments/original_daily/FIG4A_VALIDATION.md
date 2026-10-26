# ForeSWE Figure 4(a) validation

This report documents the reproduction workflow for a Figure 4(a)-style daily relative bias plot using the reproduced ForeSWE daily GP-head outputs.

## Metadata and grouping

- Metadata file inspected:
  `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/thirdparties/SWE-Forecasting/data/snotel_location_info_2.csv`
- The metadata CSV did not contain an explicit station-group column for the paper’s Group 1-4 breakdown.
- Grouping was therefore derived from `data/swe_values.csv`, following the paper’s average-annual-max-SWE grouping idea.
- Group labels were assigned directly as:
  - `Group 1`: lowest average-annual-max-SWE quartile
  - `Group 2`: second quartile
  - `Group 3`: third quartile
  - `Group 4`: highest quartile
- No `0-3` to `1-4` remapping was required.

See `METADATA_GROUP_NOTE.md` for the CSV columns, shape, and the derived SWE cut points.

## Loaded files

Actual files loaded from:

- `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/thirdparties/SWE-Forecasting/results/actual/2014_daily_window_actual.pkl`
- `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/thirdparties/SWE-Forecasting/results/actual/2015_daily_window_actual.pkl`
- `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/thirdparties/SWE-Forecasting/results/actual/2016_daily_window_actual.pkl`
- `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/thirdparties/SWE-Forecasting/results/actual/2017_daily_window_actual.pkl`
- `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/thirdparties/SWE-Forecasting/results/actual/2018_daily_window_actual.pkl`

Predicted GP files loaded from:

- `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/thirdparties/SWE-Forecasting/results/predict/2014_daily_gp_window_predict.pkl`
- `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/thirdparties/SWE-Forecasting/results/predict/2015_daily_gp_window_predict.pkl`
- `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/thirdparties/SWE-Forecasting/results/predict/2016_daily_gp_window_predict.pkl`
- `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/thirdparties/SWE-Forecasting/results/predict/2017_daily_gp_window_predict.pkl`
- `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/thirdparties/SWE-Forecasting/results/predict/2018_daily_gp_window_predict.pkl`

## Shapes by year

- `2014`: actual `(181, 512, 10)`, predicted `(181, 512, 10)`
- `2015`: actual `(182, 512, 10)`, predicted `(182, 512, 10)`
- `2016`: actual `(180, 512, 10)`, predicted `(180, 512, 10)`
- `2017`: actual `(181, 512, 10)`, predicted `(181, 512, 10)`
- `2018`: actual `(181, 512, 10)`, predicted `(181, 512, 10)`

All actual and predicted shapes matched.

## Relative bias definition

The script uses:

`relative bias (%) = 100 * (predicted SWE - actual SWE) / actual SWE`

## Minimum SWE threshold

- Minimum actual SWE threshold used: `actual SWE > 1.0 mm`
- Reason: this prevents unstable relative-bias values when actual SWE is zero or near zero.

## Month slicing

Exact month indexing was used instead of the approximate February-March-April-May slices.

- For each reproduced year label `Y`, exact dates were reconstructed over:
  `Y-12-01` through `Y+1-05-30`
- The script uses the repository’s own `transformer_ar` dataloader logic to recover the exact active-season test dates after feature-date intersection, rather than assuming every calendar day is present.
- Figure 4(a) aggregation then used only forecast windows whose start date fell in:
  - February
  - March
  - April
  - May
- Month counts used in the reproduced files were:
  - `2014`: Feb 28, Mar 31, Apr 30, May 30
  - `2015`: Feb 29, Mar 31, Apr 30, May 30
  - `2016`: Feb 28, Mar 31, Apr 30, May 30
  - `2017`: Feb 28, Mar 31, Apr 30, May 30
  - `2018`: Feb 28, Mar 31, Apr 30, May 30
- One active-season day is absent in the `2016` reproduced window set, which is why that file has `180` windows instead of `181`.

## Outputs

- Plot PNG:
  `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_reproduction/figures/reproduced_fig4a_daily_relative_bias.png`
- Plot PDF:
  `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_reproduction/figures/reproduced_fig4a_daily_relative_bias.pdf`
- Summary CSV:
  `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_reproduction/figures/reproduced_fig4a_bias_summary.csv`

## Qualitative comparison notes

Visual check of `reproduced_fig4a_daily_relative_bias.png` shows the following broad behavior:

- Bias in February and March stays relatively small overall, especially for the shorter `1-3` day horizons. February is mostly slightly negative, and March remains near zero to mildly negative for many panels.
- Bias increases in April and May, especially for longer horizons. April begins to show positive drift at `7-10` days, and May shows clear positive bias growth for Groups `2-4`.
- Later horizons have larger spread. The `10`-day boxplots are visibly wider than the `1`-day boxplots in nearly every panel, with the strongest widening in April and May.
- Group 1 and Group 2 May panels omitted/hatched: implemented in reproduction plot
- The reproduced figure appears to match the paper’s *qualitative* pattern reasonably well:
  - winter-month bias is comparatively tight,
  - spring melt months show larger spread,
  - longer horizons are less stable,
  - and the low-SWE May panels are blocked.
- The reproduction is not expected to be pixel-identical to the paper figure, and exact box positions/spreads should be treated as an approximate validation rather than a perfect visual replica.
