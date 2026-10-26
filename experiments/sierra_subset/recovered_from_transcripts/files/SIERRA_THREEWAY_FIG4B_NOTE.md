
# Sierra Three-Way Figure 4b Note

This note explains how the three-way Sierra Figure 4b plot was generated and what it shows.

## Where the 43 stations are

The map below shows where the 43 Sierra / near-Sierra SNOTEL stations are located. The left panel gives western U.S. context, and the right panel zooms into the Sierra / near-Sierra subset region.

![Sierra station map](./figures/sierra_station_locations_map.png)

## Output figure

- PNG:
  `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_sierra_station/figures/sierra_fig4b_daily_swe_availability_threeway_paper_groups.png`
- PDF:
  `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_sierra_station/figures/sierra_fig4b_daily_swe_availability_threeway_paper_groups.pdf`

![Sierra three-way Figure 4b](./figures/sierra_fig4b_daily_swe_availability_threeway_paper_groups.png)

## What is being compared

Each subplot corresponds to one Sierra paper-style SWE group.

Within each month, the plot shows three boxplots:

- `Actual`: observed Sierra SWE
- `Predicted All-Stations`: predictions from the original ForeSWE model trained on the full 512-station western U.S. set
- `Predicted Sierra-Only`: predictions from the Sierra-only ForeSWE run using the 43-station Sierra / near-Sierra subset

The months shown are February, March, April, and May.

## Data sources used

Sierra actual and Sierra-only prediction files came from:

- `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_sierra_station/results/actual`
- `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_sierra_station/results/predict`

All-station prediction files came from:

- `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/thirdparties/SWE-Forecasting/results/predict`

The Sierra station metadata and station ordering came from:

- `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_sierra_station/data_subset/snotel_location_info_2.csv`

## How the plot was generated

The plotting script was:

- `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_sierra_station/scripts/plot_sierra_fig4b_threeway_paper_groups.py`

The grouping helper was:

- `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_sierra_station/scripts/foreswe_sierra_grouping_utils.py`

The workflow was:

1. Use the 43 Sierra stations only.
2. Assign the Sierra stations into 4 paper-style groups using the same method as the earlier Figure 4 reproduction.
3. For each Sierra station, use only the historical training period from water years 1991 to 2014 to compute its average yearly peak SWE.
4. Split stations into 4 groups using the 30th, 60th, and 90th percentiles of that average yearly peak SWE.
5. Load Sierra actual SWE and Sierra-only predicted SWE for test years 2014 to 2018.
6. Load the full 512-station prediction files from the original ForeSWE run.
7. Subset those full-run prediction arrays down to the same 43 Sierra stations using `original_station_index`.
8. Verify that Sierra station IDs and names match the original 512-station metadata exactly before plotting.
9. For each month and group, pool all day-by-station SWE values across the 2014 to 2018 test years.
10. Plot the pooled distributions as boxplots for the three series: actual, all-station prediction, and Sierra-only prediction.

Only the 1-day prediction horizon was used for this Figure 4b-style SWE availability plot, matching the earlier Figure 4b setup.

## Alignment check

The all-station predictions were mapped onto the Sierra station subset using the original 512-station ordering stored in the Sierra subset metadata.

Alignment check file:

- `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_sierra_station/figures/sierra_all_station_prediction_alignment_check.csv`

This check confirmed that all 43 Sierra stations matched correctly by both station ID and station name.

## What the plot shows

The main result is that, for this Sierra subset comparison, the predictions from the original all-station-trained ForeSWE model are much closer to the actual Sierra SWE distributions than the predictions from the Sierra-only-trained run.

Examples from the summary table:

- April Group 4 median SWE:
  actual `1172.72 mm`
  all-station prediction `1176.62 mm`
  Sierra-only prediction `366.27 mm`
- February Group 3 median SWE:
  actual `493.78 mm`
  all-station prediction `476.75 mm`
  Sierra-only prediction `149.42 mm`

So in this three-way Figure 4b comparison, the all-station model preserves the Sierra SWE magnitude much better, while the Sierra-only run strongly underpredicts SWE across most groups and months.

## Supporting summary file

- `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_sierra_station/figures/sierra_fig4b_swe_summary_threeway_paper_groups.csv`

