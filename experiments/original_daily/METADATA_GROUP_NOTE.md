# ForeSWE grouping note

## Final paper-style interpretation

After checking the full paper and Appendix / Table 2 details, the correct grouping for the Figure 4 analysis is **not** plain station-wise mean SWE.

The paper-style grouping is based on:

- each station’s **maximum SWE in each training year**
- then the **average of those yearly maxima** across all training years
- then splitting stations at the **30th, 60th, and 90th percentiles**

This matches the Appendix wording:

- “Averaged Max SWE of locations across all training years”
- classified using the 30th, 60th, and 90th percentiles

## Files inspected

- Metadata:
  `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/thirdparties/SWE-Forecasting/data/snotel_location_info_2.csv`
- SWE observations:
  `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/thirdparties/SWE-Forecasting/data/swe_values.csv`

## File structure and ordering

- `snotel_location_info_2.csv` shape: `(512, 29)`
- `swe_values.csv` shape: `(10592, 513)`
- `swe_values.csv` is in wide format:
  - one `Date` column
  - 512 station SWE columns
- Station ordering is consistent across:
  - `swe_values.csv`
  - `snotel_location_info_2.csv` via `Station Name`
  - the station dimension of the actual/prediction arrays

## Parsing and value handling

- Station columns were parsed numerically.
- No missing values were found after parsing.
- No obvious fill values such as `-9999`, `-999`, `9999`, or `99999` were found.
- No negative SWE values were found.
- Zeros were retained as valid SWE values.

## Final grouping method

1. Convert each date to a **water year**:
   - months `10-12` map to `year + 1`
   - months `1-9` map to `year`
2. Use training water years:
   - `1991` through `2014`
3. For each station and each training water year:
   - compute that year’s maximum SWE
4. For each station:
   - average those yearly maximum SWE values across training water years
5. Apply percentile thresholds:
   - 30th percentile
   - 60th percentile
   - 90th percentile
6. Assign:
   - `Group 1`: lowest band
   - `Group 2`: next band
   - `Group 3`: next band
   - `Group 4`: highest band

## Percentile cuts

Using averaged yearly max SWE over training water years `1991-2014`, the cut points are:

- 30th percentile: `332.031 mm`
- 60th percentile: `511.493 mm`
- 90th percentile: `912.336 mm`

## Group counts

The resulting counts are:

- `Group 1`: `154`
- `Group 2`: `153`
- `Group 3`: `153`
- `Group 4`: `52`

These match the paper’s expected counts exactly.

## Averaged max SWE ranges

Observed ranges from the reproduced data are:

- `Group 1`: `46.355` to `331.047 mm`
- `Group 2`: `334.328` to `511.175 mm`
- `Group 3`: `511.704` to `896.620 mm`
- `Group 4`: `914.082` to `2062.903 mm`

These align closely with the paper’s reported Table 2 ranges:

- Group 1: about `50-330 mm`
- Group 2: about `331-500 mm`
- Group 3: about `503-897 mm`
- Group 4: about `899-2004 mm`

## Important correction to the previous workflow

The previous plain-average-SWE grouping was **not** paper-style.

- It grouped by station-wise mean SWE over all rows in `swe_values.csv`.
- That workflow has been archived under:
  `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_reproduction/archive/plain_average_swe_grouping`

The current final workflow now uses the paper-style averaged-max-SWE grouping.

## Saved paper-style grouping artifacts

- Station grouping CSV:
  `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_reproduction/figures/paper_style_station_groups.csv`
- Confusion table versus the archived plain-average-SWE grouping:
  `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_reproduction/figures/grouping_plain_average_swe_vs_paper_style_confusion.csv`
