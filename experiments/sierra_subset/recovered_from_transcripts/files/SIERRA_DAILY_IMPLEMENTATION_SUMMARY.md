
# Sierra Daily ForeSWE Implementation Summary

## Scope

- Implemented an isolated Sierra / near-Sierra 43-station adaptation of the released daily ForeSWE pipeline under:
  - `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_sierra_station`
- Preserved the reproduced 512-station baseline:
  - no files were modified inside `thirdparties/SWE-Forecasting`
  - no original third-party data, results, or outputs were overwritten
- No training was run.

## Files Copied

- `scripts/forecasting_att_daily_model.py`
- `scripts/forecasting_daily_gp_model.py`
- `scripts/extended_dl_forecasting.py`
- `scripts/custom_encoder_model.py`

## Files Modified

- `scripts/forecasting_att_daily_model.py`
- `scripts/forecasting_daily_gp_model.py`
- `scripts/extended_dl_forecasting.py`

## Files Added

- `data_subset/snotel_location_info_2.csv`
- `data_subset/all_stations_day_length.csv`
- `data_subset/precipitation_info.csv`
- `data_subset/rain_info.csv`
- `data_subset/rmax_info.csv`
- `data_subset/rmin_info.csv`
- `data_subset/snow_info.csv`
- `data_subset/sph_info.csv`
- `data_subset/srad_info.csv`
- `data_subset/swe_values.csv`
- `data_subset/tb_19_info.csv`
- `data_subset/tb_37_info.csv`
- `data_subset/tb_diff_info.csv`
- `data_subset/tmax_info.csv`
- `data_subset/tmin_info.csv`
- `data_subset/windspeed_info.csv`
- `scripts/check_sierra_daily_inputs.py`
- `SIERRA_DATA_SUBSET_VALIDATION.md`

## Exact Minimal Modifications Made

### Subset data creation

- Created experiment subdirectories:
  - `data_subset/`
  - `scripts/`
  - `outputs/`
  - `results/actual/`
  - `results/predict/`
  - `logs/`
  - `checkpoints/`
- Wrote subsetted copies of all station-dependent daily CSV inputs using the 43-station order from:
  - `sierra_snotel_stati<truncated omitted_approx_tokens="427" />ded a writable local Hugging Face cache configuration in the copied data loader:
  - `HF_HOME`
  - `TRANSFORMERS_CACHE`
- This avoids using the original home cache path during Sierra experiment runs.

## Subset Data Files Created And Shapes

- `data_subset/snotel_location_info_2.csv`: `43 x 30`
- `data_subset/all_stations_day_length.csv`: `10305 x 44`
- `data_subset/precipitation_info.csv`: `10592 x 44`
- `data_subset/rain_info.csv`: `10305 x 44`
- `data_subset/rmax_info.csv`: `10592 x 44`
- `data_subset/rmin_info.csv`: `10592 x 44`
- `data_subset/snow_info.csv`: `10305 x 44`
- `data_subset/sph_info.csv`: `10592 x 44`
- `data_subset/srad_info.csv`: `10592 x 44`
- `data_subset/swe_values.csv`: `10592 x 44`
- `data_subset/tb_19_info.csv`: `10592 x 44`
- `data_subset/tb_37_info.csv`: `10588 x 44`
- `data_subset/tb_diff_info.csv`: `10588 x 44`
- `data_subset/tmax_info.csv`: `10592 x 44`
- `data_subset/tmin_info.csv`: `10592 x 44`
- `data_subset/windspeed_info.csv`: `10592 x 44`

## N_STATIONS

- Active Sierra subset station count: `43`

## Dry-Run Shape Check

- Script:
  - `scripts/check_sierra_daily_inputs.py`
- Log:
  - `logs/check_sierra_daily_inputs.log`
- Result:
  - passed

### Verified dry-run shapes

- metadata: `(43, 30)`
- distance matrix: `(43, 43)`
- angularity matrix: `(43, 43)`
- model inputs: `(10305, 43, 19)`
- model outputs: `(10305, 43, 1)`

## Remaining Blockers Before Training

- No data-shape blocker remains for the 43-station Sierra subset.
- Before actual training, expect some environment-side noise during `conda run` on this system:
  - temporary shell wrapper `ERROR:: command not found` messages
  - repeated `auth.munge` socket warnings
- These did not break the Sierra shape check, but they may still appear in future logs.
- If the prompt encoder is exercised during training, the SentenceTransformer model must remain available in the active environment/cache.
