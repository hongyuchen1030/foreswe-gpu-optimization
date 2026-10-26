# D2 Domain Implementation Summary

## Scientific purpose

`D2_sierra_plus_500km` is the intermediate ForeSWE domain-size experiment between:

- `D0_sierra43`: Sierra / near-Sierra 43-station domain
- `D4_full512`: original full 512-station western-U.S. ForeSWE domain

The goal is to test how much nearby non-Sierra SNOTEL context helps Sierra SWE prediction while keeping the Sierra 43 stations as the evaluation target.

## D2 definition

`D2_sierra_plus_500km` contains every original 512-station ForeSWE SNOTEL station whose minimum haversine distance to any of the Sierra 43 evaluation stations is `<= 500 km`.

The D2 station order preserves the original 512-station ordering. Sierra evaluation stations are marked in:

- [station_subset.csv](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km/station_subset.csv)

## D2 station count

- Total D2 stations: `264`
- Sierra eval stations included: `43 / 43`

State distribution:

- `CA`: `25`
- `ID`: `48`
- `MT`: `14`
- `NV`: `23`
- `OR`: `67`
- `UT`: `63`
- `WA`: `13`
- `WY`: `11`

Distance-to-Sierra summary:

- minimum: `0.000 km`
- median: `344.004 km`
- maximum: `498.428 km`

Coordinate extent:

- latitude: `37.4863` to `46.8100`
- normalized longitude: `-123.3731` to `-110.1862`

## Experiment tree

Home root:

- [D2_sierra_plus_500km](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km)

Runtime root on `pscratch`:

- `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km`

Runtime-heavy directories live on `pscratch` and are linked into the home experiment folder:

- [outputs](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km/outputs)
- [logs](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km/logs)
- [checkpoints](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km/checkpoints)
- [results](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km/results)

## Data subset created

Metadata:

- [data_subset/snotel_location_info_2.csv](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km/data_subset/snotel_location_info_2.csv): `(264, 36)`

Station-timeseries CSVs:

- [all_stations_day_length.csv](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km/data_subset/all_stations_day_length.csv): `(10305, 265)`
- [precipitation_info.csv](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km/data_subset/precipitation_info.csv): `(10592, 265)`
- [rain_info.csv](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km/data_subset/rain_info.csv): `(10305, 265)`
- [rmax_info.csv](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km/data_subset/rmax_info.csv): `(10592, 265)`
- [rmin_info.csv](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km/data_subset/rmin_info.csv): `(10592, 265)`
- [snow_info.csv](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km/data_subset/snow_info.csv): `(10305, 265)`
- [sph_info.csv](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km/data_subset/sph_info.csv): `(10592, 265)`
- [srad_info.csv](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km/data_subset/srad_info.csv): `(10592, 265)`
- [swe_values.csv](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km/data_subset/swe_values.csv): `(10592, 265)`
- [tb_19_info.csv](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km/data_subset/tb_19_info.csv): `(10592, 265)`
- [tb_37_info.csv](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km/data_subset/tb_37_info.csv): `(10588, 265)`
- [tb_diff_info.csv](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km/data_subset/tb_diff_info.csv): `(10588, 265)`
- [tmax_info.csv](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km/data_subset/tmax_info.csv): `(10592, 265)`
- [tmin_info.csv](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km/data_subset/tmin_info.csv): `(10592, 265)`
- [windspeed_info.csv](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km/data_subset/windspeed_info.csv): `(10592, 265)`

All station-timeseries files preserve:

- `Date` as the first column
- D2 station order in original 512-station order
- strict station-column matching with no silent drops

## Scripts copied and adapted

Copied/adapted D2 scripts:

- [forecasting_att_daily_model.py](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km/scripts/forecasting_att_daily_model.py)
- [forecasting_daily_gp_model.py](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km/scripts/forecasting_daily_gp_model.py)
- [extended_dl_forecasting.py](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km/scripts/extended_dl_forecasting.py)
- [custom_encoder_model.py](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km/scripts/custom_encoder_model.py)
- [check_domain_daily_inputs.py](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km/scripts/check_domain_daily_inputs.py)

Minimal adaptations made:

- replaced Sierra-root env defaults with `FSW_DOMAIN_ROOT`
- added `FSW_STATION_SUBSET_CSV`
- kept station-count reshapes dynamic through active metadata `N_STATIONS`
- replaced `Expected 43` checks with checks against `station_subset.csv`
- kept model architecture widths such as hidden size `512` unchanged

## Dry-run shape check

Dry-run script:

- [check_domain_daily_inputs.py](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km/scripts/check_domain_daily_inputs.py)

Log:

- `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km/logs/check_domain_daily_inputs.log`

Result:

- `passed`

Key dry-run checks that passed:

- `N_STATIONS = 264`
- `Sierra eval station count = 43`
- metadata shape `(264, 36)`
- active SWE shape `(10305, 265)` after dataloader cleaning
- model input shape `(10305, 264, 19)`
- model output shape `(10305, 264, 1)`
- distance matrix `(264, 264)`
- angularity matrix `(264, 264)`

## Slurm scripts created

- [run_d2_att_daily_gpu.sbatch](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km/slurm/run_d2_att_daily_gpu.sbatch)
- [run_d2_gp_daily_gpu.sbatch](/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km/slurm/run_d2_gp_daily_gpu.sbatch)

Attention job notes:

- standard `40GB` GPU first
- pscratch logs and pscratch HF cache
- suppresses common tensor-copy warning spam via `PYTHONWARNINGS`

GP job notes:

- standard `40GB` GPU first
- if CUDA OOM occurs, retry with `-C "gpu&hbm80g"`

## Submit commands for later

Attention:

```bash
sbatch /global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km/slurm/run_d2_att_daily_gpu.sbatch
```

GP:

```bash
sbatch /global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km/slurm/run_d2_gp_daily_gpu.sbatch
```

## Warnings / blockers

- No training was run.
- No Slurm jobs were submitted.
- No original third-party ForeSWE code or data were modified.
- `transformers` emits a `TRANSFORMERS_CACHE` deprecation warning during dry-run, but this did not block the D2 check.
