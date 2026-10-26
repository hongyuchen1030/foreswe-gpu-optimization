import os

import joblib
import numpy as np
import pandas as pd


ROOT = os.environ.get(
    "FSW_WEEKLY_ROOT",
    "/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_weekly_reproduction_full512",
)
DATA_DIR = os.environ.get("FSW_DATA_DIR", os.path.join(ROOT, "data"))
RESULTS_ACTUAL_DIR = os.environ.get("FSW_RESULTS_ACTUAL", os.path.join(ROOT, "results", "actual"))

FORECASTING_WINDOW = 4
FORECASTING_STEP_SIZE = 7
TEST_YEARS = [2014, 2015, 2016, 2017, 2018]


def get_filtered_date_indices(all_dates, start_year=1991, start_month=12, start_day=1, end_month=5):
    filtered_dates = []
    filtered_date_indices = []

    for year in range(start_year, 2020):
        start_date = "{}-{:02d}-{:02d}".format(year, start_month, start_day)
        end_year = year + 1 if end_month < start_month else year
        end_date = "{}-{:02d}-30".format(end_year, end_month)
        indices = np.where(np.logical_and(all_dates >= start_date, all_dates <= end_date))[0]
        filtered_dates.append(all_dates[indices])
        filtered_date_indices.append(indices)

    return np.concatenate(filtered_dates), np.concatenate(filtered_date_indices)


def main():
    os.makedirs(RESULTS_ACTUAL_DIR, exist_ok=True)

    metadata = pd.read_csv(os.path.join(DATA_DIR, "snotel_location_info_2.csv"))
    station_names = list(metadata["Station Name"].values)

    date_files = [
        "precipitation_info.csv",
        "rmax_info.csv",
        "rmin_info.csv",
        "sph_info.csv",
        "srad_info.csv",
        "tb_19_info.csv",
        "tb_37_info.csv",
        "tb_diff_info.csv",
        "tmin_info.csv",
        "tmax_info.csv",
        "windspeed_info.csv",
        "swe_values.csv",
        "all_stations_day_length.csv",
    ]

    date_sets = []
    for fname in date_files:
        df = pd.read_csv(os.path.join(DATA_DIR, fname), usecols=["Date"])
        date_sets.append(set(df["Date"].values))

    included_dates = set.intersection(*date_sets)

    swe_values = pd.read_csv(os.path.join(DATA_DIR, "swe_values.csv"))
    swe_values = swe_values.loc[swe_values["Date"].isin(included_dates), ["Date"] + station_names].copy()
    swe_values = swe_values.sort_values("Date").reset_index(drop=True)

    all_dates = swe_values["Date"].to_numpy()
    swe_array = swe_values[station_names].to_numpy(dtype=np.float32)

    filtered_dates, filtered_date_indices = get_filtered_date_indices(all_dates)
    start_index = np.where(filtered_dates == "1995-12-01")[0][0]
    filtered_dates = filtered_dates[(start_index + 1):]
    filtered_date_indices = filtered_date_indices[(start_index + 1):]

    for year in TEST_YEARS:
        year_mask = np.logical_and(filtered_dates >= "{}-12-01".format(year), filtered_dates <= "{}-05-30".format(year + 1))
        year_indices = filtered_date_indices[np.where(year_mask)[0]]

        weekly_window_actual = []
        for input_index in year_indices:
            output_indices = sorted([input_index + (value * FORECASTING_STEP_SIZE) for value in range(1, FORECASTING_WINDOW + 1)])
            weekly_window_actual.append(swe_array[output_indices, :].T.reshape(1, len(station_names), FORECASTING_WINDOW))

        weekly_window_actual = np.concatenate(weekly_window_actual, axis=0)
        out_path = os.path.join(RESULTS_ACTUAL_DIR, "{}_weekly_window_actual.pkl".format(year))
        joblib.dump(weekly_window_actual, out_path)
        print(year, weekly_window_actual.shape, out_path)

    print("Lightweight weekly actual export complete.")


if __name__ == "__main__":
    main()
