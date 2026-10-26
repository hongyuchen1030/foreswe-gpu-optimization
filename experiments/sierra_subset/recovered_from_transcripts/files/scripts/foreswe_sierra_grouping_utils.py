
#!/usr/bin/env python3
"""Shared helpers for Sierra-only ForeSWE Figure 4 style plots."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


REPO_ROOT = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra")
THIRDPARTY_ROOT = REPO_ROOT / "thirdparties" / "SWE-Forecasting"
SIERRA_HOME_ROOT = Path(
    os.getenv(
        "FSW_HOME_SIERRA_ROOT",
        str(REPO_ROOT / "experiments" / "foreswe_sierra_station"),
    )
)
SIERRA_RUNTIME_ROOT = Path(
    os.getenv(
        "FSW_SIERRA_ROOT",
        "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_sierra_station",
    )
)
SIERRA_DATA_DIR = Path(os.getenv("FSW_DATA_DIR", str(SIERRA_HOME_ROOT / "data_subset")))
ACTUAL_DIR = Path(os.getenv("FSW_RESULTS_ACTUAL", str(SIERRA_RUNTIME_ROOT / "results" / "actual")))
PREDICT_DIR = Path(os.getenv("FSW_RESULTS_PREDICT", str(SIERRA_RUNTIME_ROOT / "results" / "predict")))
FIGURE_DIR = SIERRA_HOME_ROOT / "figures"
DIAGNOSTICS_DIR = SIERRA_HOME_ROOT / "diagnostics"

YEARS = [2014, 2015, 2016, 2017, 2018]
TRAINING_WATER_YEAR_START = 1991
TRAINING_WATER_YEAR_END = 2014
EXPECTED_STATION_COUNT = 43
EXPECTED_SHAPES = {
    2014: (181, 43, 10),
    2015: (182, 43, 10),
    2016: (180, 43, 10),
    2017: (181, 43, 10),
    2018: (181, 43, 10),
}
MONTHS = ["February", "March", "April", "May"]
MONTH_NUMBERS = {"February": 2, "March": 3, "April": 4, "May": 5}
HORIZON_LABELS = [1, 3, 5, 7, 10]
HORIZON_INDICES = [0, 2, 4, 6, 9]


def load_metadata() -> pd.DataFrame:
    metadata = pd.read_csv(SIERRA_DATA_DIR / "snotel_location_info_2.csv").reset_index()
    metadata = metadata.rename(columns={"index": "subset_station_index"})
    if len(metadata) != EXPECTED_STATION_COUNT:
        raise ValueError(f"Expected {EXPECTED_STATION_COUNT} Sierra stations, found {len(metadata)}")
    return metadata


def load_swe_wide() -> tuple[pd.DataFrame, list[str]]:
    swe_df = pd.read_csv(SIERRA_DATA_DIR / "swe_values.csv")
    station_cols = [col for col in swe_df.columns if col != "Date"]
    swe_df[station_cols] = swe_df[station_cols].apply(pd.to_numeric, errors="coerce")
    if len(station_cols) != EXPECTED_STATION_COUNT:
        raise ValueError(f"Expected {EXPECTED_STATION_COUNT} station columns in Sierra SWE CSV, found {len(station_cols)}")
    return swe_df, station_cols


def inspect_source_data() -> dict[str, object]:
    metadata = load_metadata()
    swe_df, station_cols = load_swe_wide()
    exact_order_match = metadata["Station Name"].tolist() == station_cols
    numeric_values = swe_df[station_cols].to_numpy()
    return {
        "metadata_shape": metadata.shape,
        "metadata_columns": metadata.columns.tolist(),
        "swe_shape": swe_df.shape,
        "station_count": len(station_cols),
        "station_order_matches": exact_order_match,
        "nan_count": int(np.isnan(numeric_values).sum()),
        "negative_count": int((numeric_values < 0).sum()),
        "zero_count": int((numeric_values == 0).sum()),
        "date_min": swe_df["Date"].iloc[0],
        "date_max": swe_df["Date"].iloc[-1],
    }


def _assign_percentile_groups(values: np.ndarray, percentiles: list[float]) -> tuple[np.ndarray, np.ndarray]:
    percentile_cuts = np.percentile(values, percentiles)
    labels = np.digitize(values, percentile_cuts, right=True) + 1
    return labels.astype(int), percentile_cuts


def load_plain_average_swe_groups() -> tuple[np.ndarray, pd.DataFrame]:
    metadata = load_metadata()
    swe_df, station_cols = load_swe_wide()
    average_swe = swe_df[station_cols].mean(axis=0, skipna=True).to_numpy(dtype=float)
    groups, percentile_cuts = _assign_percentile_groups(average_swe, [25, 50, 75])

    group_df = metadata.copy()
    group_df["station_name_from_swe"] = station_cols
    group_df["average_swe"] = average_swe
    group_df["average_swe_group"] = groups
    group_df["average_swe_q25_q50_q75_mm"] = ",".join(f"{cut:.3f}" for cut in percentile_cuts)
    return groups, group_df


def load_paper_style_groups() -> tuple[np.ndarray, pd.DataFrame]:
    metadata = load_metadata()
    swe_df, station_cols = load_swe_wide()
    dates = pd.to_datetime(swe_df["Date"])
    water_year = np.where(dates.dt.month >= 10, dates.dt.year + 1, dates.dt.year)

    swe_df = swe_df.copy()
    swe_df["water_year"] = water_year
    training = swe_df[swe_df["water_year"].between(TRAINING_WATER_YEAR_START, TRAINING_WATER_YEAR_END)]
    yearly_max = training.groupby("water_year")[station_cols].max()
    averaged_yearly_max_swe = yearly_max.mean(axis=0).to_numpy(dtype=float)

    groups, percentile_cuts = _assign_percentile_groups(averaged_yearly_max_swe, [30, 60, 90])

    group_df = metadata.copy()
    group_df["station_name_from_swe"] = station_cols
    group_df["averaged_yearly_max_swe"] = averaged_yearly_max_swe
    group_df["paper_group"] = groups
    group_df["paper_group_p30_p60_p90_mm"] = ",".join(f"{cut:.3f}" for cut in percentile_cuts)
    return groups, group_df


def save_paper_grouping_artifacts() -> dict[str, object]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    paper_groups, paper_df = load_paper_style_groups()
    plain_groups, plain_df = load_plain_average_swe_groups()

    original_index_col = "original_station_index" if "original_station_index" in paper_df.columns else None

    export_columns = ["subset_station_index"]
    if original_index_col:
        export_columns.append(original_index_col)
    export_columns.extend(
        [
            "Station.Id",
            "Station Name",
            "Latitude",
            "Longitude",
            "Elevation",
            "averaged_yearly_max_swe",
            "paper_group",
        ]
    )

    group_export = paper_df[export_columns].copy()
    group_export = group_export.rename(
        columns={
            "Station.Id": "station_id",
            "Station Name": "station_name",
            "averaged_yearly_max_swe": "averaged_max_swe",
            "paper_group": "paper_style_group",
        }
    )
    group_export.to_csv(FIGURE_DIR / "sierra_paper_style_station_groups.csv", index=False)

    confusion = (
        pd.DataFrame({"plain_average_swe_group": plain_groups, "paper_style_group": paper_groups})
        .groupby(["plain_average_swe_group", "paper_style_group"])
        .size()
        .reset_index(name="count")
        .sort_values(["plain_average_swe_group", "paper_style_group"])
    )
    confusion.to_csv(FIGURE_DIR / "sierra_grouping_plain_average_swe_vs_paper_style_confusion.csv", index=False)

    comparison_df = paper_df[
        ["subset_station_index", "Station Name", "averaged_yearly_max_swe", "paper_group"]
    ].merge(
        plain_df[["subset_station_index", "average_swe", "average_swe_group"]],
        on="subset_station_index",
        how="left",
        validate="one_to_one",
    )

    return {
        "paper_groups": paper_groups,
        "paper_df": paper_df,
        "plain_groups": plain_groups,
        "plain_df": plain_df,
        "comparison_df": comparison_df,
        "confusion_df": confusion,
    }


def get_test_dates_by_year() -> dict[int, pd.DatetimeIndex]:
    if str(SIERRA_HOME_ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(SIERRA_HOME_ROOT / "scripts"))

    import extended_dl_forecasting as DL

    dataloader = DL.DataLoader(test_indices=[], temp_window=7)
    dataloader.cleanDataJunk()
    dataloader.normalizeData()
    dataloader.addDayInfo()
    dataloader.prepareInputsAndOutputs()
    dataloader.completeInputsAndOutputs()

    all_dates = np.array(dataloader.all_dates)
    return {
        year: pd.DatetimeIndex(pd.to_datetime(all_dates[indices]))
        for year, indices in zip(YEARS, dataloader.getTestingDataIndices())
    }


def load_year_arrays(year: int) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(joblib.load(ACTUAL_DIR / f"{year}_daily_window_actual.pkl"))
    pred = np.asarray(joblib.load(PREDICT_DIR / f"{year}_daily_gp_window_predict.pkl"))
    if actual.shape != pred.shape:
        raise ValueError(f"Shape mismatch for {year}: actual {actual.shape} vs pred {pred.shape}")
    if actual.shape != EXPECTED_SHAPES[year]:
        raise ValueError(f"Unexpected Sierra shape for {year}: got {actual.shape}, expected {EXPECTED_SHAPES[year]}")
    return actual, pred
