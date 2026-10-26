#!/usr/bin/env python3
"""Diagnostics for the April Group 1 mismatch in reproduced ForeSWE Figure 4(a)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from foreswe_grouping_utils import (
    FIGURE_DIR,
    HORIZON_INDICES,
    HORIZON_LABELS,
    MONTHS,
    TRAINING_WATER_YEAR_END,
    TRAINING_WATER_YEAR_START,
    YEARS,
    get_test_dates_by_year,
    inspect_source_data,
    load_paper_style_groups,
    load_year_arrays,
)


REPO_ROOT = Path("/global/homes/h/hyvchen/foreswe-gpu-optimization")
REPORT_PATH = REPO_ROOT / "experiments" / "original_daily" / "FIG4A_APRIL_GROUP1_DIAGNOSTIC.md"
CSV_PATH = FIGURE_DIR / "fig4a_april_group1_diagnostic.csv"

MIN_ACTUAL_SWE_MM = 1.0
THRESHOLDS_MM = [0.0, 1.0, 10.0, 25.0, 50.0]


def build_old_naive_month_masks(length: int) -> dict[str, np.ndarray]:
    masks = {month: np.zeros(length, dtype=bool) for month in MONTHS}
    cuts = {
        "February": (0, min(28, length)),
        "March": (28, min(59, length)),
        "April": (59, min(89, length)),
        "May": (89, length),
    }
    for month, (start, end) in cuts.items():
        masks[month][start:end] = True
    return masks


def summarize_array(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "n_samples": 0,
            "mean_bias": math.nan,
            "median_bias": math.nan,
            "q25_bias": math.nan,
            "q75_bias": math.nan,
        }
    return {
        "n_samples": int(values.size),
        "mean_bias": float(np.mean(values)),
        "median_bias": float(np.median(values)),
        "q25_bias": float(np.percentile(values, 25)),
        "q75_bias": float(np.percentile(values, 75)),
    }


def summarize_station_values(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {
            "n_stations": 0,
            "station_mean_bias": math.nan,
            "station_median_bias": math.nan,
            "station_q25_bias": math.nan,
            "station_q75_bias": math.nan,
        }
    return {
        "n_stations": int(arr.size),
        "station_mean_bias": float(np.mean(arr)),
        "station_median_bias": float(np.median(arr)),
        "station_q25_bias": float(np.percentile(arr, 25)),
        "station_q75_bias": float(np.percentile(arr, 75)),
    }


def get_month_mask(
    convention: str,
    year_dates: pd.DatetimeIndex,
    horizon_days: int,
    selected_origin_mask: np.ndarray | None = None,
) -> np.ndarray:
    if convention == "A_old_naive_window_index":
        return build_old_naive_month_masks(len(year_dates))["April"]
    if convention == "B_target_date_april":
        target_dates = year_dates + pd.to_timedelta(horizon_days, unit="D")
        return target_dates.month == 4
    if convention == "C_origin_date_april":
        return year_dates.month == 4
    if convention == "D_selected_nonoverlap_target_april":
        target_dates = year_dates + pd.to_timedelta(horizon_days, unit="D")
        if selected_origin_mask is None:
            raise ValueError("selected_origin_mask is required for convention D")
        return (target_dates.month == 4) & selected_origin_mask
    raise ValueError(f"Unknown convention: {convention}")


def gather_pooled_samples(
    actual: np.ndarray,
    pred: np.ndarray,
    station_mask: np.ndarray,
    origin_mask: np.ndarray,
    horizon_index: int,
    threshold_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actual_slice = actual[origin_mask][:, station_mask, horizon_index]
    pred_slice = pred[origin_mask][:, station_mask, horizon_index]
    valid = np.isfinite(actual_slice) & np.isfinite(pred_slice) & (actual_slice > threshold_mm)
    if not valid.any():
        empty = np.asarray([], dtype=float)
        return empty, empty, empty
    actual_flat = actual_slice[valid]
    pred_flat = pred_slice[valid]
    bias_flat = 100.0 * (pred_flat - actual_flat) / actual_flat
    return actual_flat, pred_flat, bias_flat


def gather_stationwise_bias(
    actual: np.ndarray,
    pred: np.ndarray,
    station_mask: np.ndarray,
    origin_mask: np.ndarray,
    horizon_index: int,
    threshold_mm: float,
) -> tuple[list[float], list[float]]:
    actual_slice = actual[origin_mask][:, station_mask, horizon_index]
    pred_slice = pred[origin_mask][:, station_mask, horizon_index]
    valid = np.isfinite(actual_slice) & np.isfinite(pred_slice) & (actual_slice > threshold_mm)
    station_mean_biases: list[float] = []
    station_median_biases: list[float] = []
    if actual_slice.size == 0:
        return station_mean_biases, station_median_biases
    for station_idx in range(actual_slice.shape[1]):
        station_valid = valid[:, station_idx]
        if not station_valid.any():
            continue
        station_actual = actual_slice[:, station_idx][station_valid]
        station_pred = pred_slice[:, station_idx][station_valid]
        station_bias = 100.0 * (station_pred - station_actual) / station_actual
        station_mean_biases.append(float(np.mean(station_bias)))
        station_median_biases.append(float(np.median(station_bias)))
    return station_mean_biases, station_median_biases


def build_year_label_evidence() -> list[dict[str, str | int]]:
    return [
        {
            "topic": "test_year_definition",
            "file": "thirdparties/SWE-Forecasting/fore_swe/extended_dl_forecasting.py",
            "line_start": 154,
            "line_end": 158,
            "evidence": "Each test split spans year_i-12-01 through year_i+1-05-30.",
        },
        {
            "topic": "test_year_labels",
            "file": "thirdparties/SWE-Forecasting/fore_swe/extended_dl_forecasting.py",
            "line_start": 449,
            "line_end": 453,
            "evidence": "The five released daily test seasons are labeled 2014, 2015, 2016, 2017, 2018.",
        },
        {
            "topic": "actual_file_names",
            "file": "thirdparties/SWE-Forecasting/fore_swe/forecasting_att_daily_model.py",
            "line_start": 618,
            "line_end": 622,
            "evidence": "Daily window actual files are dumped using 2014-2018 names.",
        },
        {
            "topic": "prediction_file_names",
            "file": "thirdparties/SWE-Forecasting/fore_swe/forecasting_daily_gp_model.py",
            "line_start": 244,
            "line_end": 248,
            "evidence": "Daily GP window prediction files are dumped using 2014-2018 names.",
        },
        {
            "topic": "future_horizon_alignment",
            "file": "thirdparties/SWE-Forecasting/transformer_ar/helper_functions.py",
            "line_start": 49,
            "line_end": 54,
            "evidence": "Future outputs use indices index+1 through index+window, so horizon h targets origin_date + h days.",
        },
        {
            "topic": "attention_output_alignment",
            "file": "thirdparties/SWE-Forecasting/fore_swe/forecasting_att_daily_model.py",
            "line_start": 560,
            "line_end": 565,
            "evidence": "The selected daily path advances origins in steps of forecasting_window=10, creating a non-overlapping daily-alignment variant.",
        },
        {
            "topic": "output_target_alignment",
            "file": "thirdparties/SWE-Forecasting/fore_swe/forecasting_att_daily_model.py",
            "line_start": 316,
            "line_end": 321,
            "evidence": "getOutputs also uses index+1 through index+window, consistent with target_date = origin_date + horizon_days.",
        },
    ]


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    info = inspect_source_data()
    paper_groups, paper_df = load_paper_style_groups()
    group_mask = paper_groups == 1
    percentile_cuts = np.percentile(paper_df["averaged_yearly_max_swe"].to_numpy(dtype=float), [30, 60, 90])
    test_dates_by_year = get_test_dates_by_year()

    print("Grouping verification")
    print("====================")
    print("Metadata shape:", info["metadata_shape"])
    print("SWE shape:", info["swe_shape"])
    print("Station count:", info["station_count"])
    print("Station order consistent with arrays:", info["station_order_matches"])
    print("Training water years:", TRAINING_WATER_YEAR_START, "to", TRAINING_WATER_YEAR_END)
    print("Paper percentile cuts p30/p60/p90 (mm):", [round(x, 3) for x in percentile_cuts.tolist()])
    print("Paper group counts:", paper_df["paper_group"].value_counts().sort_index().to_dict())
    print("Paper group averaged max SWE ranges:")
    group_range_rows: list[dict[str, object]] = []
    for group in [1, 2, 3, 4]:
        vals = paper_df.loc[paper_df["paper_group"] == group, "averaged_yearly_max_swe"]
        group_range_rows.append(
            {
                "group": group,
                "count": int(vals.shape[0]),
                "min_averaged_max_swe_mm": float(vals.min()),
                "max_averaged_max_swe_mm": float(vals.max()),
            }
        )
        print(f"Group {group}: count {vals.shape[0]}, min {vals.min():.3f}, max {vals.max():.3f}")

    evidence_rows = build_year_label_evidence()
    print("\nTest-year / horizon evidence")
    print("===========================")
    for row in evidence_rows:
        print(
            f"{row['file']}:{row['line_start']}-{row['line_end']} - {row['evidence']}"
        )

    print("\nExact test-date spans recovered from DataLoader")
    print("=============================================")
    year_span_rows: list[dict[str, object]] = []
    for year, dates in test_dates_by_year.items():
        month_counts = {month: int((dates.month == month_num).sum()) for month, month_num in zip(["Dec", "Jan", "Feb", "Mar", "Apr", "May"], [12, 1, 2, 3, 4, 5])}
        year_span_rows.append(
            {
                "year_label": year,
                "n_origins": int(len(dates)),
                "origin_start": str(dates.min().date()),
                "origin_end": str(dates.max().date()),
                **month_counts,
            }
        )
        print(year, len(dates), dates.min().date(), dates.max().date(), month_counts)

    rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    aggregation_rows: list[dict[str, object]] = []

    conventions = [
        "A_old_naive_window_index",
        "B_target_date_april",
        "C_origin_date_april",
        "D_selected_nonoverlap_target_april",
    ]

    for year in YEARS:
        actual, pred = load_year_arrays(year)
        dates = test_dates_by_year[year]
        selected_origin_mask = np.zeros(len(dates), dtype=bool)
        selected_origin_mask[np.arange(0, len(dates), 10)] = True
        print(f"\nYear {year}: actual {actual.shape}, pred {pred.shape}, origin dates {dates.min().date()} to {dates.max().date()}")

        for horizon_label, horizon_index in zip(HORIZON_LABELS, HORIZON_INDICES):
            for convention in conventions:
                month_mask = get_month_mask(convention, dates, horizon_label, selected_origin_mask)
                actual_flat, pred_flat, bias_flat = gather_pooled_samples(
                    actual=actual,
                    pred=pred,
                    station_mask=group_mask,
                    origin_mask=month_mask,
                    horizon_index=horizon_index,
                    threshold_mm=MIN_ACTUAL_SWE_MM,
                )
                pooled_stats = summarize_array(bias_flat)
                row = {
                    "year": year,
                    "convention": convention,
                    "horizon_days": horizon_label,
                    "aggregation": "pooled_sample",
                    "threshold_mm": MIN_ACTUAL_SWE_MM,
                    **pooled_stats,
                    "actual_median_swe_mm": float(np.median(actual_flat)) if actual_flat.size else math.nan,
                    "predicted_median_swe_mm": float(np.median(pred_flat)) if pred_flat.size else math.nan,
                    "predicted_minus_actual_median_swe_mm": float(np.median(pred_flat) - np.median(actual_flat)) if actual_flat.size else math.nan,
                }
                rows.append(row)

                station_mean_biases, station_median_biases = gather_stationwise_bias(
                    actual=actual,
                    pred=pred,
                    station_mask=group_mask,
                    origin_mask=month_mask,
                    horizon_index=horizon_index,
                    threshold_mm=MIN_ACTUAL_SWE_MM,
                )
                station_mean_stats = summarize_station_values(station_mean_biases)
                station_median_stats = summarize_station_values(station_median_biases)
                aggregation_rows.append(
                    {
                        "year": year,
                        "convention": convention,
                        "horizon_days": horizon_label,
                        "aggregation": "station_mean_of_sample_bias",
                        **station_mean_stats,
                    }
                )
                aggregation_rows.append(
                    {
                        "year": year,
                        "convention": convention,
                        "horizon_days": horizon_label,
                        "aggregation": "station_median_of_sample_bias",
                        **station_median_stats,
                    }
                )

            primary_mask = get_month_mask("B_target_date_april", dates, horizon_label, selected_origin_mask)
            for threshold_mm in THRESHOLDS_MM:
                actual_flat, pred_flat, bias_flat = gather_pooled_samples(
                    actual=actual,
                    pred=pred,
                    station_mask=group_mask,
                    origin_mask=primary_mask,
                    horizon_index=horizon_index,
                    threshold_mm=threshold_mm,
                )
                threshold_rows.append(
                    {
                        "year": year,
                        "convention": "B_target_date_april",
                        "horizon_days": horizon_label,
                        "aggregation": "pooled_sample",
                        "threshold_mm": threshold_mm,
                        **summarize_array(bias_flat),
                        "actual_median_swe_mm": float(np.median(actual_flat)) if actual_flat.size else math.nan,
                        "predicted_median_swe_mm": float(np.median(pred_flat)) if pred_flat.size else math.nan,
                        "predicted_minus_actual_median_swe_mm": float(np.median(pred_flat) - np.median(actual_flat)) if actual_flat.size else math.nan,
                    }
                )

    diagnostics_df = pd.DataFrame(rows)
    threshold_df = pd.DataFrame(threshold_rows)
    aggregation_df = pd.DataFrame(aggregation_rows)

    csv_df = pd.concat(
        [
            diagnostics_df.assign(section="month_convention"),
            threshold_df.assign(section="threshold_sensitivity"),
            aggregation_df.assign(section="aggregation_sensitivity"),
        ],
        ignore_index=True,
        sort=False,
    )
    csv_df.to_csv(CSV_PATH, index=False)

    print("\nApril Group 1 diagnostics by convention")
    print("======================================")
    for convention in conventions:
        subset = diagnostics_df[diagnostics_df["convention"] == convention]
        print(f"\n{convention}")
        print(
            subset[
                [
                    "year",
                    "horizon_days",
                    "n_samples",
                    "mean_bias",
                    "median_bias",
                    "q25_bias",
                    "q75_bias",
                    "actual_median_swe_mm",
                    "predicted_median_swe_mm",
                    "predicted_minus_actual_median_swe_mm",
                ]
            ].to_string(index=False)
        )

    print("\nAggregation sensitivity for target-date April")
    print("============================================")
    print(
        aggregation_df[aggregation_df["convention"] == "B_target_date_april"][
            [
                "year",
                "horizon_days",
                "aggregation",
                "n_stations",
                "station_mean_bias",
                "station_median_bias",
                "station_q25_bias",
                "station_q75_bias",
            ]
        ].to_string(index=False)
    )

    print("\nThreshold sensitivity for target-date April")
    print("=========================================")
    print(
        threshold_df[
            [
                "year",
                "horizon_days",
                "threshold_mm",
                "n_samples",
                "median_bias",
                "actual_median_swe_mm",
                "predicted_median_swe_mm",
                "predicted_minus_actual_median_swe_mm",
            ]
        ].to_string(index=False)
    )

    convention_summary = (
        diagnostics_df.groupby(["convention", "horizon_days"], dropna=False)[["mean_bias", "median_bias"]]
        .mean()
        .reset_index()
    )
    threshold_summary = (
        threshold_df.groupby(["threshold_mm", "horizon_days"], dropna=False)[["median_bias"]]
        .mean()
        .reset_index()
    )
    station_mean_summary = (
        aggregation_df[
            (aggregation_df["convention"] == "B_target_date_april")
            & (aggregation_df["aggregation"] == "station_mean_of_sample_bias")
        ][["horizon_days", "station_mean_bias", "station_median_bias"]]
        .groupby("horizon_days", dropna=False)
        .mean()
        .reset_index()
    )
    station_median_summary = (
        aggregation_df[
            (aggregation_df["convention"] == "B_target_date_april")
            & (aggregation_df["aggregation"] == "station_median_of_sample_bias")
        ][["horizon_days", "station_mean_bias", "station_median_bias"]]
        .groupby("horizon_days", dropna=False)
        .mean()
        .reset_index()
    )

    def summary_value(frame: pd.DataFrame, horizon_days: int, column: str) -> float:
        row = frame.loc[frame["horizon_days"] == horizon_days, column]
        return float(row.iloc[0]) if not row.empty else math.nan

    report_lines: list[str] = []
    report_lines.append("# Figure 4(a) April Group 1 Diagnostic")
    report_lines.append("")
    report_lines.append("## Scope")
    report_lines.append("")
    report_lines.append("This diagnostic investigates why the reproduced ForeSWE Figure 4(a) shows April Group 1 relative bias as negative or near zero at short horizons, while the paper figure appears mostly positive.")
    report_lines.append("")
    report_lines.append("No new final figure was generated in this step.")
    report_lines.append("")
    report_lines.append("## Grouping Verification")
    report_lines.append("")
    report_lines.append(f"- Current diagnostic grouping uses averaged yearly max SWE across training water years {TRAINING_WATER_YEAR_START}-{TRAINING_WATER_YEAR_END}.")
    report_lines.append("- Thresholds are the 30th, 60th, and 90th percentiles of station-wise averaged yearly max SWE.")
    report_lines.append(f"- Group counts: {paper_df['paper_group'].value_counts().sort_index().to_dict()}.")
    report_lines.append(f"- Percentile cuts (mm): {[round(x, 3) for x in percentile_cuts.tolist()]}.")
    for row in group_range_rows:
        report_lines.append(
            f"- Group {row['group']}: count {row['count']}, averaged max SWE range {row['min_averaged_max_swe_mm']:.3f}-{row['max_averaged_max_swe_mm']:.3f} mm."
        )
    report_lines.append("")
    report_lines.append("This matches the paper-style setup much better than equal-size quartile grouping, so the current mismatch is unlikely to be caused by the station grouping itself.")
    report_lines.append("")
    report_lines.append("## Test-Year Labeling Evidence")
    report_lines.append("")
    report_lines.append("- The released code names daily outputs by the December start year, not the ending water year.")
    report_lines.append("- Therefore `2014_daily_gp_window_predict.pkl` corresponds to the test season from `2014-12-01` through `2015-05-30`, which is effectively the paper's WY2015 test season.")
    report_lines.append("- Evidence:")
    for row in evidence_rows:
        report_lines.append(
            f"  - `{row['file']}:{row['line_start']}-{row['line_end']}`: {row['evidence']}"
        )
    report_lines.append("")
    report_lines.append("## Date Alignment")
    report_lines.append("")
    report_lines.append("- Exact origin dates were recovered from the released `DataLoader` rather than guessed from array length.")
    report_lines.append("- The recovered daily test spans are:")
    for row in year_span_rows:
        report_lines.append(
            f"  - `{row['year_label']}`: {row['origin_start']} to {row['origin_end']} with {row['n_origins']} origins (Dec {row['Dec']}, Jan {row['Jan']}, Feb {row['Feb']}, Mar {row['Mar']}, Apr {row['Apr']}, May {row['May']})."
        )
    report_lines.append("- `getFutureOutputs` and `getOutputs` both use `index+1` through `index+window`, so horizon `h` maps to `origin_date + h days`, not `origin_date + h - 1`.")
    report_lines.append("")
    report_lines.append("The four April conventions tested here are:")
    report_lines.append("- A: old naive month slicing by raw window index (`0:28`, `28:59`, `59:89`, `89:`).")
    report_lines.append("- B: April defined by target date, `origin_date + horizon_days` in April.")
    report_lines.append("- C: April defined by prediction origin date in April.")
    report_lines.append("- D: target-date April using only non-overlapping origins `0, 10, 20, ...`, matching the released selected-daily stepping logic.")
    report_lines.append("")
    report_lines.append("## April Group 1: Month-Slicing Comparison")
    report_lines.append("")
    report_lines.append("Mean and median values below are averages across the five released test seasons for each horizon.")
    report_lines.append("")
    report_lines.append("| Convention | Horizon (days) | Avg pooled mean bias (%) | Avg pooled median bias (%) |")
    report_lines.append("| --- | ---: | ---: | ---: |")
    for _, row in convention_summary.iterrows():
        report_lines.append(
            f"| {row['convention']} | {int(row['horizon_days'])} | {row['mean_bias']:.3f} | {row['median_bias']:.3f} |"
        )
    report_lines.append("")
    report_lines.append("Interpretation:")
    report_lines.append("- Convention A is strongly misaligned with the actual calendar because the arrays start in December; it should not be used for paper-style month panels.")
    report_lines.append("- Convention B and C are the most relevant choices for Figure 4(a).")
    report_lines.append(
        f"- For target-date April, the average pooled median bias at 1/3/5 days is {summary_value(convention_summary[convention_summary['convention'] == 'B_target_date_april'], 1, 'median_bias'):.3f}, {summary_value(convention_summary[convention_summary['convention'] == 'B_target_date_april'], 3, 'median_bias'):.3f}, and {summary_value(convention_summary[convention_summary['convention'] == 'B_target_date_april'], 5, 'median_bias'):.3f}."
    )
    report_lines.append(
        f"- For origin-date April, the corresponding values are {summary_value(convention_summary[convention_summary['convention'] == 'C_origin_date_april'], 1, 'median_bias'):.3f}, {summary_value(convention_summary[convention_summary['convention'] == 'C_origin_date_april'], 3, 'median_bias'):.3f}, and {summary_value(convention_summary[convention_summary['convention'] == 'C_origin_date_april'], 5, 'median_bias'):.3f}."
    )
    report_lines.append("- This changes the magnitude somewhat, especially at 5+ days, but does not flip the short-horizon sign from negative to clearly positive.")
    report_lines.append("- Convention D tests the non-overlapping selected-daily stepping logic. It stays close to the target-date result and does not resolve the early-horizon mismatch.")
    report_lines.append("")
    report_lines.append("## Aggregation Diagnostic")
    report_lines.append("")
    report_lines.append("For target-date April only, the script compares pooled sample-level bias against station-level aggregation.")
    report_lines.append("- Pooled sample-level bias uses every valid station-day-horizon sample.")
    report_lines.append("- Station-level aggregation first summarizes each station's April bias across days, then aggregates the station summaries.")
    report_lines.append("")
    report_lines.append("This matters because pooled aggregation can overweight stations with more valid April days, while station-level aggregation can shift the median if low-SWE stations have sparse but extreme values.")
    report_lines.append(
        f"- At 3 days, the average pooled median bias is {summary_value(convention_summary[convention_summary['convention'] == 'B_target_date_april'], 3, 'median_bias'):.3f}, while the average median across station-level mean biases is {summary_value(station_mean_summary, 3, 'station_median_bias'):.3f}."
    )
    report_lines.append(
        f"- At 5 days, the pooled median is {summary_value(convention_summary[convention_summary['convention'] == 'B_target_date_april'], 5, 'median_bias'):.3f}, while the station-level mean-bias median is {summary_value(station_mean_summary, 5, 'station_median_bias'):.3f} and the station-level median-bias median is {summary_value(station_median_summary, 5, 'station_median_bias'):.3f}."
    )
    report_lines.append("- This is the strongest signal in the diagnostics that a paper-side aggregation or boxplot-construction difference could explain the more positive-looking April Group 1 panel.")
    report_lines.append("")
    report_lines.append("## Low-SWE Threshold Diagnostic")
    report_lines.append("")
    report_lines.append("Threshold sensitivity below is computed for target-date April.")
    report_lines.append("")
    report_lines.append("| Threshold (mm) | Horizon (days) | Avg pooled median bias (%) |")
    report_lines.append("| ---: | ---: | ---: |")
    for _, row in threshold_summary.iterrows():
        report_lines.append(
            f"| {row['threshold_mm']:.0f} | {int(row['horizon_days'])} | {row['median_bias']:.3f} |"
        )
    report_lines.append("")
    report_lines.append(
        f"- Raising the threshold from 1 mm to 50 mm moves the average 1-day pooled median bias from {summary_value(threshold_summary[threshold_summary['threshold_mm'] == 1.0], 1, 'median_bias'):.3f} to {summary_value(threshold_summary[threshold_summary['threshold_mm'] == 50.0], 1, 'median_bias'):.3f}."
    )
    report_lines.append(
        f"- At 5 days, the same threshold change moves the average pooled median from {summary_value(threshold_summary[threshold_summary['threshold_mm'] == 1.0], 5, 'median_bias'):.3f} to {summary_value(threshold_summary[threshold_summary['threshold_mm'] == 50.0], 5, 'median_bias'):.3f}."
    )
    report_lines.append("- Higher thresholds make April Group 1 slightly more negative at short horizons, so low-SWE filtering does not explain the paper's positive-looking panel.")
    report_lines.append("")
    report_lines.append("## Main Takeaway")
    report_lines.append("")
    report_lines.append("The station grouping is already consistent with the paper, and the 2014-2018 file names are just December-start labels for the paper's 2015-2019 test seasons.")
    report_lines.append("Target-date April versus origin-date April changes the short-horizon April Group 1 medians only modestly, and the non-overlapping selected-daily alignment also stays in the same regime.")
    report_lines.append("Changing the low-SWE threshold does not help; it pushes the short-horizon medians slightly more negative.")
    report_lines.append("The strongest remaining explanation is an aggregation mismatch: if the paper's Figure 4(a) boxes are based on station-level April bias summaries rather than pooled station-day samples, April Group 1 can shift from slightly negative to around zero or positive at 3-5 day horizons.")
    report_lines.append("That still does not fully explain the 1-day sign, so there may be an additional convention difference in the paper, but aggregation choice is the leading explanation from the evidence collected here.")
    report_lines.append("")
    report_lines.append("## Output Files")
    report_lines.append("")
    report_lines.append(f"- CSV: `{CSV_PATH}`")
    report_lines.append(f"- This report: `{REPORT_PATH}`")
    report_lines.append("")
    report_lines.append("For full per-year values, see the CSV.")

    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\nSaved report:", REPORT_PATH)
    print("Saved CSV:", CSV_PATH)


if __name__ == "__main__":
    main()
