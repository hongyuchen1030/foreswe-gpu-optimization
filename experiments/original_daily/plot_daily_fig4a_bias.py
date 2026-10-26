#!/usr/bin/env python3
"""Reproduce ForeSWE Figure 4(a)-style daily relative bias plot."""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


REPO_ROOT = Path("/global/homes/h/hyvchen/foreswe-gpu-optimization")
THIRDPARTY_ROOT = REPO_ROOT
ACTUAL_DIR = THIRDPARTY_ROOT / "results" / "actual"
PREDICT_DIR = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/thirdparties/SWE-Forecasting/results/predict"
)
FIGURE_DIR = REPO_ROOT / "experiments" / "original_daily" / "figures"

YEARS = [2014, 2015, 2016, 2017, 2018]
EXPECTED_SHAPES = {
    2014: (181, 512, 10),
    2015: (182, 512, 10),
    2016: (180, 512, 10),
    2017: (181, 512, 10),
    2018: (181, 512, 10),
}
MONTHS = ["February", "March", "April", "May"]
MONTH_NUMBERS = {"February": 2, "March": 3, "April": 4, "May": 5}
HORIZON_LABELS = [1, 3, 5, 7, 10]
HORIZON_INDICES = [0, 2, 4, 6, 9]
MIN_ACTUAL_SWE_MM = 1.0
TEST_STATION_INDICES = [
    43, 66, 338, 463, 115, 105, 83, 306, 481, 21, 185, 136, 110, 406, 408, 132,
    120, 197, 7, 301, 375, 236, 269, 287, 49, 241, 212, 145, 264, 435, 73, 277, 364,
    438, 372, 368, 243, 493, 497, 14, 32, 433, 313, 213, 42, 475, 160, 190, 281, 285,
    316, 461, 154, 424, 310, 340, 179, 167, 193, 466, 56, 158, 365, 492, 432, 415, 68,
    409, 76, 148, 267, 227, 91, 189, 472, 162, 147, 395, 113, 2, 271, 357, 347, 335, 507,
    465, 57, 19, 442, 381, 398, 79, 337, 501, 41, 339, 331, 63, 359, 232, 323, 258,
]


def get_station_groups() -> tuple[pd.DataFrame, list[str]]:
    """Derive paper-style location groups from average annual max SWE."""
    swe_df = pd.read_csv(THIRDPARTY_ROOT / "data" / "swe_values.csv")
    dates = pd.to_datetime(swe_df["Date"])
    station_cols = [col for col in swe_df.columns if col != "Date"]

    active_mask = dates.dt.month.isin([12, 1, 2, 3, 4, 5])
    swe_active = swe_df.loc[active_mask, ["Date"] + station_cols].copy()
    active_dates = pd.to_datetime(swe_active["Date"])
    water_year = np.where(active_dates.dt.month >= 10, active_dates.dt.year + 1, active_dates.dt.year)

    # Test files labeled 2014-2018 correspond to water years 2015-2019.
    training_mask = ~pd.Series(water_year).isin([2015, 2016, 2017, 2018, 2019]).to_numpy()
    training_swe = swe_active.loc[training_mask, station_cols]
    training_water_year = water_year[training_mask]

    annual_max = training_swe.groupby(training_water_year).max()
    avg_max_swe = annual_max.mean(axis=0)
    percentile_cuts = np.percentile(avg_max_swe.values, [25, 50, 75])

    # Groups are low-to-high SWE quartiles, matching the paper's Group 1..4 ordering.
    groups = np.digitize(avg_max_swe.values, percentile_cuts, right=True) + 1

    group_df = pd.DataFrame(
        {
            "Station Name": avg_max_swe.index,
            "avg_training_annual_max_swe_mm": avg_max_swe.values,
            "group": groups,
        }
    )
    return group_df, [f"{cut:.3f}" for cut in percentile_cuts]


def get_test_dates_by_year() -> dict[int, pd.DatetimeIndex]:
    os.environ["BASE_URL"] = str(THIRDPARTY_ROOT)
    sys.path.insert(0, str(THIRDPARTY_ROOT))
    from transformer_ar.transformer_dl import DataLoader

    dataloader = DataLoader(test_indices=TEST_STATION_INDICES, temp_window=7)
    _, _, all_dates = dataloader.getAllInputAndTargets()

    dates_by_year: dict[int, pd.DatetimeIndex] = {}
    for year, indices in zip(YEARS, dataloader.getTestingDataIndices()):
        dates_by_year[year] = pd.DatetimeIndex(pd.to_datetime(all_dates[indices]))
    return dates_by_year


def load_year_arrays(year: int) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(joblib.load(ACTUAL_DIR / f"{year}_daily_window_actual.pkl"))
    pred = np.asarray(joblib.load(PREDICT_DIR / f"{year}_daily_gp_window_predict.pkl"))

    if actual.shape != pred.shape:
        raise ValueError(f"Shape mismatch for {year}: actual {actual.shape} vs pred {pred.shape}")

    expected_shape = EXPECTED_SHAPES[year]
    if actual.shape != expected_shape:
        raise ValueError(f"Unexpected shape for {year}: got {actual.shape}, expected {expected_shape}")

    return actual, pred


def make_plot(plot_values: dict[tuple[int, str, int], list[np.ndarray]], output_png: Path, output_pdf: Path) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(16, 14), sharey=True)
    colors = ["#4C78A8", "#72B7B2", "#F2CF5B", "#E45756", "#B279A2"]

    for row_idx, group in enumerate([1, 2, 3, 4]):
        for col_idx, month in enumerate(MONTHS):
            ax = axes[row_idx, col_idx]
            ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)

            blocked_panel = month == "May" and group in (1, 2)
            if blocked_panel:
                rect = mpatches.Rectangle(
                    (0, 0),
                    1,
                    1,
                    transform=ax.transAxes,
                    facecolor="#E6E6E6",
                    edgecolor="#666666",
                    hatch="///",
                    linewidth=1.0,
                    zorder=0,
                )
                ax.add_patch(rect)
                ax.text(
                    0.5,
                    0.5,
                    "Snow-free",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="#444444",
                    transform=ax.transAxes,
                )
            else:
                data = [np.concatenate(plot_values[(group, month, horizon)]) for horizon in HORIZON_LABELS]
                bp = ax.boxplot(
                    data,
                    labels=[str(h) for h in HORIZON_LABELS],
                    patch_artist=True,
                    showfliers=False,
                    widths=0.65,
                )
                for patch, color in zip(bp["boxes"], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.8)
                for median in bp["medians"]:
                    median.set_color("black")
                    median.set_linewidth(1.2)

            ax.set_ylim(-100, 100)
            ax.set_xticks(range(1, len(HORIZON_LABELS) + 1))
            ax.set_xticklabels([str(h) for h in HORIZON_LABELS], fontsize=9)

            if row_idx == 0:
                ax.set_title(month, fontsize=12)
            if col_idx == 0:
                ax.set_ylabel(f"Group {group}\nRelative Bias %", fontsize=11)
            if row_idx == 3:
                ax.set_xlabel("Window (days)", fontsize=10)

    fig.suptitle(
        "ForeSWE Reproduction: Daily Relative Bias by Month, Location Group, and Forecast Horizon",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    metadata_df = pd.read_csv(THIRDPARTY_ROOT / "data" / "snotel_location_info_2.csv")
    station_groups, percentile_cuts = get_station_groups()

    metadata_with_groups = metadata_df.merge(station_groups, on="Station Name", how="left", validate="one_to_one")
    if metadata_with_groups["group"].isna().any():
        missing = metadata_with_groups.loc[metadata_with_groups["group"].isna(), "Station Name"].tolist()
        raise ValueError(f"Failed to assign groups to stations: {missing[:10]}")

    group_masks = {
        group: metadata_with_groups["group"].to_numpy(dtype=int) == group for group in [1, 2, 3, 4]
    }
    test_dates_by_year = get_test_dates_by_year()

    plot_values: dict[tuple[int, str, int], list[np.ndarray]] = {}
    summary_rows: list[dict[str, float | int | str]] = []
    shape_rows: list[dict[str, object]] = []

    for group in [1, 2, 3, 4]:
        for month in MONTHS:
            for horizon in HORIZON_LABELS:
                plot_values[(group, month, horizon)] = []

    print("Metadata CSV shape:", metadata_df.shape)
    print("Metadata CSV columns:", list(metadata_df.columns))
    print("Group percentile cuts (avg annual max SWE mm):", percentile_cuts)
    print("Group counts:", metadata_with_groups["group"].value_counts().sort_index().to_dict())
    print("Minimum actual SWE threshold (mm):", MIN_ACTUAL_SWE_MM)

    for year in YEARS:
        actual, pred = load_year_arrays(year)
        dates = test_dates_by_year[year]
        if len(dates) != actual.shape[0]:
            raise ValueError(
                f"Date length mismatch for {year}: {len(dates)} derived dates vs {actual.shape[0]} array rows"
            )

        shape_rows.append({"year": year, "actual_shape": actual.shape, "pred_shape": pred.shape})
        print(f"{year}: actual shape {actual.shape}, pred shape {pred.shape}, start {dates.min().date()}, end {dates.max().date()}")

        for month in MONTHS:
            month_mask = dates.month == MONTH_NUMBERS[month]

            for group in [1, 2, 3, 4]:
                station_mask = group_masks[group]

                for horizon_label, horizon_index in zip(HORIZON_LABELS, HORIZON_INDICES):
                    actual_slice = actual[month_mask][:, station_mask, horizon_index]
                    pred_slice = pred[month_mask][:, station_mask, horizon_index]

                    valid_mask = np.isfinite(actual_slice) & np.isfinite(pred_slice) & (actual_slice > MIN_ACTUAL_SWE_MM)
                    bias = 100.0 * (pred_slice[valid_mask] - actual_slice[valid_mask]) / actual_slice[valid_mask]

                    if bias.size == 0:
                        summary_rows.append(
                            {
                                "year": year,
                                "month": month,
                                "group": f"Group {group}",
                                "horizon": horizon_label,
                                "n_samples": 0,
                                "mean_bias": math.nan,
                                "median_bias": math.nan,
                                "q25_bias": math.nan,
                                "q75_bias": math.nan,
                                "min_bias": math.nan,
                                "max_bias": math.nan,
                            }
                        )
                    else:
                        plot_values[(group, month, horizon_label)].append(bias)
                        summary_rows.append(
                            {
                                "year": year,
                                "month": month,
                                "group": f"Group {group}",
                                "horizon": horizon_label,
                                "n_samples": int(bias.size),
                                "mean_bias": float(np.mean(bias)),
                                "median_bias": float(np.median(bias)),
                                "q25_bias": float(np.percentile(bias, 25)),
                                "q75_bias": float(np.percentile(bias, 75)),
                                "min_bias": float(np.min(bias)),
                                "max_bias": float(np.max(bias)),
                            }
                        )

    missing_plot_keys = [key for key, values in plot_values.items() if not values and not (key[1] == "May" and key[0] in (1, 2))]
    if missing_plot_keys:
        raise ValueError(f"Missing plot data for keys: {missing_plot_keys[:10]}")

    output_png = FIGURE_DIR / "reproduced_fig4a_daily_relative_bias.png"
    output_pdf = FIGURE_DIR / "reproduced_fig4a_daily_relative_bias.pdf"
    output_csv = FIGURE_DIR / "reproduced_fig4a_bias_summary.csv"

    make_plot(plot_values, output_png, output_pdf)
    pd.DataFrame(summary_rows).to_csv(output_csv, index=False)

    print("Saved plot PNG:", output_png)
    print("Saved plot PDF:", output_pdf)
    print("Saved summary CSV:", output_csv)


if __name__ == "__main__":
    main()
