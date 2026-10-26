#!/usr/bin/env python3
"""Reproduce Figure 4(b)-style SWE availability plots using average-SWE groups."""

from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt

from foreswe_grouping_utils import (
    FIGURE_DIR,
    MONTH_NUMBERS,
    MONTHS,
    YEARS,
    get_test_dates_by_year,
    load_year_arrays,
    save_grouping_artifacts,
)


MAIN_PNG = FIGURE_DIR / "reproduced_fig4b_daily_swe_availability_average_swe_groups.png"
MAIN_PDF = FIGURE_DIR / "reproduced_fig4b_daily_swe_availability_average_swe_groups.pdf"
ALL_PNG = FIGURE_DIR / "reproduced_fig4b_daily_swe_availability_average_swe_groups_all_horizons.png"
ALL_PDF = FIGURE_DIR / "reproduced_fig4b_daily_swe_availability_average_swe_groups_all_horizons.pdf"
OUTPUT_CSV = FIGURE_DIR / "reproduced_fig4b_swe_summary_average_swe_groups.csv"
GROUP_YLIMS = {1: (0, 700), 2: (0, 900), 3: (0, 1600), 4: (0, 3200)}


def build_group_month_values(
    horizon_mode: str,
    group_masks: dict[int, np.ndarray],
    test_dates_by_year: dict[int, pd.DatetimeIndex],
) -> tuple[dict[tuple[int, str, str], list[np.ndarray]], list[dict[str, float | int | str]]]:
    values = {
        (group, month, variable): []
        for group in [1, 2, 3, 4]
        for month in MONTHS
        for variable in ["Actual", "Predicted"]
    }
    summary_rows: list[dict[str, float | int | str]] = []

    for year in YEARS:
        actual, pred = load_year_arrays(year)
        dates = test_dates_by_year[year]

        if horizon_mode == "horizon_1":
            actual_use = actual[:, :, 0]
            pred_use = pred[:, :, 0]
        elif horizon_mode == "all_horizons":
            actual_use = actual
            pred_use = pred
        else:
            raise ValueError(f"Unsupported horizon_mode: {horizon_mode}")

        for month in MONTHS:
            month_mask = dates.month == MONTH_NUMBERS[month]
            for group in [1, 2, 3, 4]:
                station_mask = group_masks[group]

                if horizon_mode == "horizon_1":
                    actual_vals = actual_use[month_mask][:, station_mask].reshape(-1)
                    pred_vals = pred_use[month_mask][:, station_mask].reshape(-1)
                else:
                    actual_vals = actual_use[month_mask][:, station_mask, :].reshape(-1)
                    pred_vals = pred_use[month_mask][:, station_mask, :].reshape(-1)

                actual_vals = actual_vals[np.isfinite(actual_vals)]
                pred_vals = pred_vals[np.isfinite(pred_vals)]
                values[(group, month, "Actual")].append(actual_vals)
                values[(group, month, "Predicted")].append(pred_vals)

                for variable, arr in [("Actual", actual_vals), ("Predicted", pred_vals)]:
                    summary_rows.append(
                        {
                            "year": year,
                            "month": month,
                            "group": f"Group {group}",
                            "variable": variable,
                            "horizon_mode": horizon_mode,
                            "n_samples": int(arr.size),
                            "mean_swe": float(np.mean(arr)) if arr.size else math.nan,
                            "median_swe": float(np.median(arr)) if arr.size else math.nan,
                            "q25_swe": float(np.percentile(arr, 25)) if arr.size else math.nan,
                            "q75_swe": float(np.percentile(arr, 75)) if arr.size else math.nan,
                            "min_swe": float(np.min(arr)) if arr.size else math.nan,
                            "max_swe": float(np.max(arr)) if arr.size else math.nan,
                        }
                    )

    return values, summary_rows


def make_plot(values: dict[tuple[int, str, str], list[np.ndarray]], output_png, output_pdf, title: str) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(13, 16), sharex=True)
    month_positions = np.arange(len(MONTHS)) * 3.0
    actual_positions = month_positions - 0.35
    pred_positions = month_positions + 0.35

    for idx, group in enumerate([1, 2, 3, 4]):
        ax = axes[idx]
        actual_data = [np.concatenate(values[(group, month, "Actual")]) for month in MONTHS]
        pred_data = [np.concatenate(values[(group, month, "Predicted")]) for month in MONTHS]

        bp1 = ax.boxplot(
            actual_data,
            positions=actual_positions,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
        )
        bp2 = ax.boxplot(
            pred_data,
            positions=pred_positions,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
        )

        for patch in bp1["boxes"]:
            patch.set_facecolor("#4C78A8")
            patch.set_alpha(0.75)
        for patch in bp2["boxes"]:
            patch.set_facecolor("#E45756")
            patch.set_alpha(0.75)
        for median in bp1["medians"] + bp2["medians"]:
            median.set_color("black")
            median.set_linewidth(1.1)

        ax.set_ylim(*GROUP_YLIMS[group])
        ax.set_ylabel(f"Group {group}\nSWE (mm)", fontsize=11)
        ax.grid(axis="y", alpha=0.2)

        if idx == 0:
            ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Actual", "Predicted"], loc="upper left")

    axes[-1].set_xticks(month_positions)
    axes[-1].set_xticklabels(["Feb", "Mar", "Apr", "May"], fontsize=11)
    axes[-1].set_xlabel("Month", fontsize=12)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    grouping = save_grouping_artifacts()
    average_df = grouping["average_df"]
    group_masks = {
        group: average_df["average_swe_group"].to_numpy(dtype=int) == group for group in [1, 2, 3, 4]
    }
    test_dates_by_year = get_test_dates_by_year()

    print("Average SWE group counts:", average_df["average_swe_group"].value_counts().sort_index().to_dict())

    all_summary_rows: list[dict[str, float | int | str]] = []
    for horizon_mode, png, pdf, title in [
        ("horizon_1", MAIN_PNG, MAIN_PDF, "ForeSWE Reproduction: Daily SWE by Month and Average-SWE Group (1-day horizon)"),
        ("all_horizons", ALL_PNG, ALL_PDF, "ForeSWE Reproduction: Daily SWE by Month and Average-SWE Group (all horizons)"),
    ]:
        values, summary_rows = build_group_month_values(horizon_mode, group_masks, test_dates_by_year)
        all_summary_rows.extend(summary_rows)
        make_plot(values, png, pdf, title)

    summary_df = pd.DataFrame(all_summary_rows)
    summary_df.to_csv(OUTPUT_CSV, index=False)

    month_group_counts = (
        summary_df[(summary_df["variable"] == "Actual") & (summary_df["horizon_mode"] == "horizon_1")]
        .groupby(["month", "group"])["n_samples"]
        .sum()
        .reset_index()
    )
    print("Month sample counts by group (1-day horizon):")
    print(month_group_counts.to_string(index=False))

    median_table = (
        summary_df[summary_df["horizon_mode"] == "horizon_1"]
        .groupby(["month", "group", "variable"])["median_swe"]
        .mean()
        .reset_index()
    )
    print("Median actual/predicted SWE by group and month (1-day horizon):")
    print(median_table.to_string(index=False))

    def get_median(month: str, group: str, variable: str, horizon_mode: str = "horizon_1") -> float:
        sub = summary_df[
            (summary_df["month"] == month)
            & (summary_df["group"] == group)
            & (summary_df["variable"] == variable)
            & (summary_df["horizon_mode"] == horizon_mode)
        ]
        return float(sub["median_swe"].mean())

    april_g1_actual = get_median("April", "Group 1", "Actual")
    april_g1_pred = get_median("April", "Group 1", "Predicted")
    print("April Group 1 median actual SWE:", april_g1_actual)
    print("April Group 1 median predicted SWE:", april_g1_pred)
    print("April Group 1 median predicted - actual SWE:", april_g1_pred - april_g1_actual)

    for group in ["Group 3", "Group 4"]:
        april_actual = get_median("April", group, "Actual")
        april_pred = get_median("April", group, "Predicted")
        may_actual = get_median("May", group, "Actual")
        may_pred = get_median("May", group, "Predicted")
        print(f"April {group} median actual/predicted SWE:", april_actual, april_pred, "difference", april_pred - april_actual)
        print(f"May {group} median actual/predicted SWE:", may_actual, may_pred, "difference", may_pred - may_actual)

    may_horizon_stats = []
    for horizon_idx, horizon_name in enumerate(range(1, 11), start=1):
        for group in [3, 4]:
            month = "May"
            med_actual = []
            med_pred = []
            for year in YEARS:
                actual, pred = load_year_arrays(year)
                dates = test_dates_by_year[year]
                month_mask = dates.month == MONTH_NUMBERS[month]
                station_mask = group_masks[group]
                actual_vals = actual[month_mask][:, station_mask, horizon_idx - 1].reshape(-1)
                pred_vals = pred[month_mask][:, station_mask, horizon_idx - 1].reshape(-1)
                med_actual.append(float(np.median(actual_vals)))
                med_pred.append(float(np.median(pred_vals)))
            may_horizon_stats.append(
                {
                    "group": f"Group {group}",
                    "horizon": horizon_name,
                    "median_actual_swe": float(np.mean(med_actual)),
                    "median_predicted_swe": float(np.mean(med_pred)),
                    "pred_minus_actual": float(np.mean(med_pred) - np.mean(med_actual)),
                }
            )
    may_horizon_df = pd.DataFrame(may_horizon_stats)
    print("May Groups 3/4 median SWE by horizon:")
    print(may_horizon_df.to_string(index=False))

    print("Saved figure PNG:", MAIN_PNG)
    print("Saved figure PDF:", MAIN_PDF)
    print("Saved all-horizon PNG:", ALL_PNG)
    print("Saved all-horizon PDF:", ALL_PDF)
    print("Saved summary CSV:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
