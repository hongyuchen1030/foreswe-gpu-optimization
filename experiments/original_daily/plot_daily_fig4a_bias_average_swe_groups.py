#!/usr/bin/env python3
"""Reproduce Figure 4(a)-style daily relative bias using average-SWE groups."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from foreswe_grouping_utils import (
    FIGURE_DIR,
    HORIZON_INDICES,
    HORIZON_LABELS,
    MONTH_NUMBERS,
    MONTHS,
    YEARS,
    get_test_dates_by_year,
    inspect_source_data,
    load_year_arrays,
    save_grouping_artifacts,
)


MIN_ACTUAL_SWE_MM = 1.0
OUTPUT_PNG = FIGURE_DIR / "reproduced_fig4a_daily_relative_bias_average_swe_groups.png"
OUTPUT_PDF = FIGURE_DIR / "reproduced_fig4a_daily_relative_bias_average_swe_groups.pdf"
OUTPUT_CSV = FIGURE_DIR / "reproduced_fig4a_bias_summary_average_swe_groups.csv"


def make_plot(plot_values: dict[tuple[int, str, int], list[np.ndarray]]) -> None:
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
        "ForeSWE Reproduction: Daily Relative Bias by Month, Average-SWE Group, and Forecast Horizon",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    info = inspect_source_data()
    grouping = save_grouping_artifacts()
    average_df = grouping["average_df"]
    comparison_df = grouping["comparison_df"]
    confusion_df = grouping["confusion_df"]
    group_masks = {
        group: average_df["average_swe_group"].to_numpy(dtype=int) == group for group in [1, 2, 3, 4]
    }
    percentile_cuts = np.percentile(average_df["average_swe"].to_numpy(dtype=float), [25, 50, 75])
    test_dates_by_year = get_test_dates_by_year()

    print("Metadata shape:", info["metadata_shape"])
    print("Metadata columns:", info["metadata_columns"])
    print(info["metadata_head"].to_string(index=False))
    print("SWE shape:", info["swe_shape"])
    print("SWE columns sample:", info["swe_columns"][:12])
    print(info["swe_head"].iloc[:, :8].to_string(index=False))
    print("Station count:", info["station_count"])
    print("Station order consistent with arrays:", info["station_order_matches"])
    print("SWE missing values:", info["nan_count"])
    print("SWE negative values:", info["negative_count"])
    print("SWE zero values:", info["zero_count"])
    print("SWE date range:", info["date_min"], "to", info["date_max"])
    print("Average SWE quartile cuts (mm):", [round(x, 3) for x in percentile_cuts.tolist()])
    print("Average SWE group counts:", average_df["average_swe_group"].value_counts().sort_index().to_dict())
    print("Old vs new confusion table:")
    print(confusion_df.to_string(index=False))
    print("Minimum actual SWE threshold (mm):", MIN_ACTUAL_SWE_MM)

    plot_values: dict[tuple[int, str, int], list[np.ndarray]] = {
        (group, month, horizon): []
        for group in [1, 2, 3, 4]
        for month in MONTHS
        for horizon in HORIZON_LABELS
    }
    summary_rows: list[dict[str, float | int | str]] = []

    for year in YEARS:
        actual, pred = load_year_arrays(year)
        dates = test_dates_by_year[year]
        print(
            f"{year}: actual {actual.shape}, pred {pred.shape}, start {dates.min().date()}, end {dates.max().date()}"
        )
        print(
            f"{year} month window counts:",
            {month: int((dates.month == MONTH_NUMBERS[month]).sum()) for month in MONTHS},
        )

        for month in MONTHS:
            month_mask = dates.month == MONTH_NUMBERS[month]
            for group in [1, 2, 3, 4]:
                station_mask = group_masks[group]
                for horizon_label, horizon_index in zip(HORIZON_LABELS, HORIZON_INDICES):
                    actual_slice = actual[month_mask][:, station_mask, horizon_index]
                    pred_slice = pred[month_mask][:, station_mask, horizon_index]
                    valid_mask = np.isfinite(actual_slice) & np.isfinite(pred_slice) & (actual_slice > MIN_ACTUAL_SWE_MM)
                    bias = 100.0 * (pred_slice[valid_mask] - actual_slice[valid_mask]) / actual_slice[valid_mask]
                    if bias.size:
                        plot_values[(group, month, horizon_label)].append(bias)

                    summary_rows.append(
                        {
                            "year": year,
                            "month": month,
                            "group": f"Group {group}",
                            "horizon": horizon_label,
                            "n_samples": int(bias.size),
                            "mean_bias": float(np.mean(bias)) if bias.size else math.nan,
                            "median_bias": float(np.median(bias)) if bias.size else math.nan,
                            "q25_bias": float(np.percentile(bias, 25)) if bias.size else math.nan,
                            "q75_bias": float(np.percentile(bias, 75)) if bias.size else math.nan,
                            "min_bias": float(np.min(bias)) if bias.size else math.nan,
                            "max_bias": float(np.max(bias)) if bias.size else math.nan,
                        }
                    )

    missing_plot_keys = [key for key, values in plot_values.items() if not values and not (key[1] == "May" and key[0] in (1, 2))]
    if missing_plot_keys:
        raise ValueError(f"Missing plot data for {missing_plot_keys[:10]}")

    make_plot(plot_values)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_CSV, index=False)

    april_group1 = summary_df[(summary_df["month"] == "April") & (summary_df["group"] == "Group 1")]
    print("April Group 1 mean/median bias by horizon with average-SWE grouping:")
    print(april_group1[["year", "horizon", "n_samples", "mean_bias", "median_bias"]].to_string(index=False))

    old_summary_path = FIGURE_DIR / "reproduced_fig4a_bias_summary.csv"
    if old_summary_path.exists():
        old_summary = pd.read_csv(old_summary_path)
        old_april_group1 = old_summary[(old_summary["month"] == "April") & (old_summary["group"] == "Group 1")]
        merged = old_april_group1.merge(
            april_group1,
            on=["year", "month", "group", "horizon"],
            suffixes=("_old_max_swe_grouping", "_new_average_swe_grouping"),
        )
        print("April Group 1 old-vs-new median bias comparison:")
        print(
            merged[
                [
                    "year",
                    "horizon",
                    "median_bias_old_max_swe_grouping",
                    "median_bias_new_average_swe_grouping",
                    "mean_bias_old_max_swe_grouping",
                    "mean_bias_new_average_swe_grouping",
                ]
            ].to_string(index=False)
        )

    print("Saved grouping CSV:", FIGURE_DIR / "average_swe_station_groups.csv")
    print("Saved confusion CSV:", FIGURE_DIR / "grouping_old_vs_average_swe_confusion.csv")
    print("Saved figure PNG:", OUTPUT_PNG)
    print("Saved figure PDF:", OUTPUT_PDF)
    print("Saved summary CSV:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
