#!/usr/bin/env python3
"""Diagnostic Figure 4(a) variant using station-level monthly mean aggregation."""

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
    load_paper_style_groups,
    load_year_arrays,
)


REPO_ROOT = Path("/global/homes/h/hyvchen/foreswe-gpu-optimization")
MIN_ACTUAL_SWE_MM = 1.0
OUTPUT_PNG = FIGURE_DIR / "reproduced_fig4a_daily_relative_bias_station_mean_aggregation.png"
OUTPUT_PDF = FIGURE_DIR / "reproduced_fig4a_daily_relative_bias_station_mean_aggregation.pdf"
OUTPUT_CSV = FIGURE_DIR / "reproduced_fig4a_bias_summary_station_mean_aggregation.csv"
OUTPUT_REPORT = REPO_ROOT / "experiments" / "original_daily" / "FIG4A_STATION_MEAN_AGGREGATION_COMPARISON.md"


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
                data = []
                for horizon in HORIZON_LABELS:
                    arrays = plot_values[(group, month, horizon)]
                    data.append(np.concatenate(arrays) if arrays else np.asarray([], dtype=float))
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
        "ForeSWE Reproduction: Daily Relative Bias by Month, Paper-Style Group,\n"
        "and Forecast Horizon (Station-Level Monthly Mean Aggregation)",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


def summarize(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "n_stations": 0,
            "mean_bias": math.nan,
            "median_bias": math.nan,
            "q25_bias": math.nan,
            "q75_bias": math.nan,
            "min_bias": math.nan,
            "max_bias": math.nan,
        }
    return {
        "n_stations": int(values.size),
        "mean_bias": float(np.mean(values)),
        "median_bias": float(np.median(values)),
        "q25_bias": float(np.percentile(values, 25)),
        "q75_bias": float(np.percentile(values, 75)),
        "min_bias": float(np.min(values)),
        "max_bias": float(np.max(values)),
    }


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    paper_groups, paper_df = load_paper_style_groups()
    group_masks = {group: paper_groups == group for group in [1, 2, 3, 4]}
    test_dates_by_year = get_test_dates_by_year()

    print("Station counts by paper-style group:", paper_df["paper_group"].value_counts().sort_index().to_dict())
    print("Minimum actual SWE threshold (mm):", MIN_ACTUAL_SWE_MM)
    print("Month assignment: target date = origin_date + horizon_days")
    print("Aggregation: station-level monthly mean bias")

    plot_values = {(group, month, horizon): [] for group in [1, 2, 3, 4] for month in MONTHS for horizon in HORIZON_LABELS}
    summary_rows: list[dict[str, float | int | str]] = []
    comparison_rows: list[dict[str, float | int | str]] = []

    for year in YEARS:
        actual, pred = load_year_arrays(year)
        origin_dates = test_dates_by_year[year]
        print(f"{year}: actual {actual.shape}, pred {pred.shape}, origin start {origin_dates.min().date()}, origin end {origin_dates.max().date()}")

        for month in MONTHS:
            for group in [1, 2, 3, 4]:
                station_mask = group_masks[group]
                station_indices = np.flatnonzero(station_mask)

                for horizon_label, horizon_index in zip(HORIZON_LABELS, HORIZON_INDICES):
                    target_dates = origin_dates + pd.to_timedelta(horizon_label, unit="D")
                    month_mask = target_dates.month == MONTH_NUMBERS[month]

                    actual_slice = actual[month_mask][:, station_mask, horizon_index]
                    pred_slice = pred[month_mask][:, station_mask, horizon_index]
                    valid = np.isfinite(actual_slice) & np.isfinite(pred_slice) & (actual_slice > MIN_ACTUAL_SWE_MM)

                    station_mean_biases: list[float] = []
                    pooled_biases: list[np.ndarray] = []
                    for station_col, station_index in enumerate(station_indices):
                        station_valid = valid[:, station_col]
                        if not station_valid.any():
                            continue
                        station_actual = actual_slice[:, station_col][station_valid]
                        station_pred = pred_slice[:, station_col][station_valid]
                        station_bias = 100.0 * (station_pred - station_actual) / station_actual
                        station_mean_biases.append(float(np.mean(station_bias)))
                        pooled_biases.append(station_bias)

                    station_mean_bias_arr = np.asarray(station_mean_biases, dtype=float)
                    if station_mean_bias_arr.size:
                        plot_values[(group, month, horizon_label)].append(station_mean_bias_arr)

                    summary_rows.append(
                        {
                            "year": year,
                            "month": month,
                            "group": f"Group {group}",
                            "horizon": horizon_label,
                            **summarize(station_mean_bias_arr),
                        }
                    )

                    if month == "April" and group == 1:
                        pooled = np.concatenate(pooled_biases) if pooled_biases else np.asarray([], dtype=float)
                        pooled_summary = summarize(pooled)
                        station_summary = summarize(station_mean_bias_arr)
                        comparison_rows.append(
                            {
                                "year": year,
                                "horizon": horizon_label,
                                "pooled_n_samples": int(pooled.size),
                                "pooled_mean_bias": pooled_summary["mean_bias"],
                                "pooled_median_bias": pooled_summary["median_bias"],
                                "station_n": station_summary["n_stations"],
                                "station_mean_bias": station_summary["mean_bias"],
                                "station_median_bias": station_summary["median_bias"],
                            }
                        )

    make_plot(plot_values)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_CSV, index=False)
    comparison_df = pd.DataFrame(comparison_rows)

    print("\nApril Group 1 pooled vs station-level mean aggregation:")
    print(comparison_df.to_string(index=False))

    avg_comparison = (
        comparison_df.groupby("horizon", dropna=False)[
            ["pooled_mean_bias", "pooled_median_bias", "station_mean_bias", "station_median_bias"]
        ]
        .mean()
        .reset_index()
    )

    report_lines = [
        "# Figure 4(a) Station-Mean Aggregation Comparison",
        "",
        "This diagnostic plot keeps the paper-style grouping and target-date month assignment fixed,",
        "but changes the boxplot samples from pooled station-day bias values to station-level monthly mean bias values.",
        "",
        "Settings:",
        f"- Groups: paper-style averaged max SWE, 30/60/90 percentile thresholds.",
        f"- Month assignment: target date = origin date + horizon days.",
        f"- Horizons: {HORIZON_LABELS}.",
        f"- Relative bias: `100 * (predicted - actual) / actual`.",
        f"- Minimum actual SWE threshold: > {MIN_ACTUAL_SWE_MM} mm.",
        "",
        "## April Group 1 Comparison",
        "",
        "Per-year comparison between pooled station-day aggregation and station-level mean aggregation:",
        "",
        "```text",
        comparison_df.to_string(index=False),
        "```",
        "",
        "Average across the five released test seasons:",
        "",
        "```text",
        avg_comparison.to_string(index=False),
        "```",
        "",
        "## Interpretation",
        "",
        (
            f"- At 3 days, the average pooled median bias is {avg_comparison.loc[avg_comparison['horizon'] == 3, 'pooled_median_bias'].iloc[0]:.3f}, "
            f"while the average station-level median bias is {avg_comparison.loc[avg_comparison['horizon'] == 3, 'station_median_bias'].iloc[0]:.3f}."
        ),
        (
            f"- At 5 days, the average pooled median bias is {avg_comparison.loc[avg_comparison['horizon'] == 5, 'pooled_median_bias'].iloc[0]:.3f}, "
            f"while the average station-level median bias is {avg_comparison.loc[avg_comparison['horizon'] == 5, 'station_median_bias'].iloc[0]:.3f}."
        ),
        (
            f"- At 1 day, the average pooled median bias is {avg_comparison.loc[avg_comparison['horizon'] == 1, 'pooled_median_bias'].iloc[0]:.3f}, "
            f"and the average station-level median bias is {avg_comparison.loc[avg_comparison['horizon'] == 1, 'station_median_bias'].iloc[0]:.3f}."
        ),
        "- This means station-level mean aggregation does move April Group 1 upward, especially at 3-5 days.",
        "- It does not fully flip the 1-day April Group 1 bias positive, so it is a plausible partial explanation rather than a complete resolution on its own.",
        "",
        "## Outputs",
        "",
        f"- PNG: `{OUTPUT_PNG}`",
        f"- PDF: `{OUTPUT_PDF}`",
        f"- CSV: `{OUTPUT_CSV}`",
    ]
    OUTPUT_REPORT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\nSaved figure PNG:", OUTPUT_PNG)
    print("Saved figure PDF:", OUTPUT_PDF)
    print("Saved summary CSV:", OUTPUT_CSV)
    print("Saved report:", OUTPUT_REPORT)


if __name__ == "__main__":
    main()
