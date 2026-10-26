
#!/usr/bin/env python3
"""Create Sierra-only Figure 4(b)-style daily SWE availability plot using paper-style groups."""

from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt

from foreswe_sierra_grouping_utils import (
    DIAGNOSTICS_DIR,
    FIGURE_DIR,
    MONTH_NUMBERS,
    MONTHS,
    TRAINING_WATER_YEAR_END,
    TRAINING_WATER_YEAR_START,
    YEARS,
    get_test_dates_by_year,
    load_year_arrays,
    save_paper_grouping_artifacts,
)


OUTPUT_PNG = FIGURE_DIR / "sierra_fig4b_daily_swe_availability_paper_groups.png"
OUTPUT_PDF = FIGURE_DIR / "sierra_fig4b_daily_swe_availability_paper_groups.pdf"
OUTPUT_CSV = FIGURE_DIR / "sierra_fig4b_swe_summary_paper_groups.csv"
GROUP_YLIMS = {1: (0, 700), 2: (0, 900), 3: (0, 1600), 4: (0, 3200)}


def make_plot(values: dict[tuple[int, str, str], list[np.ndarray]]) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(13, 16), sharex=True)
    month_positions = np.arange(len(MONTHS)) * 3.0
    actual_positions = month_positions - 0.35
    pred_positions = month_positions + 0.35

    for idx, group in enumerate([1, 2, 3, 4]):
        ax = axes[idx]
        actual_data = [np.concatenate(values[(group, month, "Actual")]) for month in MONTHS]
        pred_data = [np.concatenate(values[(group, month, "Predicted")]) for month in MONTHS]

        bp1 = ax.boxplot(actual_data, positions=actual_positions, widths=0.55, patch_artist=True, showfliers=False)
        bp2 = ax.boxplot(pred_data, positions=pred_positions, widths=0.55, patch_artist=True, showfliers=False)

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
    fig.suptitle("Sierra ForeSWE: Daily SWE by Month and Paper-Style Group", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    grouping = save_paper_grouping_artifacts()
    paper_df = grouping["paper_df"]
    group_masks = {group: paper_df["paper_group"].to_numpy(dtype=int) == group for group in [1, 2, 3, 4]}
    test_dates_by_year = get_test_dates_by_year()

    print("Training water years used for grouping:", TRAINING_WATER_YEAR_START, "to", TRAINING_WATER_YEAR_END)
    print("Paper-style group counts:", paper_df["paper_group"].value_counts().sort_index().to_dict())

    values = {(group, month, variable): [] for group in [1, 2, 3, 4] for month in MONTHS for variable in ["Actual", "Predicted"]}
    summary_rows: list[dict[str, float | int | str]] = []

    for year in YEARS:
        actual, pred = load_year_arrays(year)
        dates = test_dates_by_year[year]
        actual_use = actual[:, :, 0]
        pred_use = pred[:, :, 0]
        for month in MONTHS:
            month_mask = dates.month == MONTH_NUMBERS[month]
            for group in [1, 2, 3, 4]:
                station_mask = group_masks[group]
                actual_vals = actual_use[month_mask][:, station_mask].reshape(-1)
                pred_vals = pred_use[month_mask][:, station_mask].reshape(-1)
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
                            "n_samples": int(arr.size),
                            "mean_swe": float(np.mean(arr)) if arr.size else math.nan,
                            "median_swe": float(np.median(arr)) if arr.size else math.nan,
                            "q25_swe": float(np.percentile(arr, 25)) if arr.size else math.nan,
                            "q75_swe": float(np.percentile(arr, 75)) if arr.size else math.nan,
                            "min_swe": float(np.min(arr)) if arr.size else math.nan,
                            "max_swe": float(np.max(arr)) if arr.size else math.nan,
                        }
                    )

    make_plot(values)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_CSV, index=False)

    month_group_counts = summary_df[summary_df["variable"] == "Actual"].groupby(["month", "group"])["n_samples"].sum().reset_index()
    print("Month sample counts by group:")
    print(month_group_counts.to_string(index=False))

    print("Saved grouping CSV:", FIGURE_DIR / "sierra_paper_style_station_groups.csv")
    print("Saved figure PNG:", OUTPUT_PNG)
    print("Saved figure PDF:", OUTPUT_PDF)
    print("Saved summary CSV:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
