
#!/usr/bin/env python3
"""Create Sierra-only Figure 4b-style scatter plots for actual vs all-station predictions."""

from __future__ import annotations

import math
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt

from foreswe_sierra_grouping_utils import (
    FIGURE_DIR,
    MONTH_NUMBERS,
    MONTHS,
    TRAINING_WATER_YEAR_END,
    TRAINING_WATER_YEAR_START,
    YEARS,
    get_test_dates_by_year,
    load_metadata,
    load_year_arrays,
    save_paper_grouping_artifacts,
)


ALL_STATION_PREDICT_DIR = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/thirdparties/SWE-Forecasting/results/predict"
)
OUTPUT_PNG = FIGURE_DIR / "sierra_fig4b_daily_swe_scatter_allstation_paper_groups.png"
OUTPUT_PDF = FIGURE_DIR / "sierra_fig4b_daily_swe_scatter_allstation_paper_groups.pdf"
OUTPUT_CSV = FIGURE_DIR / "sierra_fig4b_daily_swe_scatter_allstation_summary_paper_groups.csv"
GROUP_YLIMS = {1: (0, 700), 2: (0, 900), 3: (0, 1600), 4: (0, 3200)}


def load_all_station_prediction_subset(year: int, subset_metadata: pd.DataFrame) -> np.ndarray:
    pred = np.asarray(joblib.load(ALL_STATION_PREDICT_DIR / f"{year}_daily_gp_window_predict.pkl"))
    subset_indices = subset_metadata["original_station_index"].to_numpy(dtype=int)
    subset_pred = pred[:, subset_indices, :]
    return subset_pred


def make_plot(values: dict[tuple[int, str, str], np.ndarray]) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(18, 14), sharey=False)

    for row_idx, group in enumerate([1, 2, 3, 4]):
        for col_idx, month in enumerate(MONTHS):
            ax = axes[row_idx, col_idx]

            actual_vals = values[(group, month, "Actual")]
            pred_vals = values[(group, month, "Predicted All-Stations")]
            x = np.arange(actual_vals.size)

            ax.scatter(
                x,
                actual_vals,
                s=8,
                color="#4C78A8",
                alpha=0.5,
                edgecolors="none",
                label="Actual",
            )
            ax.scatter(
                x,
                pred_vals,
                s=8,
                color="#E45756",
                alpha=0.5,
                edgecolors="none",
                label="Predicted",
            )

            ax.set_ylim(*GROUP_YLIMS[group])
            ax.grid(axis="y", alpha=0.2)
            ax.set_xticks([])

            if row_idx == 0:
                ax.set_title(month, fontsize=12)
            if col_idx == 0:
                ax.set_ylabel(f"Group {group}\nSWE (mm)", fontsize=11)
            if row_idx == 3:
                ax.set_xlabel("Pooled daily station samples", fontsize=10)

            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="upper left")

    fig.suptitle("Sierra ForeSWE: Daily SWE Scatter by Month and Paper-Style Group\nActual vs All-Station-Trained Predictions", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    grouping = save_paper_grouping_artifacts()
    paper_df = grouping["paper_df"]
    group_masks = {group: paper_df["paper_group"].to_numpy(dtype=int) == group for group in [1, 2, 3, 4]}
    test_dates_by_year = get_test_dates_by_year()
    subset_metadata = load_metadata()

    print("Training water years used for grouping:", TRAINING_WATER_YEAR_START, "to", TRAINING_WATER_YEAR_END)
    print("Paper-style group counts:", paper_df["paper_group"].value_counts().sort_index().to_dict())
    print("Using only all-station-trained predictions from:", ALL_STATION_PREDICT_DIR)

    values_lists = {
        (group, month, variable): []
        for group in [1, 2, 3, 4]
        for month in MONTHS
        for variable in ["Actual", "Predicted All-Stations"]
    }
    summary_rows: list[dict[str, float | int | str]] = []

    for year in YEARS:
        actual_sierra, _ = load_year_arrays(year)
        pred_all_subset = load_all_station_prediction_subset(year, subset_metadata)
        dates = test_dates_by_year[year]

        if actual_sierra.shape != pred_all_subset.shape:
            raise ValueError(
                f"Shape mismatch for {year}: actual {actual_sierra.shape}, all-station subset pred {pred_all_subset.shape}"
            )

        actual_use = actual_sierra[:, :, 0]
        pred_use = pred_all_subset[:, :, 0]

        for month in MONTHS:
            month_mask = dates.month == MONTH_NUMBERS[month]
            for group in [1, 2, 3, 4]:
                station_mask = group_masks[group]
                actual_vals = actual_use[month_mask][:, station_mask].reshape(-1)
                pred_vals = pred_use[month_mask][:, station_mask].reshape(-1)

                valid_mask = np.isfinite(actual_vals) & np.isfinite(pred_vals)
                actual_vals = actual_vals[valid_mask]
                pred_vals = pred_vals[valid_mask]

                values_lists[(group, month, "Actual")].append(actual_vals)
                values_lists[(group, month, "Predicted All-Stations")].append(pred_vals)

                for variable, arr in [("Actual", actual_vals), ("Predicted All-Stations", pred_vals)]:
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
                        }
                    )

    values = {
        key: np.concatenate(value_list) if value_list else np.array([])
        for key, value_list in values_lists.items()
    }

    make_plot(values)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_CSV, index=False)

    month_group_counts = (
        summary_df[summary_df["variable"] == "Actual"]
        .groupby(["month", "group"])["n_samples"]
        .sum()
        .reset_index()
    )
    print("Month sample counts by group:")
    print(month_group_counts.to_string(index=False))
    print("Saved figure PNG:", OUTPUT_PNG)
    print("Saved figure PDF:", OUTPUT_PDF)
    print("Saved summary CSV:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
