
#!/usr/bin/env python3
"""Create Sierra-only predicted-vs-actual scatter plots by month and paper-style group."""

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
OUTPUT_PNG = FIGURE_DIR / "sierra_fig4b_pred_vs_actual_scatter_allstation_paper_groups.png"
OUTPUT_PDF = FIGURE_DIR / "sierra_fig4b_pred_vs_actual_scatter_allstation_paper_groups.pdf"
OUTPUT_CSV = FIGURE_DIR / "sierra_fig4b_pred_vs_actual_scatter_allstation_summary_paper_groups.csv"
GROUP_LIMITS = {1: 700, 2: 900, 3: 1600, 4: 3200}


def load_all_station_prediction_subset(year: int, subset_metadata: pd.DataFrame) -> np.ndarray:
    pred = np.asarray(joblib.load(ALL_STATION_PREDICT_DIR / f"{year}_daily_gp_window_predict.pkl"))
    subset_indices = subset_metadata["original_station_index"].to_numpy(dtype=int)
    return pred[:, subset_indices, :]


def make_plot(values: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(16, 14), sharex=False, sharey=False)

    for row_idx, group in enumerate([1, 2, 3, 4]):
        lim = GROUP_LIMITS[group]
        for col_idx, month in <truncated omitted_approx_tokens="685" />            pred_vals = pred_use[month_mask][:, station_mask].reshape(-1)

                valid_mask = np.isfinite(actual_vals) & np.isfinite(pred_vals)
                actual_vals = actual_vals[valid_mask]
                pred_vals = pred_vals[valid_mask]

                values_lists[(group, month)]["actual"].append(actual_vals)
                values_lists[(group, month)]["pred"].append(pred_vals)

                abs_error = np.abs(pred_vals - actual_vals)
                summary_rows.append(
                    {
                        "year": year,
                        "month": month,
                        "group": f"Group {group}",
                        "n_samples": int(actual_vals.size),
                        "actual_mean_swe": float(np.mean(actual_vals)) if actual_vals.size else math.nan,
                        "pred_mean_swe": float(np.mean(pred_vals)) if pred_vals.size else math.nan,
                        "actual_median_swe": float(np.median(actual_vals)) if actual_vals.size else math.nan,
                        "pred_median_swe": float(np.median(pred_vals)) if pred_vals.size else math.nan,
                        "mae": float(np.mean(abs_error)) if abs_error.size else math.nan,
                        "corr": float(np.corrcoef(actual_vals, pred_vals)[0, 1]) if actual_vals.size > 1 else math.nan,
                    }
                )

    values = {
        key: (
            np.concatenate(value["actual"]) if value["actual"] else np.array([]),
            np.concatenate(value["pred"]) if value["pred"] else np.array([]),
        )
        for key, value in values_lists.items()
    }

    make_plot(values)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_CSV, index=False)

    print("Saved figure PNG:", OUTPUT_PNG)
    print("Saved figure PDF:", OUTPUT_PDF)
    print("Saved summary CSV:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
