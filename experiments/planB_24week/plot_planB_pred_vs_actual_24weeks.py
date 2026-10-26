import os
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PLANB_ROOT = Path(
    os.getenv(
        "FSW_PLANB_ROOT",
        "/global/homes/h/hyvchen/foreswe-gpu-optimization/experiments/planB_24week",
    )
)
PROJECT_ROOT = Path("/global/homes/h/hyvchen/foreswe-gpu-optimization")
SIERRA_STATION_SOURCE = (
    PROJECT_ROOT
    / "experiments"
    / "domain_sensitivity"
    / "D2_sierra_plus_500km"
    / "station_subset.csv"
)
TEST_YEARS = [2015, 2016, 2017, 2018, 2019]
LEAD_LABELS = [f"Week {i} ({7 * i}d)" for i in range(1, 25)]


def load_planb_arrays():
    actual_by_year = joblib.load(PLANB_ROOT / "planB_feb1_24week_actual.pkl")
    mean_by_year = joblib.load(PLANB_ROOT / "planB_feb1_24week_mean.pkl")
    actual = np.concatenate([np.asarray(actual_by_year[year]) for year in TEST_YEARS], axis=0)
    pred = np.concatenate([np.asarray(mean_by_year[year]) for year in TEST_YEARS], axis=0)
    return actual, pred


def load_sierra_indices():
    df = pd.read_csv(SIERRA_STATION_SOURCE)
    sierra = df[df["is_sierra_eval_station"] == True].copy()
    sierra = sierra.sort_values("original_station_index")
    return sierra["original_station_index"].astype(int).to_numpy(), sierra


def horizon_summary(actual_h, pred_h):
    valid = np.isfinite(actual_h) & np.isfinite(pred_h)
    x = actual_h[valid]
    y = pred_h[valid]
    rmse = float(np.sqrt(np.mean((y - x) ** 2)))
    bias = float(np.mean(y - x))
    corr = float(np.corrcoef(x, y)[0, 1]) if x.size > 1 else float("nan")
    return {
        "n_points": int(valid.sum()),
        "rmse": rmse,
        "bias": bias,
        "corr": corr,
        "actual_min": float(np.min(x)),
        "actual_max": float(np.max(x)),
        "pred_min": float(np.min(y)),
        "pred_max": float(np.max(y)),
    }


def make_plot(actual, pred, title, png_path, summary_csv):
    max_val = float(np.nanmax([actual, pred]))
    fig, axes = plt.subplots(4, 6, figsize=(22, 14), constrained_layout=True, sharex=True, sharey=True)
    summary_rows = []

    for horizon_idx, ax in enumerate(axes.flat):
        actual_h = actual[:, :, horizon_idx].reshape(-1)
        pred_h = pred[:, :, horizon_idx].reshape(-1)
        valid = np.isfinite(actual_h) & np.isfinite(pred_h)

        ax.scatter(
            actual_h[valid],
            pred_h[valid],
            s=8,
            alpha=0.22,
            color="#1f77b4",
            edgecolors="none",
            rasterized=True,
        )
        ax.plot([0, max_val], [0, max_val], linestyle="--", linewidth=1.2, color="black")
        ax.set_title(LEAD_LABELS[horizon_idx], fontsize=10)
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.grid(alpha=0.2)

        stats = horizon_summary(actual_h, pred_h)
        stats["horizon_index"] = horizon_idx + 1
        stats["horizon_label"] = LEAD_LABELS[horizon_idx]
        summary_rows.append(stats)
        ax.text(
            0.03,
            0.97,
            f"r={stats['corr']:.2f}\nRMSE={stats['rmse']:.0f}\nBias={stats['bias']:.0f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    for ax in axes[-1, :]:
        ax.set_xlabel("Actual SWE (mm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted SWE (mm)")

    fig.suptitle(title, fontsize=16)
    fig.savefig(png_path, dpi=260, bbox_inches="tight")
    fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)


def main():
    actual, pred = load_planb_arrays()

    full_png = PLANB_ROOT / "planB_pred_vs_actual_24weeks_full512.png"
    full_summary = PLANB_ROOT / "planB_pred_vs_actual_24weeks_full512_summary.csv"
    make_plot(
        actual,
        pred,
        "Plan B 24-week Prediction vs Actual by Forecast Horizon (Full 512 Stations)",
        full_png,
        full_summary,
    )

    sierra_indices, sierra_df = load_sierra_indices()
    sierra_png = PLANB_ROOT / "planB_pred_vs_actual_24weeks_sierra43.png"
    sierra_summary = PLANB_ROOT / "planB_pred_vs_actual_24weeks_sierra43_summary.csv"
    make_plot(
        actual[:, sierra_indices, :],
        pred[:, sierra_indices, :],
        "Plan B 24-week Prediction vs Actual by Forecast Horizon (Sierra 43 Stations)",
        sierra_png,
        sierra_summary,
    )

    sierra_df.to_csv(PLANB_ROOT / "planB_pred_vs_actual_24weeks_sierra43_stations.csv", index=False)

    print("Saved", full_png)
    print("Saved", sierra_png)
    print("Saved", full_summary)
    print("Saved", sierra_summary)


if __name__ == "__main__":
    main()
