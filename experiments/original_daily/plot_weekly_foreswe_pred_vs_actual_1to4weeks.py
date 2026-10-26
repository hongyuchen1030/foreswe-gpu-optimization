from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path("/global/homes/h/hyvchen/foreswe-gpu-optimization")
FIGURES_DIR = REPO_ROOT / "experiments" / "original_daily" / "figures"
WEEKLY_ACTUAL_DIR = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_weekly_reproduction_full512/results/actual"
)
WEEKLY_PREDICT_DIR = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_weekly_reproduction_full512/results/predict"
)
TEST_YEARS = [2014, 2015, 2016, 2017, 2018]
HORIZON_LABELS = ["Week 1 (7d)", "Week 2 (14d)", "Week 3 (21d)", "Week 4 (28d)"]


def load_weekly_arrays():
    actuals = []
    preds = []
    alignment_rows = []
    for year in TEST_YEARS:
        actual = np.asarray(joblib.load(WEEKLY_ACTUAL_DIR / f"{year}_weekly_window_actual.pkl"))
        pred = np.asarray(joblib.load(WEEKLY_PREDICT_DIR / f"{year}_weekly_gp_window_predict.pkl"))
        common_length = min(actual.shape[0], pred.shape[0])
        actuals.append(actual[:common_length])
        preds.append(pred[:common_length])
        alignment_rows.append(
            {
                "year": year,
                "actual_windows": actual.shape[0],
                "pred_windows": pred.shape[0],
                "used_windows": common_length,
            }
        )
    return np.concatenate(actuals, axis=0), np.concatenate(preds, axis=0), pd.DataFrame(alignment_rows)


def horizon_summary(actual_h, pred_h):
    valid = np.isfinite(actual_h) & np.isfinite(pred_h)
    x = actual_h[valid]
    y = pred_h[valid]
    rmse = float(np.sqrt(np.mean((y - x) ** 2)))
    bias = float(np.mean(y - x))
    corr = float(np.corrcoef(x, y)[0, 1])
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


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    actual, pred, align_df = load_weekly_arrays()
    max_val = float(np.nanmax([actual, pred]))

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), constrained_layout=True, sharex=True, sharey=True)
    summary_rows = []

    for horizon_idx, ax in enumerate(axes):
        actual_h = actual[:, :, horizon_idx].reshape(-1)
        pred_h = pred[:, :, horizon_idx].reshape(-1)
        valid = np.isfinite(actual_h) & np.isfinite(pred_h)

        ax.scatter(
            actual_h[valid],
            pred_h[valid],
            s=4,
            alpha=0.18,
            color="#1f77b4",
            edgecolors="none",
            rasterized=True,
        )
        ax.plot([0, max_val], [0, max_val], linestyle="--", linewidth=1.5, color="black")
        ax.set_title(HORIZON_LABELS[horizon_idx])
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.grid(alpha=0.2)

        stats = horizon_summary(actual_h, pred_h)
        stats["horizon_index"] = horizon_idx + 1
        stats["horizon_label"] = HORIZON_LABELS[horizon_idx]
        summary_rows.append(stats)
        ax.text(
            0.03,
            0.97,
            f"r={stats['corr']:.3f}\nRMSE={stats['rmse']:.1f}\nBias={stats['bias']:.1f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    axes[0].set_ylabel("Predicted SWE (mm)")
    for ax in axes:
        ax.set_xlabel("Actual SWE (mm)")

    fig.suptitle("ForeSWE Weekly Prediction vs Actual by Forecast Horizon", fontsize=14)

    png_path = FIGURES_DIR / "weekly_foreswe_pred_vs_actual_1to4weeks.png"
    pdf_path = FIGURES_DIR / "weekly_foreswe_pred_vs_actual_1to4weeks.pdf"
    summary_csv = FIGURES_DIR / "weekly_foreswe_pred_vs_actual_1to4weeks_summary.csv"
    align_csv = FIGURES_DIR / "weekly_foreswe_pred_vs_actual_1to4weeks_alignment.csv"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    align_df.to_csv(align_csv, index=False)

    print("Saved:", png_path)
    print("Saved:", pdf_path)
    print("Saved:", summary_csv)
    print("Saved:", align_csv)


if __name__ == "__main__":
    main()
