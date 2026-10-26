from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PLANB_ROOT = Path("/global/homes/h/hyvchen/foreswe-gpu-optimization/experiments/planB_24week")
PROJECT_ROOT = Path("/global/homes/h/hyvchen/foreswe-gpu-optimization")
SIERRA_STATION_SOURCE = (
    PROJECT_ROOT
    / "experiments"
    / "domain_sensitivity"
    / "D2_sierra_plus_500km"
    / "station_subset.csv"
)
TEST_YEARS = [2015, 2016, 2017, 2018, 2019]
BINS = [
    ("0to50", 0.0, 50.0, "0 < Actual SWE <= 50 mm"),
    ("50to100", 50.0, 100.0, "50 < Actual SWE <= 100 mm"),
    ("gt100", 100.0, np.inf, "Actual SWE > 100 mm"),
]


def load_arrays():
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


def summarize(x, y):
    if x.size == 0:
        return {"n_points": 0, "corr": np.nan, "rmse": np.nan, "bias": np.nan}
    return {
        "n_points": int(x.size),
        "corr": float(np.corrcoef(x, y)[0, 1]) if x.size > 1 else np.nan,
        "rmse": float(np.sqrt(np.mean((y - x) ** 2))),
        "bias": float(np.mean(y - x)),
    }


def make_plot(actual, pred, title, out_prefix):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), constrained_layout=True)
    summary_rows = []

    for ax, (bin_slug, low, high, label) in zip(axes, BINS):
        actual_flat = actual.reshape(-1)
        pred_flat = pred.reshape(-1)
        valid = np.isfinite(actual_flat) & np.isfinite(pred_flat) & (actual_flat > low) & (actual_flat <= high)
        x = actual_flat[valid]
        y = pred_flat[valid]
        stats = summarize(x, y)
        summary_rows.append(
            {
                "bin_slug": bin_slug,
                "bin_label": label,
                **stats,
            }
        )

        if x.size > 0:
            local_max = float(np.nanmax([x, y]))
            local_max = max(local_max, 1.0)
            ax.scatter(
                x,
                y,
                s=8,
                alpha=0.22,
                color="#1f77b4",
                edgecolors="none",
                rasterized=True,
            )
            ax.plot([0, local_max], [0, local_max], linestyle="--", linewidth=1.2, color="black")
            ax.set_xlim(0, local_max)
            ax.set_ylim(0, local_max)
        ax.set_title(label)
        ax.set_xlabel("Actual SWE (mm)")
        ax.grid(alpha=0.2)
        ax.text(
            0.03,
            0.97,
            f"n={stats['n_points']}\nr={stats['corr']:.2f}\nRMSE={stats['rmse']:.0f}\nBias={stats['bias']:.0f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    axes[0].set_ylabel("Predicted SWE (mm)")
    fig.suptitle(title, fontsize=15)

    png_path = PLANB_ROOT / f"{out_prefix}.png"
    pdf_path = PLANB_ROOT / f"{out_prefix}.pdf"
    fig.savefig(png_path, dpi=260, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    summary_path = PLANB_ROOT / f"{out_prefix}_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    return png_path, pdf_path, summary_path


def main():
    actual, pred = load_arrays()
    sierra_indices, sierra_df = load_sierra_indices()

    outputs = []
    outputs.append(
        make_plot(
            actual,
            pred,
            "Plan B Predicted vs Actual SWE (Full 512 Stations)",
            "planB_pred_vs_actual_full512_3panel",
        )
    )
    outputs.append(
        make_plot(
            actual[:, sierra_indices, :],
            pred[:, sierra_indices, :],
            "Plan B Predicted vs Actual SWE (Sierra 43 Stations)",
            "planB_pred_vs_actual_sierra43_3panel",
        )
    )

    sierra_df.to_csv(PLANB_ROOT / "planB_pred_vs_actual_sierra43_3panel_stations.csv", index=False)

    for group in outputs:
        for path in group:
            print(path)


if __name__ == "__main__":
    main()
