from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PLANB_ROOT = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week")
SUMMARY_CSV = PLANB_ROOT / "planB_pred_vs_actual_threshold_summary.csv"
THRESHOLDS = [0.0, 50.0, 100.0]
COLORS = {0.0: "#1f77b4", 50.0: "#ff7f0e", 100.0: "#2ca02c"}


def plot_group(df, group_name, out_prefix):
    sub = df[df["group"] == group_name].copy()
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), constrained_layout=True, sharex=True)
    metric_specs = [
        ("rmse", "RMSE (mm)"),
        ("bias", "Bias (mm)"),
        ("n_points", "Sample Count"),
    ]

    for ax, (metric, ylabel) in zip(axes, metric_specs):
        for threshold in THRESHOLDS:
            thr = sub[sub["threshold_mm"] == threshold].sort_values("horizon_index")
            ax.plot(
                thr["horizon_index"],
                thr[metric],
                marker="o",
                linewidth=2,
                markersize=4,
                color=COLORS[threshold],
                label=f"actual SWE > {int(threshold)} mm",
            )
        if metric == "bias":
            ax.axhline(0, linestyle="--", color="black", linewidth=1)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(frameon=True)

    axes[-1].set_xlabel("Forecast Horizon Week")
    axes[-1].set_xticks(range(1, 25))
    fig.suptitle(f"Plan B Thresholded Weekly Metrics by Forecast Horizon ({group_name})", fontsize=15)
    png_path = PLANB_ROOT / f"{out_prefix}.png"
    pdf_path = PLANB_ROOT / f"{out_prefix}.pdf"
    fig.savefig(png_path, dpi=260, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main():
    df = pd.read_csv(SUMMARY_CSV)
    paths = []
    paths.extend(plot_group(df, "full512", "planB_threshold_metrics_full512"))
    paths.extend(plot_group(df, "sierra43", "planB_threshold_metrics_sierra43"))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
