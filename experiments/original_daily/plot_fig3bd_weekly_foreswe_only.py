import os
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


REPO_ROOT = Path("/global/homes/h/hyvchen/foreswe-gpu-optimization")
FIGURES_DIR = REPO_ROOT / "experiments" / "original_daily" / "figures"
METADATA_PATH = REPO_ROOT / "data" / "snotel_location_info_2.csv"

WEEKLY_ACTUAL_DIR = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_weekly_reproduction_full512/results/actual"
)
WEEKLY_PREDICT_DIR = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_weekly_reproduction_full512/results/predict"
)

TEST_YEARS = [2014, 2015, 2016, 2017, 2018]

# Paper-style bins from the Figure 3 legend, ordered from best to worst.
BIN_SPECS = [
    (">=0.9", 0.9, np.inf, "#2647a7"),
    (">=0.75 & <0.9", 0.75, 0.9, "#2d86c7"),
    (">=0.5 & <0.75", 0.5, 0.75, "#2fb2d4"),
    (">=0.3 & <0.5", 0.3, 0.5, "#4ec6c2"),
    (">=0 & <0.3", 0.0, 0.3, "#a9e3b6"),
    ("<0", -np.inf, 0.0, "#ddee9b"),
]


def load_concat_arrays():
    actuals = []
    preds = []
    year_lengths = []
    for year in TEST_YEARS:
        actual = np.asarray(joblib.load(WEEKLY_ACTUAL_DIR / f"{year}_weekly_window_actual.pkl"))
        pred = np.asarray(joblib.load(WEEKLY_PREDICT_DIR / f"{year}_weekly_gp_window_predict.pkl"))
        common_length = min(actual.shape[0], pred.shape[0])
        actuals.append(actual[:common_length])
        preds.append(pred[:common_length])
        year_lengths.append((year, actual.shape[0], pred.shape[0], common_length))
    return np.concatenate(actuals, axis=0), np.concatenate(preds, axis=0), year_lengths


def nse_per_station(actual, pred):
    station_nse = np.full(actual.shape[1], np.nan, dtype=np.float64)
    for station_idx in range(actual.shape[1]):
        y_true = actual[:, station_idx, :].reshape(-1)
        y_pred = pred[:, station_idx, :].reshape(-1)
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        if not np.any(valid):
            continue
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        denom = np.sum((y_true - np.mean(y_true)) ** 2)
        if denom <= 0:
            continue
        station_nse[station_idx] = 1.0 - np.sum((y_pred - y_true) ** 2) / denom
    return station_nse


def bin_counts(values):
    counts = []
    for _label, lower, upper, _color in BIN_SPECS:
        counts.append(int(np.sum((values >= lower) & (values < upper))))
    return counts


def build_rmp_curve(station_nse):
    # With only ForeSWE available, use deviation from the perfect NSE score of 1.0.
    deviations = 1.0 - station_nse[np.isfinite(station_nse)]
    deviations = np.sort(np.clip(deviations, 0.0, None))
    y = np.arange(1, len(deviations) + 1, dtype=np.float64) / len(deviations) * 100.0
    return deviations, y


def save_station_table(metadata, station_nse):
    table = metadata.copy()
    table["weekly_foreswe_nse"] = station_nse
    table["weekly_foreswe_deviation_from_perfect_nse"] = 1.0 - station_nse
    out_csv = FIGURES_DIR / "reproduced_fig3bd_weekly_foreswe_only_station_nse.csv"
    table.to_csv(out_csv, index=False)
    return out_csv


def make_plot(station_nse, counts):
    valid_count = int(np.sum(np.isfinite(station_nse)))
    nse_gt_075 = int(np.sum(station_nse >= 0.75))
    deviations, yvals = build_rmp_curve(station_nse)

    fig, (ax_bar, ax_rmp) = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    left = 0
    for count, (label, _lower, _upper, color) in zip(counts, BIN_SPECS):
        ax_bar.barh(["ForeSWE"], [count], left=left, color=color, edgecolor="none", label=label)
        left += count

    ax_bar.axvline(nse_gt_075, color="red", linewidth=4)
    ax_bar.set_xlim(0, valid_count)
    ax_bar.set_xlabel("Locations count")
    ax_bar.set_title("(b) NSE for locations (weekly)")
    ax_bar.grid(axis="x", alpha=0.2)

    ax_rmp.plot(deviations, yvals, color="#1f77b4", linewidth=2.5, marker="o", markersize=3, label="ForeSWE")
    ax_rmp.set_xlim(0, 2.0)
    ax_rmp.set_ylim(0, 100)
    ax_rmp.set_xlabel("Deviation from perfect NSE")
    ax_rmp.set_ylabel("Locations Count (%)")
    ax_rmp.set_title("(d) RMP: NSE (weekly, ForeSWE only)")
    ax_rmp.grid(alpha=0.2)
    ax_rmp.legend(loc="lower right", frameon=True)

    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for _label, _lower, _upper, color in BIN_SPECS]
    labels = [label for label, _lower, _upper, _color in BIN_SPECS]
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.08), frameon=True)

    return fig


def write_note(valid_count, counts, nse_csv):
    note_path = FIGURES_DIR / "reproduced_fig3bd_weekly_foreswe_only_NOTE.md"
    lines = [
        "# ForeSWE-only Figure 3b/3d Reproduction",
        "",
        "This partial reproduction uses only the finished weekly ForeSWE run.",
        "",
        "- Weekly actuals: `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_weekly_reproduction_full512/results/actual/*_weekly_window_actual.pkl`",
        "- Weekly predictions: `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_weekly_reproduction_full512/results/predict/*_weekly_gp_window_predict.pkl`",
        f"- Valid station count: `{valid_count}`",
        f"- NSE bin counts: `{counts}`",
        f"- Station-level NSE table: `{nse_csv}`",
        "",
        "Method summary:",
        "- For each station, all weekly test windows from 2014-2018 were concatenated.",
        "- Weekly NSE was computed by flattening all forecast windows and all 4 weekly horizons for that station.",
        "- Panel (b) is the weekly NSE bin distribution for ForeSWE only.",
        "- Panel (d) is a ForeSWE-only proxy for the paper's RMP view: it plots the cumulative distribution of `1 - NSE`, which is the deviation from a perfect NSE score of 1.0.",
        "- Because the other comparison models were not rerun here, this is not the full cross-model RMP used in the paper.",
        "",
        "![ForeSWE-only weekly Figure 3b/3d](reproduced_fig3bd_weekly_foreswe_only.png)",
    ]
    note_path.write_text("\n".join(lines))
    return note_path


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(METADATA_PATH)
    actual, pred, year_lengths = load_concat_arrays()
    station_nse = nse_per_station(actual, pred)
    counts = bin_counts(station_nse[np.isfinite(station_nse)])
    fig = make_plot(station_nse, counts)

    png_path = FIGURES_DIR / "reproduced_fig3bd_weekly_foreswe_only.png"
    pdf_path = FIGURES_DIR / "reproduced_fig3bd_weekly_foreswe_only.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    nse_csv = save_station_table(metadata, station_nse)
    note_path = write_note(int(np.sum(np.isfinite(station_nse))), counts, nse_csv.name)

    print("Year lengths (actual, pred, used):", year_lengths)
    print("Saved:", png_path)
    print("Saved:", pdf_path)
    print("Saved:", nse_csv)
    print("Saved:", note_path)


if __name__ == "__main__":
    main()
