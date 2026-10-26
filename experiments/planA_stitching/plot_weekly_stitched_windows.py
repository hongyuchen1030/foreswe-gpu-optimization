from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path("/global/homes/h/hyvchen/foreswe-gpu-optimization")
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "planA_stitching"
THIRDPARTY_ROOT = PROJECT_ROOT
WEEKLY_RESULTS_ROOT = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_weekly_reproduction_full512/results"
)

TEST_YEAR = 2017
FORECASTING_WINDOW = 4
FORECASTING_STEP_SIZE = 7


def classify_shape(arr):
    shape = tuple(arr.shape)
    if len(shape) == 3 and shape[-1] == 4:
        return "window_form_[num_issue_times,num_stations,4]"
    if len(shape) == 3 and shape[-1] == 1:
        return "stitched_form_[num_stations,num_time_points,1]"
    return f"other_{shape}"


def load_year_files(year):
    actual_path = WEEKLY_RESULTS_ROOT / "actual" / f"{year}_weekly_window_actual.pkl"
    mean_window_path = WEEKLY_RESULTS_ROOT / "predict" / f"{year}_weekly_gp_window_predict.pkl"
    low_window_path = WEEKLY_RESULTS_ROOT / "predict" / f"{year}_weekly_gp_low_window_predict.pkl"
    high_window_path = WEEKLY_RESULTS_ROOT / "predict" / f"{year}_weekly_gp_high_window_predict.pkl"
    mean_stitched_path = WEEKLY_RESULTS_ROOT / "predict" / f"{year}_weekly_gp_predict.pkl"
    low_stitched_path = WEEKLY_RESULTS_ROOT / "predict" / f"{year}_weekly_gp_lower_predict.pkl"
    high_stitched_path = WEEKLY_RESULTS_ROOT / "predict" / f"{year}_weekly_gp_higher_predict.pkl"

    required = [
        actual_path,
        mean_window_path,
        low_window_path,
        high_window_path,
        mean_stitched_path,
        low_stitched_path,
        high_stitched_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing weekly GP files:\n" + "\n".join(missing))

    return {
        "actual_window": (actual_path, np.asarray(joblib.load(actual_path))),
        "mean_window": (mean_window_path, np.asarray(joblib.load(mean_window_path))),
        "low_window": (low_window_path, np.asarray(joblib.load(low_window_path))),
        "high_window": (high_window_path, np.asarray(joblib.load(high_window_path))),
        "mean_stitched": (mean_stitched_path, np.asarray(joblib.load(mean_stitched_path))),
        "low_stitched": (low_stitched_path, np.asarray(joblib.load(low_stitched_path))),
        "high_stitched": (high_stitched_path, np.asarray(joblib.load(high_stitched_path))),
    }


def choose_station(actual_window):
    station_max = np.nanmax(actual_window, axis=(0, 2))
    station_index = int(np.nanargmax(station_max))
    return station_index, float(station_max[station_index])


def choose_issue_indices(n_issue_times):
    preferred = [8, 12, 16, 20]
    selected = [idx for idx in preferred if idx < n_issue_times]
    if not selected:
        selected = list(range(min(4, n_issue_times)))
    return selected


def build_stitched_chunks(actual_window, mean_window, low_window, high_window, station_index):
    issue_indices = list(range(0, actual_window.shape[0], FORECASTING_WINDOW * FORECASTING_STEP_SIZE))

    actual_chunks = []
    mean_chunks = []
    low_chunks = []
    high_chunks = []
    chunk_labels = []

    for issue_idx in issue_indices:
        actual_chunks.append(actual_window[issue_idx, station_index, :])
        mean_chunks.append(mean_window[issue_idx, station_index, :])
        low_chunks.append(low_window[issue_idx, station_index, :])
        high_chunks.append(high_window[issue_idx, station_index, :])
        chunk_labels.extend(
            [
                f"issue {issue_idx} -> t+7",
                f"issue {issue_idx} -> t+14",
                f"issue {issue_idx} -> t+21",
                f"issue {issue_idx} -> t+28",
            ]
        )

    return {
        "issue_indices": issue_indices,
        "actual": np.concatenate(actual_chunks),
        "mean": np.concatenate(mean_chunks),
        "low": np.concatenate(low_chunks),
        "high": np.concatenate(high_chunks),
        "labels": chunk_labels,
    }


def write_shapes_report(arrays):
    lines = []
    for name, (path, arr) in arrays.items():
        lines.append(f"{name}")
        lines.append(f"file: {path}")
        lines.append(f"shape: {tuple(arr.shape)}")
        lines.append(f"classification: {classify_shape(arr)}")
        lines.append("")
    out_path = OUTPUT_DIR / "weekly_window_shapes.txt"
    out_path.write_text("\n".join(lines))
    return out_path


def make_plot_a(actual_window, mean_window, low_window, high_window, station_index, issue_indices):
    fig, axes = plt.subplots(1, len(issue_indices), figsize=(4.2 * len(issue_indices), 4.8), sharey=True, constrained_layout=True)
    if len(issue_indices) == 1:
        axes = [axes]

    x = np.arange(1, FORECASTING_WINDOW + 1)
    x_labels = ["t+7", "t+14", "t+21", "t+28"]

    for ax, issue_idx in zip(axes, issue_indices):
        actual = actual_window[issue_idx, station_index, :]
        mean = mean_window[issue_idx, station_index, :]
        low = low_window[issue_idx, station_index, :]
        high = high_window[issue_idx, station_index, :]

        ax.plot(x, actual, color="black", marker="o", linewidth=2, label="Actual")
        ax.plot(x, mean, color="#1f77b4", marker="o", linewidth=2, label="GP mean")
        ax.fill_between(x, low, high, color="#1f77b4", alpha=0.2, label="PI")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=0)
        ax.set_title(f"Issue index {issue_idx}")
        ax.set_xlabel("Forecast lead")
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("SWE (mm)")
    axes[0].legend(loc="upper left", frameon=True)
    fig.suptitle(f"Raw 4-week ForeSWE weekly GP windows\nStation {station_index}", fontsize=14)
    out_path = OUTPUT_DIR / "weekly_raw_4lead_windows.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_plot_b(stitched, station_index, year):
    x = np.arange(len(stitched["mean"]))

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.plot(x, stitched["actual"], color="black", linewidth=2, label="Actual SWE")
    ax.plot(x, stitched["mean"], color="#1f77b4", linewidth=2, label="ForeSWE GP mean")
    ax.fill_between(x, stitched["low"], stitched["high"], color="#1f77b4", alpha=0.2, label="Prediction interval")

    for boundary in range(4, len(x), 4):
        ax.axvline(boundary - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Stitched weekly forecast point")
    ax.set_ylabel("SWE (mm)")
    ax.set_title(f"Stitched 4-week ForeSWE weekly GP windows\nYear {year}, station {station_index}")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", frameon=True)

    out_path = OUTPUT_DIR / "weekly_stitched_4week_windows.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_summary(loaded, station_index, station_max, year, issue_indices, stitched):
    lines = [
        "# Weekly ForeSWE Stitching Summary",
        "",
        "This analysis uses existing weekly GP outputs only. No model code was modified or rerun.",
        "",
        "## Files loaded",
    ]

    for name, (path, arr) in loaded.items():
        lines.append(f"- `{name}`: `{path}`")
        lines.append(f"  shape: `{tuple(arr.shape)}`")
        lines.append(f"  classification: `{classify_shape(arr)}`")

    lines.extend(
        [
            "",
            "## Selection",
            f"- Year selected: `{year}`",
            f"- Station selected automatically: `{station_index}`",
            f"- Station selection rule: station with highest maximum actual SWE in `{year}`",
            f"- Station maximum actual SWE: `{station_max:.3f}` mm",
            f"- Plot A issue indices: `{issue_indices}`",
            "",
            "## Stitching logic",
            f"- Forecasting window from code: `{FORECASTING_WINDOW}`",
            f"- Forecasting step size from code: `{FORECASTING_STEP_SIZE}` days",
            "- Each weekly issue date contributes exactly four leads: `t+7, t+14, t+21, t+28`.",
            f"- Stitched issue indices used for Plot B: `{stitched['issue_indices']}`",
            "- Plot B concatenates non-overlapping 4-week chunks from successive issue indices spaced by `forecasting_window * forecasting_step_size = 28` days.",
            "- This matches the behavior in `getDailyPredictions()` where the loop steps by `forecasting_window * forecasting_step_size` and appends one 4-lead block per selected issue date.",
            "",
            "## Confirmation from outputs",
            "- The `*_weekly_gp_window_predict.pkl` arrays are window-form outputs with a last dimension of `4`, confirming four simultaneous weekly leads.",
            "- The `*_weekly_gp_predict.pkl` arrays are stitched-form outputs with a last dimension of `1`, consistent with a longer weekly curve assembled from repeated 4-lead forecasts.",
            "",
            "## Ambiguity remaining",
            "- The code constructs stitched outputs, but the original GitHub repo does not include Figure 6 plotting code itself.",
            "- The original upstream weekly attention script computes longer stitched arrays in memory but does not save them in the upstream file shown here; the saved stitched arrays used here come from the weekly GP stage.",
        ]
    )

    out_path = OUTPUT_DIR / "weekly_stitching_summary.md"
    out_path.write_text("\n".join(lines))
    return out_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    loaded = load_year_files(TEST_YEAR)
    shapes_path = write_shapes_report(loaded)

    actual_window = loaded["actual_window"][1]
    mean_window = loaded["mean_window"][1]
    low_window = loaded["low_window"][1]
    high_window = loaded["high_window"][1]

    station_index, station_max = choose_station(actual_window)
    issue_indices = choose_issue_indices(actual_window.shape[0])
    stitched = build_stitched_chunks(actual_window, mean_window, low_window, high_window, station_index)

    plot_a_path = make_plot_a(actual_window, mean_window, low_window, high_window, station_index, issue_indices)
    plot_b_path = make_plot_b(stitched, station_index, TEST_YEAR)
    summary_path = write_summary(loaded, station_index, station_max, TEST_YEAR, issue_indices, stitched)

    print("Commands reference:")
    print("  find results -maxdepth 3 -type f | grep -E \"weekly|gp|actual|predict|low|high|window\"")
    print("Generated:")
    print(" ", shapes_path)
    print(" ", plot_a_path)
    print(" ", plot_b_path)
    print(" ", summary_path)
    print("Selected year:", TEST_YEAR)
    print("Selected station index:", station_index)
    print("Issue indices for raw windows:", issue_indices)
    print("Issue indices for stitched plot:", stitched["issue_indices"])


if __name__ == "__main__":
    main()
