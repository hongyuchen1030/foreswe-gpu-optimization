from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra")
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "fig6_weekly_stitching"
SIERRA_STATION_SOURCE = (
    PROJECT_ROOT
    / "experiments"
    / "foreswe_domain_sensitivity"
    / "D2_sierra_plus_500km"
    / "station_subset.csv"
)
WEEKLY_RESULTS_ROOT = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_weekly_reproduction_full512/results"
)

TEST_YEAR = 2017
FORECASTING_WINDOW = 4
FORECASTING_STEP_SIZE = 7


def load_sierra_station_frame():
    df = pd.read_csv(SIERRA_STATION_SOURCE)
    sierra = df[df["is_sierra_eval_station"] == True].copy()
    sierra = sierra.sort_values("original_station_index")
    return sierra


def load_year_files(year, station_indices):
    actual_path = WEEKLY_RESULTS_ROOT / "actual" / f"{year}_weekly_window_actual.pkl"
    mean_window_path = WEEKLY_RESULTS_ROOT / "predict" / f"{year}_weekly_gp_window_predict.pkl"
    low_window_path = WEEKLY_RESULTS_ROOT / "predict" / f"{year}_weekly_gp_low_window_predict.pkl"
    high_window_path = WEEKLY_RESULTS_ROOT / "predict" / f"{year}_weekly_gp_high_window_predict.pkl"
    mean_stitched_path = WEEKLY_RESULTS_ROOT / "predict" / f"{year}_weekly_gp_predict.pkl"
    low_stitched_path = WEEKLY_RESULTS_ROOT / "predict" / f"{year}_weekly_gp_lower_predict.pkl"
    high_stitched_path = WEEKLY_RESULTS_ROOT / "predict" / f"{year}_weekly_gp_higher_predict.pkl"

    files = [
        actual_path,
        mean_window_path,
        low_window_path,
        high_window_path,
        mean_stitched_path,
        low_stitched_path,
        high_stitched_path,
    ]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing weekly GP files:\n" + "\n".join(missing))

    def sub(arr):
        arr = np.asarray(arr)
        if arr.ndim >= 2:
            return arr[:, station_indices, :] if arr.shape[0] != len(station_indices) else arr[station_indices, :, :]
        return arr

    return {
        "actual_window": (actual_path, sub(joblib.load(actual_path))),
        "mean_window": (mean_window_path, sub(joblib.load(mean_window_path))),
        "low_window": (low_window_path, sub(joblib.load(low_window_path))),
        "high_window": (high_window_path, sub(joblib.load(high_window_path))),
        "mean_stitched": (mean_stitched_path, np.asarray(joblib.load(mean_stitched_path))[station_indices, :, :]),
        "low_stitched": (low_stitched_path, np.asarray(joblib.load(low_stitched_path))[station_indices, :, :]),
        "high_stitched": (high_stitched_path, np.asarray(joblib.load(high_stitched_path))[station_indices, :, :]),
    }


def classify_shape(arr):
    shape = tuple(arr.shape)
    if len(shape) == 3 and shape[-1] == 4:
        return "window_form_[num_issue_times,num_stations,4]"
    if len(shape) == 3 and shape[-1] == 1:
        return "stitched_form_[num_stations,num_time_points,1]"
    return "other_%s" % (shape,)


def choose_station(actual_window, sierra_df):
    station_max = np.nanmax(actual_window, axis=(0, 2))
    local_index = int(np.nanargmax(station_max))
    row = sierra_df.iloc[local_index]
    return local_index, row, float(station_max[local_index])


def build_stitched_chunks(actual_window, mean_window, low_window, high_window, station_index):
    issue_indices = list(range(0, actual_window.shape[0], FORECASTING_WINDOW * FORECASTING_STEP_SIZE))
    actual_chunks = []
    mean_chunks = []
    low_chunks = []
    high_chunks = []
    for issue_idx in issue_indices:
        actual_chunks.append(actual_window[issue_idx, station_index, :])
        mean_chunks.append(mean_window[issue_idx, station_index, :])
        low_chunks.append(low_window[issue_idx, station_index, :])
        high_chunks.append(high_window[issue_idx, station_index, :])
    return {
        "issue_indices": issue_indices,
        "actual": np.concatenate(actual_chunks),
        "mean": np.concatenate(mean_chunks),
        "low": np.concatenate(low_chunks),
        "high": np.concatenate(high_chunks),
    }


def make_plot(stitched, station_row):
    x = np.arange(len(stitched["mean"]))
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.plot(x, stitched["actual"], color="black", linewidth=2, label="Actual SWE")
    ax.plot(x, stitched["mean"], color="#1f77b4", linewidth=2, label="ForeSWE GP mean")
    ax.fill_between(x, stitched["low"], stitched["high"], color="#1f77b4", alpha=0.2, label="Prediction interval")
    for boundary in range(4, len(x), 4):
        ax.axvline(boundary - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Stitched weekly forecast point")
    ax.set_ylabel("SWE (mm)")
    ax.set_title(
        "Stitched 4-week ForeSWE weekly GP windows (Sierra 43)\nYear %s, station %s"
        % (TEST_YEAR, station_row["Station Name"])
    )
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", frameon=True)
    out_path = OUTPUT_DIR / "weekly_stitched_4week_windows_sierra43.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_summary(loaded, station_row, station_max, stitched):
    lines = [
        "# Sierra 43 Weekly ForeSWE Stitching Summary",
        "",
        "- Year selected: `%s`" % TEST_YEAR,
        "- Sierra station count: `43`",
        "- Station selected automatically: `%s` (original index `%s`, state `%s`)"
        % (station_row["Station Name"], station_row["original_station_index"], station_row["State"]),
        "- Selection rule: Sierra station with highest maximum actual SWE in the selected year",
        "- Station maximum actual SWE: `%.3f` mm" % station_max,
        "- Stitched issue indices: `%s`" % stitched["issue_indices"],
        "",
        "Loaded shapes:",
    ]
    for name, (path, arr) in loaded.items():
        lines.append("- `%s`: `%s`, shape `%s`, classification `%s`" % (name, path, tuple(arr.shape), classify_shape(arr)))
    lines.extend(
        [
            "",
            "The same stitching logic is used as the full-domain plot: non-overlapping 4-week chunks from issue indices spaced by 28 days.",
        ]
    )
    out_path = OUTPUT_DIR / "weekly_stitching_summary_sierra43.md"
    out_path.write_text("\n".join(lines))
    return out_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sierra_df = load_sierra_station_frame()
    station_indices = sierra_df["original_station_index"].astype(int).to_numpy()
    loaded = load_year_files(TEST_YEAR, station_indices)
    actual_window = loaded["actual_window"][1]
    mean_window = loaded["mean_window"][1]
    low_window = loaded["low_window"][1]
    high_window = loaded["high_window"][1]
    local_index, station_row, station_max = choose_station(actual_window, sierra_df)
    stitched = build_stitched_chunks(actual_window, mean_window, low_window, high_window, local_index)
    plot_path = make_plot(stitched, station_row)
    summary_path = write_summary(loaded, station_row, station_max, stitched)
    print("Saved:", plot_path)
    print("Saved:", summary_path)
    print("Selected Sierra station:", station_row["Station Name"], int(station_row["original_station_index"]))


if __name__ == "__main__":
    main()
