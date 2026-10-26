from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra")
OUTDIR = PROJECT_ROOT / "experiments" / "fig6_weekly_stitching"
WEEKLY_RESULTS_ROOT = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_weekly_reproduction_full512/results"
)
SIERRA_STATION_SOURCE = (
    PROJECT_ROOT
    / "experiments"
    / "foreswe_domain_sensitivity"
    / "D2_sierra_plus_500km"
    / "station_subset.csv"
)

YEAR = 2017
ISSUE_WEEKS = [8, 12, 16, 20]
FORECASTING_WINDOW = 4


def load_arrays():
    return {
        "actual": np.asarray(joblib.load(WEEKLY_RESULTS_ROOT / "actual" / f"{YEAR}_weekly_window_actual.pkl")),
        "mean": np.asarray(joblib.load(WEEKLY_RESULTS_ROOT / "predict" / f"{YEAR}_weekly_gp_window_predict.pkl")),
        "low": np.asarray(joblib.load(WEEKLY_RESULTS_ROOT / "predict" / f"{YEAR}_weekly_gp_low_window_predict.pkl")),
        "high": np.asarray(joblib.load(WEEKLY_RESULTS_ROOT / "predict" / f"{YEAR}_weekly_gp_high_window_predict.pkl")),
    }


def choose_station(actual_window, station_indices=None):
    arr = actual_window if station_indices is None else actual_window[:, station_indices, :]
    station_max = np.nanmax(arr, axis=(0, 2))
    local_idx = int(np.nanargmax(station_max))
    if station_indices is None:
        return local_idx, float(station_max[local_idx])
    return int(station_indices[local_idx]), float(station_max[local_idx])


def build_fig6_series(arrays, station_index):
    stitched_weeks = []
    actual = []
    mean = []
    low = []
    high = []
    issue_rows = []

    for issue_week in ISSUE_WEEKS:
        issue_idx = (issue_week - 1) * 7
        if issue_idx >= arrays["actual"].shape[0]:
            continue
        target_weeks = [issue_week + lead for lead in range(1, FORECASTING_WINDOW + 1)]
        stitched_weeks.extend(target_weeks)
        actual.extend(arrays["actual"][issue_idx, station_index, :].tolist())
        mean.extend(arrays["mean"][issue_idx, station_index, :].tolist())
        low.extend(arrays["low"][issue_idx, station_index, :].tolist())
        high.extend(arrays["high"][issue_idx, station_index, :].tolist())
        issue_rows.append(
            {
                "issue_week": issue_week,
                "issue_row_index": issue_idx,
                "target_weeks": target_weeks,
            }
        )

    return {
        "weeks": np.array(stitched_weeks),
        "actual": np.array(actual),
        "mean": np.array(mean),
        "low": np.array(low),
        "high": np.array(high),
        "issues": issue_rows,
    }


def plot_series(series, title, out_path):
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(series["weeks"], series["actual"], color="red", linewidth=2.2, label="Actual SWE")
    ax.plot(series["weeks"], series["mean"], color="green", linewidth=2.2, label="Forecasted SWE")
    ax.fill_between(series["weeks"], series["low"], series["high"], color="#4da3d9", alpha=0.35, label="Prediction Interval")
    for boundary in [12.5, 16.5, 20.5]:
        ax.axvline(boundary, linestyle="--", color="gray", linewidth=0.9, alpha=0.6)
    ax.set_xlabel("Weeks from Dec 1")
    ax.set_ylabel("SWE (mm)")
    ax.set_xticks(series["weeks"])
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper center", ncol=3, frameon=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary(path, station_label, series):
    lines = [
        "# Figure 6-style Weekly Stitching",
        "",
        f"- Year file used: `{YEAR}_weekly_*`",
        f"- Station: `{station_label}`",
        f"- Issue weeks used: `{ISSUE_WEEKS}`",
        "- Each issue week contributes four targets: `week+1, week+2, week+3, week+4`.",
        f"- Final stitched week axis: `{series['weeks'].tolist()}`",
        "",
        "This is the paper-aligned stitched view starting from the Feb 1 / week-8 issue and then continuing with issue weeks 12, 16, and 20.",
    ]
    path.write_text("\n".join(lines))


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    arrays = load_arrays()

    full_station, full_max = choose_station(arrays["actual"])
    full_series = build_fig6_series(arrays, full_station)
    full_path = OUTDIR / "weekly_fig6_style_full512.png"
    plot_series(full_series, f"ForeSWE weekly Figure 6-style stitching\nYear {YEAR}, station {full_station}", full_path)
    write_summary(OUTDIR / "weekly_fig6_style_full512.md", f"full512 station {full_station} (max SWE {full_max:.1f} mm)", full_series)

    sierra_df = pd.read_csv(SIERRA_STATION_SOURCE)
    sierra_df = sierra_df[sierra_df["is_sierra_eval_station"] == True].sort_values("original_station_index")
    sierra_indices = sierra_df["original_station_index"].astype(int).to_numpy()
    sierra_station, sierra_max = choose_station(arrays["actual"], sierra_indices)
    sierra_local = np.where(sierra_indices == sierra_station)[0][0]
    sierra_name = sierra_df.iloc[sierra_local]["Station Name"]
    sierra_series = build_fig6_series(arrays, sierra_station)
    sierra_path = OUTDIR / "weekly_fig6_style_sierra43.png"
    plot_series(sierra_series, f"ForeSWE weekly Figure 6-style stitching (Sierra 43)\nYear {YEAR}, station {sierra_name}", sierra_path)
    write_summary(OUTDIR / "weekly_fig6_style_sierra43.md", f"{sierra_name} / original station {sierra_station} (max SWE {sierra_max:.1f} mm)", sierra_series)

    print("Saved:", full_path)
    print("Saved:", sierra_path)
    print("Full station:", full_station)
    print("Sierra station:", sierra_name, sierra_station)
    print("Weeks:", full_series["weeks"].tolist())


if __name__ == "__main__":
    main()
