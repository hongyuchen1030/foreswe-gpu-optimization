from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/global/homes/h/hyvchen/foreswe-gpu-optimization")
OUTDIR = PROJECT_ROOT / "experiments" / "planA_stitching"
WEEKLY_RESULTS_ROOT = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_weekly_reproduction_full512/results"
)
SIERRA_STATION_SOURCE = (
    PROJECT_ROOT
    / "experiments"
    / "domain_sensitivity"
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


def build_paper_style_series(arrays, station_index):
    # Paper-style display convention:
    # issue week 8 contributes displayed weeks 8-11,
    # issue week 12 contributes displayed weeks 12-15, etc.
    display_weeks = []
    actual = []
    mean = []
    low = []
    high = []
    issue_rows = []

    for issue_week in ISSUE_WEEKS:
        issue_idx = (issue_week - 1) * 7
        if issue_idx >= arrays["actual"].shape[0]:
            continue
        weeks_here = list(range(issue_week, issue_week + FORECASTING_WINDOW))
        display_weeks.extend(weeks_here)
        actual.extend(arrays["actual"][issue_idx, station_index, :].tolist())
        mean.extend(arrays["mean"][issue_idx, station_index, :].tolist())
        low.extend(arrays["low"][issue_idx, station_index, :].tolist())
        high.extend(arrays["high"][issue_idx, station_index, :].tolist())
        issue_rows.append(
            {
                "issue_week": issue_week,
                "issue_row_index": issue_idx,
                "display_weeks": weeks_here,
            }
        )

    return {
        "weeks": np.array(display_weeks),
        "actual": np.array(actual),
        "mean": np.array(mean),
        "low": np.array(low),
        "high": np.array(high),
        "issues": issue_rows,
    }


def plot_series(series, title, out_path):
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    ax.plot(series["weeks"], series["actual"], color="red", linewidth=2.4, label="Actual SWE")
    ax.plot(series["weeks"], series["mean"], color="green", linewidth=2.4, label="Forecasted SWE")
    ax.fill_between(series["weeks"], series["low"], series["high"], color="#4da3d9", alpha=0.4, label="Prediction Interval")
    for boundary in [11.5, 15.5, 19.5]:
        ax.axvline(boundary, linestyle="--", color="gray", linewidth=0.8, alpha=0.45)
    ax.set_xlabel("Weeks")
    ax.set_ylabel("SWE (in mm)")
    ax.set_xlim(7.5, 23.5)
    ax.set_xticks([8, 12, 16, 20])
    ax.grid(alpha=0.15)
    ax.set_title(title)
    ax.legend(loc="upper center", ncol=3, frameon=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary(path, station_label, series):
    lines = [
        "# Figure 6 Weekly Paper-Style Recreation",
        "",
        "This recreation follows the visual convention of the paper's weekly panel:",
        "- x-axis starts at week 8",
        "- issue weeks used: 8, 12, 16, 20",
        "- each issue contributes a 4-point chunk displayed over that same 4-week block",
        "",
        "Important note:",
        "- this is a paper-style display axis recreation",
        "- the upstream model code still defines each issue as predicting four future leads",
        "",
        f"- Year file used: `{YEAR}_weekly_*`",
        f"- Station: `{station_label}`",
        f"- Displayed weeks: `{series['weeks'].tolist()}`",
    ]
    path.write_text("\n".join(lines))


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    arrays = load_arrays()

    full_station, full_max = choose_station(arrays["actual"])
    full_series = build_paper_style_series(arrays, full_station)
    full_path = OUTDIR / "weekly_fig6_paper_style_full512.png"
    plot_series(full_series, f"ForeSWE weekly Figure 6-style panel\nYear {YEAR}, station {full_station}", full_path)
    write_summary(OUTDIR / "weekly_fig6_paper_style_full512.md", f"full512 station {full_station} (max SWE {full_max:.1f} mm)", full_series)

    sierra_df = pd.read_csv(SIERRA_STATION_SOURCE)
    sierra_df = sierra_df[sierra_df["is_sierra_eval_station"] == True].sort_values("original_station_index")
    sierra_indices = sierra_df["original_station_index"].astype(int).to_numpy()
    sierra_station, sierra_max = choose_station(arrays["actual"], sierra_indices)
    sierra_local = np.where(sierra_indices == sierra_station)[0][0]
    sierra_name = sierra_df.iloc[sierra_local]["Station Name"]
    sierra_series = build_paper_style_series(arrays, sierra_station)
    sierra_path = OUTDIR / "weekly_fig6_paper_style_sierra43.png"
    plot_series(sierra_series, f"ForeSWE weekly Figure 6-style panel (Sierra 43)\nYear {YEAR}, station {sierra_name}", sierra_path)
    write_summary(OUTDIR / "weekly_fig6_paper_style_sierra43.md", f"{sierra_name} / original station {sierra_station} (max SWE {sierra_max:.1f} mm)", sierra_series)

    print("Saved:", full_path)
    print("Saved:", sierra_path)
    print("Weeks:", full_series["weeks"].tolist())
    print("Sierra station:", sierra_name, sierra_station)


if __name__ == "__main__":
    main()
