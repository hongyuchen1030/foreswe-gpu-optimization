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
ORIG_REPO_ROOT = Path(
    os.getenv(
        "FSW_ORIG_REPO",
        "/global/homes/h/hyvchen/foreswe-gpu-optimization",
    )
)
DATA_DIR = Path(os.getenv("FSW_DATA_DIR", str(ORIG_REPO_ROOT / "data")))
SIERRA_STATION_SOURCE = (
    PROJECT_ROOT
    / "experiments"
    / "domain_sensitivity"
    / "D2_sierra_plus_500km"
    / "station_subset.csv"
)

PLOT_YEAR = int(os.getenv("PLANB_PLOT_YEAR", "2017"))
STITCHED_COMPARISON_YEAR = PLOT_YEAR - 1
# The Feb 1 issue date predicts t+7..t+168, so the first displayed target week
# should align with week 9 rather than week 8 in the stitched Figure 6B convention.
DISPLAY_WEEKS = np.arange(9, 33)


def load_arrays():
    metadata = pd.read_csv(DATA_DIR / "snotel_location_info_2.csv")
    arrays = {
        "actual": np.asarray(joblib.load(PLANB_ROOT / "planB_feb1_24week_actual.pkl")[PLOT_YEAR]).reshape(-1, len(metadata), 24)[0],
        "mean": np.asarray(joblib.load(PLANB_ROOT / "planB_feb1_24week_mean.pkl")[PLOT_YEAR]).reshape(-1, len(metadata), 24)[0],
        "low": np.asarray(joblib.load(PLANB_ROOT / "planB_feb1_24week_low.pkl")[PLOT_YEAR]).reshape(-1, len(metadata), 24)[0],
        "high": np.asarray(joblib.load(PLANB_ROOT / "planB_feb1_24week_high.pkl")[PLOT_YEAR]).reshape(-1, len(metadata), 24)[0],
    }
    return metadata, arrays


def choose_station(actual_arr, station_indices=None):
    subset = actual_arr if station_indices is None else actual_arr[station_indices]
    station_max = np.nanmax(subset, axis=1)
    local_idx = int(np.nanargmax(station_max))
    if station_indices is None:
        return local_idx, float(station_max[local_idx])
    return int(station_indices[local_idx]), float(station_max[local_idx])


def build_series(arrays, station_index):
    return {
        "weeks": DISPLAY_WEEKS,
        "actual": arrays["actual"][station_index],
        "mean": arrays["mean"][station_index],
        "low": arrays["low"][station_index],
        "high": arrays["high"][station_index],
    }


def plot_series(series, title, out_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(series["weeks"], series["actual"], color="red", linewidth=2.5, label="Actual SWE")
    ax.plot(series["weeks"], series["mean"], color="green", linewidth=2.5, label="Predicted SWE mean")
    ax.fill_between(series["weeks"], series["low"], series["high"], color="#5dade2", alpha=0.35, label="Prediction interval")
    ax.axvline(8, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Weeks from Dec 1 (Figure 6B target-week axis convention)")
    ax.set_ylabel("SWE (mm)")
    ax.set_xticks(np.arange(9, 33, 4))
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    if out_path.suffix == ".png":
        fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def write_summary(path, year, station_index, station_name, station_max, series, label):
    lines = [
        "# Plan B Figure 6B Recreation",
        "",
        f"- Year: `{year}`",
        f"- Matching stitched weekly file year: `{STITCHED_COMPARISON_YEAR}`",
        f"- Station group: `{label}`",
        f"- Station index: `{station_index}`",
        f"- Station name: `{station_name}`",
        f"- Max actual SWE across displayed 24 weeks: `{station_max:.1f} mm`",
        f"- Display weeks: `{series['weeks'].tolist()}`",
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    metadata, arrays = load_arrays()

    full_station, full_max = choose_station(arrays["actual"])
    full_name = metadata.iloc[full_station].get("Station", f"station_{full_station}")
    full_series = build_series(arrays, full_station)
    full_png = PLANB_ROOT / "planB_fig6b_full512.png"
    plot_series(full_series, f"Plan B 24-week Figure 6B\nYear {PLOT_YEAR}, station {full_station} ({full_name})", full_png)
    write_summary(
        PLANB_ROOT / "planB_fig6b_full512.md",
        PLOT_YEAR,
        full_station,
        full_name,
        full_max,
        full_series,
        "full512",
    )

    sierra_df = pd.read_csv(SIERRA_STATION_SOURCE)
    sierra_df = sierra_df[sierra_df["is_sierra_eval_station"] == True].sort_values("original_station_index")
    sierra_indices = sierra_df["original_station_index"].astype(int).to_numpy()
    sierra_station, sierra_max = choose_station(arrays["actual"], sierra_indices)
    sierra_name = metadata.iloc[sierra_station].get("Station", f"station_{sierra_station}")
    sierra_series = build_series(arrays, sierra_station)
    sierra_png = PLANB_ROOT / "planB_fig6b_sierra43.png"
    plot_series(
        sierra_series,
        f"Plan B 24-week Figure 6B (Sierra 43)\nYear {PLOT_YEAR}, station {sierra_station} ({sierra_name})",
        sierra_png,
    )
    write_summary(
        PLANB_ROOT / "planB_fig6b_sierra43.md",
        PLOT_YEAR,
        sierra_station,
        sierra_name,
        sierra_max,
        sierra_series,
        "sierra43",
    )

    selection_lines = [
        f"selected_year={PLOT_YEAR}",
        f"stitched_comparison_year={STITCHED_COMPARISON_YEAR}",
        f"full_station={full_station}",
        f"full_station_name={full_name}",
        f"sierra43_station={sierra_station}",
        f"sierra43_station_name={sierra_name}",
        f"display_weeks={int(DISPLAY_WEEKS[0])}..{int(DISPLAY_WEEKS[-1])}",
    ]
    (PLANB_ROOT / "planB_fig6b_selection.txt").write_text("\n".join(selection_lines) + "\n")

    print("Saved", full_png)
    print("Saved", sierra_png)
    print("Full station", full_station, full_name, full_max)
    print("Sierra station", sierra_station, sierra_name, sierra_max)


if __name__ == "__main__":
    main()
