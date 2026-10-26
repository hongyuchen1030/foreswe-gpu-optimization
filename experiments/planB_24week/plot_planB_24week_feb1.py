import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLANB_ROOT = os.getenv(
    "FSW_PLANB_ROOT",
    "/global/homes/h/hyvchen/foreswe-gpu-optimization/experiments/planB_24week",
)
ORIG_REPO_ROOT = os.getenv(
    "FSW_ORIG_REPO",
    "/global/homes/h/hyvchen/foreswe-gpu-optimization",
)
DATA_DIR = os.getenv("FSW_DATA_DIR", os.path.join(ORIG_REPO_ROOT, "data"))

mean_by_year = joblib.load(os.path.join(PLANB_ROOT, "planB_feb1_24week_mean.pkl"))
low_by_year = joblib.load(os.path.join(PLANB_ROOT, "planB_feb1_24week_low.pkl"))
high_by_year = joblib.load(os.path.join(PLANB_ROOT, "planB_feb1_24week_high.pkl"))
actual_by_year = joblib.load(os.path.join(PLANB_ROOT, "planB_feb1_24week_actual.pkl"))

metadata = pd.read_csv(os.path.join(DATA_DIR, "snotel_location_info_2.csv"))

selected_year = int(os.getenv("PLANB_PLOT_YEAR", "2017"))
selected_station = int(os.getenv("PLANB_PLOT_STATION", "329"))
actual_arr = np.array(actual_by_year[selected_year]).reshape(-1, len(metadata), 24)[0]
station_name = metadata.iloc[selected_station].get("Station", "station_{}".format(selected_station))

mean_arr = np.array(mean_by_year[selected_year]).reshape(-1, len(metadata), 24)[0]
low_arr = np.array(low_by_year[selected_year]).reshape(-1, len(metadata), 24)[0]
high_arr = np.array(high_by_year[selected_year]).reshape(-1, len(metadata), 24)[0]

# Match the Figure 6B display convention where the Feb 1 issue date is shown at week 8.
# The 24 forecasted weekly points are laid out over weeks 8..31 for direct comparison
# to the stitched 4-week-window panel.
display_weeks = np.arange(8, 32)
actual = actual_arr[selected_station]
pred = mean_arr[selected_station]
low = low_arr[selected_station]
high = high_arr[selected_station]

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(display_weeks, actual, color="red", linewidth=2.5, label="Actual SWE")
ax.plot(display_weeks, pred, color="green", linewidth=2.5, label="Predicted SWE mean")
ax.fill_between(display_weeks, low, high, color="#5dade2", alpha=0.35, label="Prediction interval")
ax.axvline(8, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Weeks from Dec 1 (Figure 6B axis convention)")
ax.set_ylabel("SWE (mm)")
ax.set_title("Plan B 24-week Feb 1 forecast\nYear {}, station {} ({})".format(selected_year, selected_station, station_name))
ax.set_xticks(np.arange(8, 32, 4))
ax.grid(alpha=0.25)
ax.legend(frameon=True)
fig.tight_layout()
fig.savefig(os.path.join(PLANB_ROOT, "planB_feb1_24week_plot.png"), dpi=220)
fig.savefig(os.path.join(PLANB_ROOT, "planB_feb1_24week_plot.pdf"))

with open(os.path.join(PLANB_ROOT, "planB_plot_selection.txt"), "w") as f:
    f.write("selected_year={}\n".format(selected_year))
    f.write("selected_station={}\n".format(selected_station))
    f.write("selected_station_name={}\n".format(station_name))
    f.write("display_weeks={}..{}\n".format(int(display_weeks[0]), int(display_weeks[-1])))
