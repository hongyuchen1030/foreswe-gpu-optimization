import os
from pathlib import Path

import joblib
import numpy as np


PLANB_ROOT = Path(
    os.getenv(
        "FSW_PLANB_ROOT",
        "/global/homes/h/hyvchen/foreswe-gpu-optimization/experiments/planB_24week",
    )
)
WEEKLY_RESULTS_ROOT = Path(
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_weekly_reproduction_full512/results"
)

PLANB_YEAR = int(os.getenv("PLANB_PLOT_YEAR", "2017"))
STITCHED_YEAR = PLANB_YEAR - 1
STATION = int(os.getenv("PLANB_VERIFY_STATION", "239"))
STITCHED_ROW = int(os.getenv("PLANB_VERIFY_ROW", "61"))


def main():
    planb_actual = np.asarray(joblib.load(PLANB_ROOT / "planB_feb1_24week_actual.pkl")[PLANB_YEAR]).reshape(512, 24)[STATION]
    planb_mean = np.asarray(joblib.load(PLANB_ROOT / "planB_feb1_24week_mean.pkl")[PLANB_YEAR]).reshape(512, 24)[STATION]
    stitched_actual = np.asarray(joblib.load(WEEKLY_RESULTS_ROOT / "actual" / f"{STITCHED_YEAR}_weekly_window_actual.pkl"))
    stitched_mean = np.asarray(joblib.load(WEEKLY_RESULTS_ROOT / "predict" / f"{STITCHED_YEAR}_weekly_gp_window_predict.pkl"))

    actual_delta = planb_actual[:4] - stitched_actual[STITCHED_ROW, STATION, :]
    mean_delta = planb_mean[:4] - stitched_mean[STITCHED_ROW, STATION, :]

    lines = [
        "# Plan B vs Stitched Figure 6B Check",
        "",
        f"- Plan B year: `{PLANB_YEAR}`",
        f"- Matching stitched file year: `{STITCHED_YEAR}`",
        f"- Station index: `{STATION}`",
        f"- Stitched row: `{STITCHED_ROW}`",
        "",
        f"- Plan B actual first 4: `{planb_actual[:4].tolist()}`",
        f"- Stitched actual first 4: `{stitched_actual[STITCHED_ROW, STATION, :].tolist()}`",
        f"- Actual delta: `{actual_delta.tolist()}`",
        "",
        f"- Plan B mean first 4: `{planb_mean[:4].tolist()}`",
        f"- Stitched mean first 4: `{stitched_mean[STITCHED_ROW, STATION, :].tolist()}`",
        f"- Mean delta: `{mean_delta.tolist()}`",
    ]

    out_path = PLANB_ROOT / "planB_vs_stitched_check.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(out_path)


if __name__ == "__main__":
    main()
