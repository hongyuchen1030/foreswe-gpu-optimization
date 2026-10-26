import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PLANB_ROOT = Path(
    os.getenv(
        "FSW_PLANB_ROOT",
        "/global/homes/h/hyvchen/foreswe-gpu-optimization/experiments/planB_24week",
    )
)
PROJECT_ROOT = Path("/global/homes/h/hyvchen/foreswe-gpu-optimization")
SIERRA_STATION_SOURCE = (
    PROJECT_ROOT
    / "experiments"
    / "domain_sensitivity"
    / "D2_sierra_plus_500km"
    / "station_subset.csv"
)
TEST_YEARS = [2015, 2016, 2017, 2018, 2019]
THRESHOLDS_MM = [0.0, 50.0, 100.0]
LEAD_LABELS = [f"Week {i} ({7 * i}d)" for i in range(1, 25)]


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
    return sierra["original_station_index"].astype(int).to_numpy()


def summarize_group(actual, pred, group_label):
    rows = []
    for threshold_mm in THRESHOLDS_MM:
        for horizon_idx in range(actual.shape[2]):
            actual_h = actual[:, :, horizon_idx].reshape(-1)
            pred_h = pred[:, :, horizon_idx].reshape(-1)
            valid = np.isfinite(actual_h) & np.isfinite(pred_h) & (actual_h > threshold_mm)
            x = actual_h[valid]
            y = pred_h[valid]

            if x.size == 0:
                rows.append(
                    {
                        "group": group_label,
                        "threshold_mm": threshold_mm,
                        "horizon_index": horizon_idx + 1,
                        "horizon_label": LEAD_LABELS[horizon_idx],
                        "n_points": 0,
                        "corr": np.nan,
                        "rmse": np.nan,
                        "bias": np.nan,
                        "actual_median": np.nan,
                        "pred_median": np.nan,
                    }
                )
                continue

            corr = float(np.corrcoef(x, y)[0, 1]) if x.size > 1 else np.nan
            rmse = float(np.sqrt(np.mean((y - x) ** 2)))
            bias = float(np.mean(y - x))
            rows.append(
                {
                    "group": group_label,
                    "threshold_mm": threshold_mm,
                    "horizon_index": horizon_idx + 1,
                    "horizon_label": LEAD_LABELS[horizon_idx],
                    "n_points": int(x.size),
                    "corr": corr,
                    "rmse": rmse,
                    "bias": bias,
                    "actual_median": float(np.median(x)),
                    "pred_median": float(np.median(y)),
                }
            )
    return pd.DataFrame(rows)


def main():
    actual, pred = load_arrays()
    sierra_indices = load_sierra_indices()

    full_df = summarize_group(actual, pred, "full512")
    sierra_df = summarize_group(actual[:, sierra_indices, :], pred[:, sierra_indices, :], "sierra43")
    summary = pd.concat([full_df, sierra_df], ignore_index=True)

    out_csv = PLANB_ROOT / "planB_pred_vs_actual_threshold_summary.csv"
    summary.to_csv(out_csv, index=False)

    md_lines = [
        "# Plan B Prediction vs Actual Threshold Summary",
        "",
        "Thresholded evaluation keeps only samples where actual SWE exceeds the chosen threshold.",
        "",
    ]
    for group in ["full512", "sierra43"]:
        md_lines.append(f"## {group}")
        for threshold in THRESHOLDS_MM:
            sub = summary[(summary["group"] == group) & (summary["threshold_mm"] == threshold)]
            week1 = sub[sub["horizon_index"] == 1].iloc[0]
            week12 = sub[sub["horizon_index"] == 12].iloc[0]
            week24 = sub[sub["horizon_index"] == 24].iloc[0]
            md_lines.append(f"- Threshold `{threshold:.0f} mm`:")
            md_lines.append(
                "  Week 1: n={n1}, r={r1:.3f}, RMSE={rmse1:.1f}, Bias={b1:.1f}".format(
                    n1=int(week1["n_points"]),
                    r1=week1["corr"],
                    rmse1=week1["rmse"],
                    b1=week1["bias"],
                )
            )
            md_lines.append(
                "  Week 12: n={n12}, r={r12:.3f}, RMSE={rmse12:.1f}, Bias={b12:.1f}".format(
                    n12=int(week12["n_points"]),
                    r12=week12["corr"],
                    rmse12=week12["rmse"],
                    b12=week12["bias"],
                )
            )
            md_lines.append(
                "  Week 24: n={n24}, r={r24:.3f}, RMSE={rmse24:.1f}, Bias={b24:.1f}".format(
                    n24=int(week24["n_points"]),
                    r24=week24["corr"] if not np.isnan(week24["corr"]) else float("nan"),
                    rmse24=week24["rmse"] if not np.isnan(week24["rmse"]) else float("nan"),
                    b24=week24["bias"] if not np.isnan(week24["bias"]) else float("nan"),
                )
            )
        md_lines.append("")

    out_md = PLANB_ROOT / "planB_pred_vs_actual_threshold_summary.md"
    out_md.write_text("\n".join(md_lines) + "\n")
    print(out_csv)
    print(out_md)


if __name__ == "__main__":
    main()
