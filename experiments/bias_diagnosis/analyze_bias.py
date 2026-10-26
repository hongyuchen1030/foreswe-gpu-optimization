import math
import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path("/global/homes/h/hyvchen/foreswe-gpu-optimization")
THIRDPARTY_ROOT = REPO_ROOT
FORE_SWE_DIR = THIRDPARTY_ROOT / "fore_swe"
WEEKLY_ROOT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_weekly_reproduction_full512")
PLANB_ROOT = REPO_ROOT / "experiments" / "planB_24week"
OUT_DIR = REPO_ROOT / "experiments" / "bias_diagnosis"

sys.path.insert(0, str(FORE_SWE_DIR))
os.environ["BASE_URL"] = str(THIRDPARTY_ROOT)

import extended_dl_forecasting as DL  # noqa: E402


FULL512_TEST_INDICES = sorted([
    43, 66, 338, 463, 115, 105, 83, 306, 481, 21, 185, 136, 110, 406, 408, 132,
    120, 197, 7, 301, 375, 236, 269, 287, 49, 241, 212, 145, 264, 435, 73, 277, 364,
    438, 372, 368, 243, 493, 497, 14, 32, 433, 313, 213, 42, 475, 160, 190, 281, 285,
    316, 461, 154, 424, 310, 340, 179, 167, 193, 466, 56, 158, 365, 492, 432, 415, 68,
    409, 76, 148, 267, 227, 91, 189, 472, 162, 147, 395, 113, 2, 271, 357, 347, 335, 507,
    465, 57, 19, 442, 381, 398, 79, 337, 501, 41, 339, 331, 63, 359, 232, 323, 258,
])
TEST_YEAR_LABELS = [2015, 2016, 2017, 2018, 2019]
SWE_BIN_LABELS = ["SWE=0", "0-100", "100-300", "300-600", "600-1000", ">1000"]
SCATTER_BINS = [
    ("0-100", 0.0, 100.0),
    ("100-300", 100.0, 300.0),
    ("300-600", 300.0, 600.0),
    ("600-1000", 600.0, 1000.0),
    (">1000", 1000.0, np.inf),
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_torch_list(path: Path) -> np.ndarray:
    items = torch.load(path, map_location="cpu")
    arrays = []
    for item in items:
        if isinstance(item, torch.Tensor):
            arrays.append(item.detach().cpu().numpy())
        else:
            arrays.append(np.asarray(item))
    return np.concatenate(arrays, axis=0).astype(np.float32, copy=False)


def load_year_array_dict(path: Path) -> dict:
    raw = joblib.load(path)
    return {int(k): np.asarray(v, dtype=np.float32) for k, v in raw.items()}


def concat_year_pkls(root: Path, suffix: str) -> np.ndarray:
    arrays = []
    for year in range(2014, 2019):
        arrays.append(np.asarray(joblib.load(root / f"{year}_{suffix}.pkl"), dtype=np.float32))
    return np.concatenate(arrays, axis=0)


def concat_year_dict(data: dict) -> np.ndarray:
    return np.concatenate([data[year] for year in sorted(data)], axis=0).astype(np.float32, copy=False)


def load_original_weekly_gp_aligned():
    year_rows = []
    actual_parts = []
    mean_parts = []
    low_parts = []
    high_parts = []
    for year in range(2014, 2019):
        actual = np.asarray(joblib.load(WEEKLY_ROOT / "results" / "actual" / f"{year}_weekly_window_actual.pkl"), dtype=np.float32)
        mean = np.asarray(joblib.load(WEEKLY_ROOT / "results" / "predict" / f"{year}_weekly_gp_window_predict.pkl"), dtype=np.float32)
        low = np.asarray(joblib.load(WEEKLY_ROOT / "results" / "predict" / f"{year}_weekly_gp_low_window_predict.pkl"), dtype=np.float32)
        high = np.asarray(joblib.load(WEEKLY_ROOT / "results" / "predict" / f"{year}_weekly_gp_high_window_predict.pkl"), dtype=np.float32)
        common = min(actual.shape[0], mean.shape[0], low.shape[0], high.shape[0])
        year_rows.append({
            "year": year,
            "actual_rows": int(actual.shape[0]),
            "gp_mean_rows": int(mean.shape[0]),
            "gp_low_rows": int(low.shape[0]),
            "gp_high_rows": int(high.shape[0]),
            "used_rows": int(common),
        })
        actual_parts.append(actual[:common])
        mean_parts.append(mean[:common])
        low_parts.append(low[:common])
        high_parts.append(high[:common])
    return (
        pd.DataFrame(year_rows),
        np.concatenate(actual_parts, axis=0),
        np.concatenate(mean_parts, axis=0),
        np.concatenate(low_parts, axis=0),
        np.concatenate(high_parts, axis=0),
    )


def build_calendar():
    dataloader = DL.DataLoader(test_indices=FULL512_TEST_INDICES, temp_window=7)
    dataloader.cleanDataJunk()
    dataloader.normalizeData()
    dataloader.addDayInfo()
    dataloader.prepareInputsAndOutputs()
    dataloader.completeInputsAndOutputs()

    all_dates = np.array(pd.to_datetime(dataloader.all_dates))
    orig_train_issue_indices = np.array(dataloader.getTrainingDataIndices())[: -(4 * 7)]
    orig_test_issue_indices = [np.array(x) for x in dataloader.getTestingDataIndices()]
    planb_train_issue_indices = np.array(dataloader.getTrainingDataIndices())[: -(24 * 7)]
    planb_test_issue_indices = {}
    for year, indices in zip(TEST_YEAR_LABELS, dataloader.getTestingDataIndices()):
        matched = [int(idx) for idx in indices if str(all_dates[idx])[:10] == f"{year}-02-01"]
        if len(matched) != 1:
            raise RuntimeError(f"Expected 1 Feb 1 issue date for {year}, found {matched}")
        planb_test_issue_indices[year] = np.array(matched, dtype=int)
    return all_dates, orig_train_issue_indices, orig_test_issue_indices, planb_train_issue_indices, planb_test_issue_indices


def build_target_dates(all_dates, issue_indices: np.ndarray, n_leads: int, step_days: int) -> np.ndarray:
    issue_indices = np.asarray(issue_indices, dtype=int).reshape(-1)
    lead_offsets = np.arange(1, n_leads + 1, dtype=int) * step_days
    target_indices = issue_indices[:, None] + lead_offsets[None, :]
    return all_dates[target_indices]


def sample_scatter(actual: np.ndarray, pred: np.ndarray, max_points: int = 200000):
    actual = actual.reshape(-1)
    pred = pred.reshape(-1)
    mask = np.isfinite(actual) & np.isfinite(pred)
    actual = actual[mask]
    pred = pred[mask]
    if actual.size > max_points:
        rng = np.random.default_rng(42)
        chosen = rng.choice(actual.size, size=max_points, replace=False)
        actual = actual[chosen]
        pred = pred[chosen]
    return actual, pred


def regression_stats(actual: np.ndarray, pred: np.ndarray):
    actual = actual.reshape(-1)
    pred = pred.reshape(-1)
    mask = np.isfinite(actual) & np.isfinite(pred)
    actual = actual[mask]
    pred = pred[mask]
    if actual.size < 2:
        return np.nan, np.nan, np.nan, np.nan
    x_mean = float(actual.mean())
    y_mean = float(pred.mean())
    x_var = float(np.mean((actual - x_mean) ** 2))
    if x_var == 0:
        slope = np.nan
        intercept = np.nan
        r2 = np.nan
    else:
        cov = float(np.mean((actual - x_mean) * (pred - y_mean)))
        slope = cov / x_var
        intercept = y_mean - slope * x_mean
        ss_res = float(np.sum((pred - (slope * actual + intercept)) ** 2))
        ss_tot = float(np.sum((pred - y_mean) ** 2))
        r2 = np.nan if ss_tot == 0 else 1.0 - (ss_res / ss_tot)
    corr = np.nan
    if actual.size >= 2 and np.std(actual) > 0 and np.std(pred) > 0:
        corr = float(np.corrcoef(actual, pred)[0, 1])
    return slope, intercept, r2, corr


def compute_bias_metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    actual = actual.reshape(-1)
    pred = pred.reshape(-1)
    mask = np.isfinite(actual) & np.isfinite(pred)
    actual = actual[mask]
    pred = pred[mask]
    if actual.size == 0:
        return {
            "count": 0,
            "mean_actual": np.nan,
            "mean_pred": np.nan,
            "mean_error": np.nan,
            "median_error": np.nan,
            "relative_bias": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
            "r2": np.nan,
            "corr": np.nan,
        }
    err = pred - actual
    slope, intercept, r2, corr = regression_stats(actual, pred)
    mean_actual = float(actual.mean())
    mean_pred = float(pred.mean())
    mean_error = float(err.mean())
    return {
        "count": int(actual.size),
        "mean_actual": mean_actual,
        "mean_pred": mean_pred,
        "mean_error": mean_error,
        "median_error": float(np.median(err)),
        "relative_bias": np.nan if mean_actual == 0 else float(mean_error / mean_actual),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "corr": corr,
    }


def swe_bin_masks(actual: np.ndarray):
    return [
        actual == 0,
        (actual > 0) & (actual < 100),
        (actual >= 100) & (actual < 300),
        (actual >= 300) & (actual < 600),
        (actual >= 600) & (actual < 1000),
        actual >= 1000,
    ]


def summarize_by_swe_bin(dataset_name: str, stage: str, source: str, actual: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    rows = []
    masks = swe_bin_masks(actual)
    for label, mask in zip(SWE_BIN_LABELS, masks):
        metrics = compute_bias_metrics(actual[mask], pred[mask])
        rows.append({"dataset": dataset_name, "stage": stage, "source": source, "swe_bin": label, **metrics})
    return pd.DataFrame(rows)


def summarize_by_lead(dataset_name: str, stage: str, source: str, actual: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for lead in range(actual.shape[-1]):
        metrics = compute_bias_metrics(actual[:, :, lead], pred[:, :, lead])
        rows.append({"dataset": dataset_name, "stage": stage, "source": source, "lead": lead + 1, **metrics})
    return pd.DataFrame(rows)


def summarize_distribution_bins(dataset_name: str, split_name: str, actual: np.ndarray) -> pd.DataFrame:
    actual = actual.reshape(-1)
    total = actual.size
    rows = []
    for label, mask in zip(SWE_BIN_LABELS, swe_bin_masks(actual)):
        count = int(np.count_nonzero(mask))
        rows.append({
            "dataset": dataset_name,
            "split": split_name,
            "swe_bin": label,
            "count": count,
            "fraction": count / total if total else np.nan,
        })
    return pd.DataFrame(rows)


def summarize_distribution_by_lead(dataset_name: str, split_name: str, actual: np.ndarray) -> pd.DataFrame:
    rows = []
    for lead in range(actual.shape[-1]):
        lead_actual = actual[:, :, lead].reshape(-1)
        for label, mask in zip(SWE_BIN_LABELS, swe_bin_masks(lead_actual)):
            count = int(np.count_nonzero(mask))
            rows.append({
                "dataset": dataset_name,
                "split": split_name,
                "lead": lead + 1,
                "swe_bin": label,
                "count": count,
                "fraction": count / lead_actual.size if lead_actual.size else np.nan,
            })
    return pd.DataFrame(rows)


def summarize_distribution_by_time(dataset_name: str, split_name: str, actual: np.ndarray, target_dates: np.ndarray, key_name: str, extractor) -> pd.DataFrame:
    rows = []
    values = extractor(target_dates)
    for lead in range(actual.shape[-1]):
        lead_actual = actual[:, :, lead]
        lead_values = values[:, lead]
        unique_values = sorted(pd.unique(lead_values))
        for key in unique_values:
            mask_issue = lead_values == key
            masked_actual = lead_actual[mask_issue, :].reshape(-1)
            total = masked_actual.size
            if total == 0:
                continue
            for label, bin_mask in zip(SWE_BIN_LABELS, swe_bin_masks(masked_actual)):
                count = int(np.count_nonzero(bin_mask))
                rows.append({
                    "dataset": dataset_name,
                    "split": split_name,
                    "lead": lead + 1,
                    key_name: key,
                    "swe_bin": label,
                    "count": count,
                    "fraction": count / total if total else np.nan,
                })
    return pd.DataFrame(rows)


def station_peak_groups(actual: np.ndarray, n_groups: int = 4):
    station_peaks = actual.max(axis=(0, 2))
    quantiles = np.quantile(station_peaks, np.linspace(0, 1, n_groups + 1))
    groups = np.zeros_like(station_peaks, dtype=int)
    for idx in range(n_groups):
        lo = quantiles[idx]
        hi = quantiles[idx + 1]
        if idx == n_groups - 1:
            mask = (station_peaks >= lo) & (station_peaks <= hi)
        else:
            mask = (station_peaks >= lo) & (station_peaks < hi)
        groups[mask] = idx + 1
    labels = {
        idx + 1: f"Q{idx + 1} peak={quantiles[idx]:.1f}-{quantiles[idx + 1]:.1f}"
        for idx in range(n_groups)
    }
    return groups, station_peaks, labels


def summarize_station_group(dataset_name: str, stage: str, source: str, actual: np.ndarray, pred: np.ndarray) -> tuple:
    groups, station_peaks, labels = station_peak_groups(actual)
    lead_rows = []
    bin_rows = []
    for group_id, label in labels.items():
        station_mask = groups == group_id
        group_actual = actual[:, station_mask, :]
        group_pred = pred[:, station_mask, :]
        for lead in range(actual.shape[-1]):
            metrics = compute_bias_metrics(group_actual[:, :, lead], group_pred[:, :, lead])
            lead_rows.append({
                "dataset": dataset_name,
                "stage": stage,
                "source": source,
                "station_group": label,
                "group_id": group_id,
                "lead": lead + 1,
                **metrics,
            })
        masks = swe_bin_masks(group_actual)
        for swe_bin, mask in zip(SWE_BIN_LABELS, masks):
            metrics = compute_bias_metrics(group_actual[mask], group_pred[mask])
            bin_rows.append({
                "dataset": dataset_name,
                "stage": stage,
                "source": source,
                "station_group": label,
                "group_id": group_id,
                "swe_bin": swe_bin,
                **metrics,
            })
    peak_df = pd.DataFrame({
        "station_index": np.arange(station_peaks.size),
        "peak_actual_swe": station_peaks,
        "group_id": groups,
        "station_group": [labels[g] for g in groups],
    })
    return pd.DataFrame(lead_rows), pd.DataFrame(bin_rows), peak_df


def plot_hexbin_panel(ax, actual: np.ndarray, pred: np.ndarray, title: str, xlim: tuple):
    x, y = sample_scatter(actual, pred)
    hb = ax.hexbin(x, y, gridsize=70, mincnt=1, cmap="Blues")
    ax.plot(xlim, xlim, "--", color="black", linewidth=1)
    ax.set_xlim(*xlim)
    ax.set_ylim(*xlim)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Actual SWE (mm)")
    ax.set_ylabel("Predicted SWE (mm)")
    metrics = compute_bias_metrics(x, y)
    ax.text(
        0.03,
        0.97,
        f"n={metrics['count']:,}\nBias={metrics['mean_error']:.1f}\nRMSE={metrics['rmse']:.1f}\nr={metrics['corr']:.2f}\nslope={metrics['slope']:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    return hb


def save_pred_vs_actual_figure(path: Path, title: str, actual_att: np.ndarray, pred_att: np.ndarray, actual_gp: np.ndarray, pred_gp: np.ndarray, low_gp: np.ndarray, high_gp: np.ndarray):
    x_max = float(np.quantile(np.concatenate([actual_att.reshape(-1), actual_gp.reshape(-1)]), 0.995))
    x_max = max(x_max, 1000.0)
    xlim = (0.0, math.ceil(x_max / 100.0) * 100.0)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    plot_hexbin_panel(axes[0, 0], actual_att, pred_att, "Attention output vs actual", xlim)
    plot_hexbin_panel(axes[0, 1], actual_gp, pred_gp, "GP mean vs actual", xlim)
    plot_hexbin_panel(axes[1, 0], actual_gp, low_gp, "GP lower bound vs actual", xlim)
    plot_hexbin_panel(axes[1, 1], actual_gp, high_gp, "GP upper bound vs actual", xlim)
    fig.suptitle(title, fontsize=14)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_bias_by_bin(path: Path, df: pd.DataFrame):
    plot_df = df[df["swe_bin"] != "SWE=0"].copy()
    datasets = [("original_weekly", "Original Weekly"), ("planB_24week", "Plan B 24-week")]
    stages = [("attention", "Attention"), ("gp_mean", "GP mean")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True, sharey=True)
    for row, (dataset_key, dataset_title) in enumerate(datasets):
        for col, (stage_key, stage_title) in enumerate(stages):
            ax = axes[row, col]
            subset = plot_df[(plot_df["dataset"] == dataset_key) & (plot_df["stage"] == stage_key)]
            ax.axhline(0.0, color="black", linewidth=1)
            ax.bar(subset["swe_bin"], subset["mean_error"], color="#c65d28")
            ax.set_title(f"{dataset_title}: {stage_title}")
            ax.set_ylabel("Mean error (pred - actual, mm)")
            ax.tick_params(axis="x", rotation=25)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_bias_by_lead(path: Path, lead_df: pd.DataFrame):
    datasets = [("original_weekly", "Original Weekly"), ("planB_24week", "Plan B 24-week")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for row, (dataset_key, dataset_title) in enumerate(datasets):
        subset = lead_df[lead_df["dataset"] == dataset_key]
        for stage_key, color, label in [("attention", "#1b7f3b", "Attention"), ("gp_mean", "#225ea8", "GP mean")]:
            stage_subset = subset[subset["stage"] == stage_key]
            axes[row, 0].plot(stage_subset["lead"], stage_subset["mean_error"], label=label, color=color)
            axes[row, 1].plot(stage_subset["lead"], stage_subset["slope"], label=label, color=color)
        axes[row, 0].axhline(0.0, color="black", linewidth=1)
        axes[row, 1].axhline(1.0, color="black", linewidth=1)
        axes[row, 0].set_title(f"{dataset_title}: Mean error by lead")
        axes[row, 1].set_title(f"{dataset_title}: Regression slope by lead")
        axes[row, 0].set_ylabel("Mean error (mm)")
        axes[row, 1].set_ylabel("Slope")
        axes[row, 0].set_xlabel("Lead week")
        axes[row, 1].set_xlabel("Lead week")
        axes[row, 0].legend()
        axes[row, 1].legend()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def heatmap_counts(actual: np.ndarray, y_bins: np.ndarray) -> np.ndarray:
    counts = []
    for lead in range(actual.shape[-1]):
        hist, _ = np.histogram(actual[:, :, lead].reshape(-1), bins=y_bins)
        counts.append(hist)
    return np.array(counts).T


def save_target_distribution(path: Path, original_train: np.ndarray, original_test: np.ndarray, planb_train: np.ndarray, planb_test: np.ndarray):
    y_bins = np.array([0, 1, 50, 100, 300, 600, 1000, 1500, 2000, 3000, 4000], dtype=float)
    configs = [
        ("Original weekly train", original_train),
        ("Original weekly test", original_test),
        ("Plan B train", planb_train),
        ("Plan B test", planb_test),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for ax, (title, actual) in zip(axes.flatten(), configs):
        counts = heatmap_counts(actual, y_bins)
        im = ax.imshow(counts, origin="lower", aspect="auto", cmap="YlGnBu")
        ax.set_title(title)
        ax.set_xlabel("Lead week")
        ax.set_ylabel("SWE bin")
        ax.set_xticks(np.arange(actual.shape[-1]))
        ax.set_xticklabels(np.arange(1, actual.shape[-1] + 1))
        ax.set_yticks(np.arange(len(y_bins) - 1))
        ax.set_yticklabels([f"{int(y_bins[i])}-{int(y_bins[i+1])}" for i in range(len(y_bins) - 1)])
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_attention_vs_gp_comparison(path: Path, bin_df: pd.DataFrame, lead_df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    datasets = [("original_weekly", "Original Weekly"), ("planB_24week", "Plan B 24-week")]
    for row, (dataset_key, dataset_title) in enumerate(datasets):
        bin_subset = bin_df[(bin_df["dataset"] == dataset_key) & (bin_df["swe_bin"] != "SWE=0")]
        for stage_key, color, label in [("attention", "#1b7f3b", "Attention"), ("gp_mean", "#225ea8", "GP mean")]:
            stage_bin = bin_subset[bin_subset["stage"] == stage_key]
            stage_lead = lead_df[(lead_df["dataset"] == dataset_key) & (lead_df["stage"] == stage_key)]
            axes[row, 0].plot(stage_bin["swe_bin"], stage_bin["mean_error"], marker="o", color=color, label=label)
            axes[row, 1].plot(stage_lead["lead"], stage_lead["relative_bias"], color=color, label=label)
        axes[row, 0].axhline(0.0, color="black", linewidth=1)
        axes[row, 1].axhline(0.0, color="black", linewidth=1)
        axes[row, 0].set_title(f"{dataset_title}: Mean error by actual SWE bin")
        axes[row, 1].set_title(f"{dataset_title}: Relative bias by lead")
        axes[row, 0].set_ylabel("Mean error (mm)")
        axes[row, 1].set_ylabel("Relative bias")
        axes[row, 0].tick_params(axis="x", rotation=25)
        axes[row, 1].set_xlabel("Lead week")
        axes[row, 0].legend()
        axes[row, 1].legend()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_three_bin_scatter(path: Path, actual: np.ndarray, pred: np.ndarray, title: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    xlim = (0.0, max(1000.0, float(np.quantile(actual.reshape(-1), 0.995))))
    xlim = (0.0, math.ceil(xlim[1] / 100.0) * 100.0)
    ranges = [("0-50", 0.0, 50.0), ("50-100", 50.0, 100.0), (">100", 100.0, np.inf)]
    for ax, (label, lo, hi) in zip(axes, ranges):
        mask = (actual > lo) & (actual <= hi)
        x, y = sample_scatter(actual[mask], pred[mask], max_points=120000)
        ax.hexbin(x, y, gridsize=60, mincnt=1, cmap="Blues")
        ax.plot(xlim, xlim, "--", color="black", linewidth=1)
        ax.set_xlim(*xlim)
        ax.set_ylim(*xlim)
        ax.set_title(f"{label} mm")
        ax.set_xlabel("Actual SWE (mm)")
        ax.set_ylabel("Predicted SWE (mm)")
        metrics = compute_bias_metrics(x, y)
        ax.text(0.03, 0.97, f"n={metrics['count']:,}\nBias={metrics['mean_error']:.1f}\nRMSE={metrics['rmse']:.1f}\nr={metrics['corr']:.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    fig.suptitle(title, fontsize=14)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def table_to_markdown(df: pd.DataFrame, columns: list, round_digits: int = 3, head: int = None) -> str:
    subset = df.loc[:, columns].copy()
    subset = subset.round(round_digits)
    if head is not None:
        subset = subset.head(head)
    values = [columns]
    for row in subset.itertuples(index=False, name=None):
        values.append(["" if pd.isna(v) else str(v) for v in row])
    widths = [max(len(str(values[r][c])) for r in range(len(values))) for c in range(len(columns))]
    header = "| " + " | ".join(str(columns[i]).ljust(widths[i]) for i in range(len(columns))) + " |"
    divider = "| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |"
    rows = [
        "| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(columns))) + " |"
        for row in values[1:]
    ]
    return "\n".join([header, divider] + rows)


def main():
    ensure_dir(OUT_DIR)
    all_dates, orig_train_issue_indices, orig_test_issue_indices, planb_train_issue_indices, planb_test_issue_indices = build_calendar()

    original_attention_train_actual = load_torch_list(WEEKLY_ROOT / "outputs" / "forecasting_modified_weekly_att_outputs_for_gp_act.pt")
    original_attention_train_pred = load_torch_list(WEEKLY_ROOT / "outputs" / "forecasting_modified_weekly_att_predicted_outputs_for_gp_act.pt")
    original_gp_year_alignment, original_gp_test_actual, original_gp_test_mean, original_gp_test_low, original_gp_test_high = load_original_weekly_gp_aligned()
    original_gp_year_alignment.to_csv(OUT_DIR / "original_weekly_gp_year_alignment.csv", index=False)

    planb_attention_train_actual = load_torch_list(PLANB_ROOT / "outputs" / "forecasting_modified_weekly_att_outputs_for_gp_act.pt")
    planb_attention_train_pred = load_torch_list(PLANB_ROOT / "outputs" / "forecasting_modified_weekly_att_predicted_outputs_for_gp_act.pt")
    planb_attention_test_actual_by_year = load_year_array_dict(PLANB_ROOT / "planB_feb1_attention_actual_by_year.pkl")
    planb_attention_test_pred_by_year = load_year_array_dict(PLANB_ROOT / "planB_feb1_attention_predicted_by_year.pkl")
    planb_gp_test_actual_by_year = load_year_array_dict(PLANB_ROOT / "planB_feb1_24week_actual.pkl")
    planb_gp_test_mean_by_year = load_year_array_dict(PLANB_ROOT / "planB_feb1_24week_mean.pkl")
    planb_gp_test_low_by_year = load_year_array_dict(PLANB_ROOT / "planB_feb1_24week_low.pkl")
    planb_gp_test_high_by_year = load_year_array_dict(PLANB_ROOT / "planB_feb1_24week_high.pkl")

    planb_attention_test_actual = concat_year_dict(planb_attention_test_actual_by_year)
    planb_attention_test_pred = concat_year_dict(planb_attention_test_pred_by_year)
    planb_gp_test_actual = concat_year_dict(planb_gp_test_actual_by_year)
    planb_gp_test_mean = concat_year_dict(planb_gp_test_mean_by_year)
    planb_gp_test_low = concat_year_dict(planb_gp_test_low_by_year)
    planb_gp_test_high = concat_year_dict(planb_gp_test_high_by_year)

    orig_train_target_dates = build_target_dates(all_dates, orig_train_issue_indices, n_leads=4, step_days=7)
    orig_test_issue_flat = np.concatenate(orig_test_issue_indices)
    aligned_test_issue_parts = []
    cursor = 0
    for _, row in original_gp_year_alignment.iterrows():
        issue_part = orig_test_issue_flat[cursor: cursor + int(row["actual_rows"])]
        aligned_test_issue_parts.append(issue_part[: int(row["used_rows"])])
        cursor += int(row["actual_rows"])
    orig_test_target_dates = build_target_dates(all_dates, np.concatenate(aligned_test_issue_parts), n_leads=4, step_days=7)
    planb_train_target_dates = build_target_dates(all_dates, planb_train_issue_indices, n_leads=24, step_days=7)
    planb_test_issue_flat = np.concatenate([planb_test_issue_indices[y] for y in TEST_YEAR_LABELS])
    planb_test_target_dates = build_target_dates(all_dates, planb_test_issue_flat, n_leads=24, step_days=7)

    shapes_rows = [
        {"name": "original_attention_train_actual", "shape": tuple(original_attention_train_actual.shape)},
        {"name": "original_attention_train_pred", "shape": tuple(original_attention_train_pred.shape)},
        {"name": "original_gp_test_actual", "shape": tuple(original_gp_test_actual.shape)},
        {"name": "original_gp_test_mean", "shape": tuple(original_gp_test_mean.shape)},
        {"name": "planb_attention_train_actual", "shape": tuple(planb_attention_train_actual.shape)},
        {"name": "planb_attention_train_pred", "shape": tuple(planb_attention_train_pred.shape)},
        {"name": "planb_attention_test_actual", "shape": tuple(planb_attention_test_actual.shape)},
        {"name": "planb_attention_test_pred", "shape": tuple(planb_attention_test_pred.shape)},
        {"name": "planb_gp_test_actual", "shape": tuple(planb_gp_test_actual.shape)},
        {"name": "planb_gp_test_mean", "shape": tuple(planb_gp_test_mean.shape)},
    ]
    pd.DataFrame(shapes_rows).to_csv(OUT_DIR / "loaded_tensor_shapes.csv", index=False)

    distribution_frames = [
        summarize_distribution_bins("original_weekly", "train_attention_saved", original_attention_train_actual),
        summarize_distribution_bins("original_weekly", "test_gp_saved", original_gp_test_actual),
        summarize_distribution_bins("planB_24week", "train_attention_saved", planb_attention_train_actual),
        summarize_distribution_bins("planB_24week", "test_gp_saved", planb_gp_test_actual),
    ]
    distribution_bin_df = pd.concat(distribution_frames, ignore_index=True)
    distribution_bin_df.to_csv(OUT_DIR / "target_distribution_bins.csv", index=False)

    distribution_lead_df = pd.concat([
        summarize_distribution_by_lead("original_weekly", "train_attention_saved", original_attention_train_actual),
        summarize_distribution_by_lead("original_weekly", "test_gp_saved", original_gp_test_actual),
        summarize_distribution_by_lead("planB_24week", "train_attention_saved", planb_attention_train_actual),
        summarize_distribution_by_lead("planB_24week", "test_gp_saved", planb_gp_test_actual),
    ], ignore_index=True)
    distribution_lead_df.to_csv(OUT_DIR / "target_distribution_by_lead.csv", index=False)

    distribution_month_df = pd.concat([
        summarize_distribution_by_time("original_weekly", "train_attention_saved", original_attention_train_actual, orig_train_target_dates, "month", lambda d: pd.DatetimeIndex(d.reshape(-1)).month.to_numpy().reshape(d.shape)),
        summarize_distribution_by_time("original_weekly", "test_gp_saved", original_gp_test_actual, orig_test_target_dates, "month", lambda d: pd.DatetimeIndex(d.reshape(-1)).month.to_numpy().reshape(d.shape)),
        summarize_distribution_by_time("planB_24week", "train_attention_saved", planb_attention_train_actual, planb_train_target_dates, "month", lambda d: pd.DatetimeIndex(d.reshape(-1)).month.to_numpy().reshape(d.shape)),
        summarize_distribution_by_time("planB_24week", "test_gp_saved", planb_gp_test_actual, planb_test_target_dates, "month", lambda d: pd.DatetimeIndex(d.reshape(-1)).month.to_numpy().reshape(d.shape)),
    ], ignore_index=True)
    distribution_month_df.to_csv(OUT_DIR / "target_distribution_by_month.csv", index=False)

    distribution_year_df = pd.concat([
        summarize_distribution_by_time("original_weekly", "train_attention_saved", original_attention_train_actual, orig_train_target_dates, "target_year", lambda d: pd.DatetimeIndex(d.reshape(-1)).year.to_numpy().reshape(d.shape)),
        summarize_distribution_by_time("original_weekly", "test_gp_saved", original_gp_test_actual, orig_test_target_dates, "target_year", lambda d: pd.DatetimeIndex(d.reshape(-1)).year.to_numpy().reshape(d.shape)),
        summarize_distribution_by_time("planB_24week", "train_attention_saved", planb_attention_train_actual, planb_train_target_dates, "target_year", lambda d: pd.DatetimeIndex(d.reshape(-1)).year.to_numpy().reshape(d.shape)),
        summarize_distribution_by_time("planB_24week", "test_gp_saved", planb_gp_test_actual, planb_test_target_dates, "target_year", lambda d: pd.DatetimeIndex(d.reshape(-1)).year.to_numpy().reshape(d.shape)),
    ], ignore_index=True)
    distribution_year_df.to_csv(OUT_DIR / "target_distribution_by_year.csv", index=False)

    conditional_bias_df = pd.concat([
        summarize_by_swe_bin("original_weekly", "attention", "train_attention_saved", original_attention_train_actual, original_attention_train_pred),
        summarize_by_swe_bin("original_weekly", "gp_mean", "test_gp_saved", original_gp_test_actual, original_gp_test_mean),
        summarize_by_swe_bin("planB_24week", "attention", "test_attention_saved", planb_attention_test_actual, planb_attention_test_pred),
        summarize_by_swe_bin("planB_24week", "gp_mean", "test_gp_saved", planb_gp_test_actual, planb_gp_test_mean),
    ], ignore_index=True)
    conditional_bias_df.to_csv(OUT_DIR / "conditional_bias_by_swe_bin.csv", index=False)

    lead_bias_df = pd.concat([
        summarize_by_lead("original_weekly", "attention", "train_attention_saved", original_attention_train_actual, original_attention_train_pred),
        summarize_by_lead("original_weekly", "gp_mean", "test_gp_saved", original_gp_test_actual, original_gp_test_mean),
        summarize_by_lead("planB_24week", "attention", "test_attention_saved", planb_attention_test_actual, planb_attention_test_pred),
        summarize_by_lead("planB_24week", "gp_mean", "test_gp_saved", planb_gp_test_actual, planb_gp_test_mean),
    ], ignore_index=True)
    lead_bias_df.to_csv(OUT_DIR / "bias_by_lead_week.csv", index=False)

    station_group_frames = []
    station_group_bin_frames = []
    station_peak_frames = []
    for args in [
        ("original_weekly", "attention", "train_attention_saved", original_attention_train_actual, original_attention_train_pred),
        ("original_weekly", "gp_mean", "test_gp_saved", original_gp_test_actual, original_gp_test_mean),
        ("planB_24week", "attention", "test_attention_saved", planb_attention_test_actual, planb_attention_test_pred),
        ("planB_24week", "gp_mean", "test_gp_saved", planb_gp_test_actual, planb_gp_test_mean),
    ]:
        lead_df, bin_df, peak_df = summarize_station_group(*args)
        station_group_frames.append(lead_df)
        station_group_bin_frames.append(bin_df)
        peak_df["dataset"] = args[0]
        peak_df["stage"] = args[1]
        peak_df["source"] = args[2]
        station_peak_frames.append(peak_df)
    station_group_lead_df = pd.concat(station_group_frames, ignore_index=True)
    station_group_bin_df = pd.concat(station_group_bin_frames, ignore_index=True)
    station_peak_df = pd.concat(station_peak_frames, ignore_index=True)
    station_group_lead_df.to_csv(OUT_DIR / "station_group_bias_by_lead.csv", index=False)
    station_group_bin_df.to_csv(OUT_DIR / "station_group_bias_by_swe_bin.csv", index=False)
    station_peak_df.to_csv(OUT_DIR / "station_peak_groups.csv", index=False)

    save_pred_vs_actual_figure(
        OUT_DIR / "predicted_vs_actual_original_weekly.png",
        "Original weekly 4-lead saved outputs",
        original_attention_train_actual,
        original_attention_train_pred,
        original_gp_test_actual,
        original_gp_test_mean,
        original_gp_test_low,
        original_gp_test_high,
    )
    save_pred_vs_actual_figure(
        OUT_DIR / "predicted_vs_actual_planB_24week.png",
        "Plan B 24-week saved outputs",
        planb_attention_test_actual,
        planb_attention_test_pred,
        planb_gp_test_actual,
        planb_gp_test_mean,
        planb_gp_test_low,
        planb_gp_test_high,
    )
    save_bias_by_bin(OUT_DIR / "bias_by_actual_swe_bin.png", conditional_bias_df)
    save_bias_by_lead(OUT_DIR / "bias_by_lead_week.png", lead_bias_df)
    save_target_distribution(
        OUT_DIR / "target_distribution_by_lead.png",
        original_attention_train_actual,
        original_gp_test_actual,
        planb_attention_train_actual,
        planb_gp_test_actual,
    )
    save_attention_vs_gp_comparison(OUT_DIR / "attention_vs_gp_bias_comparison.png", conditional_bias_df, lead_bias_df)
    save_three_bin_scatter(OUT_DIR / "original_weekly_attention_3panel.png", original_attention_train_actual, original_attention_train_pred, "Original weekly attention vs actual by actual SWE range")
    save_three_bin_scatter(OUT_DIR / "planB_gp_3panel.png", planb_gp_test_actual, planb_gp_test_mean, "Plan B GP mean vs actual by actual SWE range")

    key_rows = []
    for dataset_name, stage, source in [
        ("original_weekly", "attention", "train_attention_saved"),
        ("original_weekly", "gp_mean", "test_gp_saved"),
        ("planB_24week", "attention", "test_attention_saved"),
        ("planB_24week", "gp_mean", "test_gp_saved"),
    ]:
        subset = conditional_bias_df[
            (conditional_bias_df["dataset"] == dataset_name)
            & (conditional_bias_df["stage"] == stage)
            & (conditional_bias_df["source"] == source)
            & (conditional_bias_df["swe_bin"].isin(["0-100", "100-300", "300-600", "600-1000", ">1000"]))
        ]
        low_row = subset[subset["swe_bin"] == "0-100"].iloc[0]
        high_row = subset[subset["swe_bin"] == ">1000"].iloc[0]
        lead_subset = lead_bias_df[
            (lead_bias_df["dataset"] == dataset_name)
            & (lead_bias_df["stage"] == stage)
            & (lead_bias_df["source"] == source)
        ]
        key_rows.append({
            "dataset": dataset_name,
            "stage": stage,
            "source": source,
            "low_bin_mean_error_mm": low_row["mean_error"],
            "high_bin_mean_error_mm": high_row["mean_error"],
            "mean_lead_slope": lead_subset["slope"].mean(),
            "max_lead_bias_mm": lead_subset["mean_error"].max(),
            "min_lead_bias_mm": lead_subset["mean_error"].min(),
        })
    key_evidence_df = pd.DataFrame(key_rows)
    key_evidence_df.to_csv(OUT_DIR / "key_bias_evidence.csv", index=False)

    plot_paths = [
        "predicted_vs_actual_original_weekly.png",
        "predicted_vs_actual_planB_24week.png",
        "bias_by_actual_swe_bin.png",
        "bias_by_lead_week.png",
        "target_distribution_by_lead.png",
        "attention_vs_gp_bias_comparison.png",
        "original_weekly_attention_3panel.png",
        "planB_gp_3panel.png",
    ]

    report_lines = []
    report_lines.append("# ForeSWE Bias Diagnosis")
    report_lines.append("")
    report_lines.append("## Files inspected")
    report_lines.extend([
        "- `thirdparties/SWE-Forecasting/fore_swe/forecasting_att_weekly_model.py`",
        "- `thirdparties/SWE-Forecasting/fore_swe/forecasting_weekly_gp_model.py`",
        "- `thirdparties/SWE-Forecasting/fore_swe/extended_dl_forecasting.py`",
        "- `experiments/foreswe_planB_24week/planB_24week_att_model.py`",
        "- `experiments/foreswe_planB_24week/planB_24week_gp_model.py`",
        "- `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_weekly_reproduction_full512/...` saved outputs",
        "- `experiments/foreswe_planB_24week/...` saved outputs",
    ])
    report_lines.append("")
    report_lines.append("## Code-level diagnosis")
    report_lines.append("")
    report_lines.append("### Attention loss")
    report_lines.extend([
        "- Original weekly attention loss: `nn.MSELoss()` at `forecasting_att_weekly_model.py:244`, applied at `forecasting_att_weekly_model.py:348-355` during training and again at `forecasting_att_weekly_model.py:517-524` during test logging.",
        "- Plan B attention loss: `nn.MSELoss()` at `planB_24week_att_model.py:282`, applied at `planB_24week_att_model.py:320-324`.",
        "- The loss is plain MSE on raw SWE targets. No high-SWE weighting, no lead weighting, and no station-group weighting were found in either attention script.",
    ])
    report_lines.append("")
    report_lines.append("### Target normalization and inverse transform")
    report_lines.extend([
        "- Inputs are globally min-max normalized across all station columns in `extended_dl_forecasting.py:20-29`.",
        "- Input normalization is applied in `extended_dl_forecasting.py:418-435` to meteorology, partitioned rain/snow, input SWE, and day length.",
        "- `extended_dl_forecasting.py:257-271` contains a likely feature bug: `self.snow_info` is overwritten from `self.rain_info` at line 270, and `self.rain_info` is then rebuilt from the already overwritten `self.snow_info` at line 271. That means the shared dataloader used by both the original weekly model and Plan B likely corrupts or duplicates the rain/snow feature channels.",
        "- The target SWE is **not normalized** before the attention loss. `getModelInputsAndOutputs()` stores raw SWE targets from `self.swe_values` in `extended_dl_forecasting.py:123-131`.",
        "- `getMaxMinSWE()` returns `(highest_swe + 500, 0)` in `extended_dl_forecasting.py:410-416`, but this is used only when replacing SWE inputs during recursive daily prediction, not to scale the weekly training targets.",
        "- Original weekly GP and Plan B GP both define a no-op input normalizer (`return tensor`) in `forecasting_weekly_gp_model.py:66-68` and `planB_24week_gp_model.py:93-95`.",
        "- No inverse transform is applied to the saved attention or GP outputs because the targets are already on the raw SWE scale.",
        "- No clipping, sigmoid, tanh, ReLU, softplus, or clamp is applied to the final SWE outputs in either attention head.",
        "- SWE zeros are kept as ordinary training targets; there is no special positive-only transform or zero inflation handling.",
    ])
    report_lines.append("")
    report_lines.append("### Output head")
    report_lines.extend([
        "- Original weekly attention head: `Linear(3*model_dim -> model_dim)`, `GELU`, `Dropout(0.05)`, `Linear(model_dim -> 128)`, `GELU`, `Linear(128 -> 8)`, `GELU`, final `Linear(8 -> n_output_heads)` in `forecasting_att_weekly_model.py:203-214`.",
        "- Plan B attention head uses the same structure for `n_output_heads=24` in `planB_24week_att_model.py:255-264`.",
        "- The final projection is linear, so the head can represent large SWE values in principle. There is no positivity constraint or explicit large-SWE amplification.",
    ])
    report_lines.append("")
    report_lines.append("### GP model")
    report_lines.extend([
        "- Original weekly GP sets `torch.float64` in `forecasting_weekly_gp_model.py:29`, reshapes training labels to `(-1, 4)` in `forecasting_weekly_gp_model.py:70-72`, and trains a multitask variational GP with `LMCVariationalStrategy` in `forecasting_weekly_gp_model.py:83-129`.",
        "- Original weekly GP likelihood is `MultitaskGaussianLikelihood(num_tasks=4)` at `forecasting_weekly_gp_model.py:143-157` and the training objective is `-VariationalELBO` at `forecasting_weekly_gp_model.py:157-167`.",
        "- Plan B GP also uses `torch.float64`, leaves the 8-dim attention features unnormalized, and trains `MultitaskGPModel(ApproximateGP)` with `LMCVariationalStrategy`, `ScaleKernel(RBF + Linear)`, and `MultitaskGaussianLikelihood(num_tasks=24)` in `planB_24week_gp_model.py:139-200`.",
        "- The GP mean is the saved `preds.mean`, while the intervals come from `confidence_region()` in both weekly and Plan B GP scripts.",
    ])
    report_lines.append("")
    report_lines.append("## Loaded tensor shapes")
    report_lines.append("")
    report_lines.append(table_to_markdown(pd.DataFrame(shapes_rows), ["name", "shape"], round_digits=0))
    report_lines.append("")
    report_lines.append("### Original weekly GP saved-output alignment")
    report_lines.append("")
    report_lines.append("The original weekly GP reproduction artifacts are not perfectly self-consistent: `2016_weekly_gp_window_predict.pkl`, `2016_weekly_gp_low_window_predict.pkl`, and `2016_weekly_gp_high_window_predict.pkl` each have 182 issue rows, while `2016_weekly_window_actual.pkl` has 180 rows. The analysis aligned each year to the common saved row count and wrote the exact bookkeeping to `original_weekly_gp_year_alignment.csv`.")
    report_lines.append("")
    report_lines.append("### Source caveat for original weekly attention")
    report_lines.append("")
    report_lines.append("The expected original weekly attention test pickle files (`2014_weekly_window_predict.pkl` ... `2018_weekly_window_predict.pkl`) are not present in the saved reproduction directory, even though the script contains `joblib.dump` calls for them. The original weekly attention-vs-actual analysis therefore uses the saved attention outputs that were written for GP training (`forecasting_modified_weekly_att_predicted_outputs_for_gp_act.pt` and `forecasting_modified_weekly_att_outputs_for_gp_act.pt`), while the original weekly GP analysis uses the saved per-year GP test outputs.")
    report_lines.append("")
    report_lines.append("## Generated plots")
    report_lines.extend([f"- `{path}`" for path in plot_paths])
    report_lines.append("")
    report_lines.append("## Distribution imbalance")
    report_lines.append("")
    report_lines.append(table_to_markdown(distribution_bin_df, ["dataset", "split", "swe_bin", "count", "fraction"], round_digits=4))
    report_lines.append("")
    report_lines.append("## Conditional bias by actual SWE bin")
    report_lines.append("")
    report_lines.append(table_to_markdown(conditional_bias_df, ["dataset", "stage", "source", "swe_bin", "count", "mean_error", "relative_bias", "rmse", "mae", "corr", "slope"], round_digits=3))
    report_lines.append("")
    report_lines.append("## Lead-dependent bias")
    report_lines.append("")
    report_lines.append(table_to_markdown(lead_bias_df, ["dataset", "stage", "source", "lead", "mean_actual", "mean_pred", "mean_error", "relative_bias", "rmse", "mae", "slope", "intercept", "r2"], round_digits=3, head=40))
    report_lines.append("")
    report_lines.append("## Station-group bias files")
    report_lines.extend([
        "- `station_group_bias_by_lead.csv`",
        "- `station_group_bias_by_swe_bin.csv`",
        "- `station_peak_groups.csv`",
    ])
    report_lines.append("")
    report_lines.append("## Bias before vs after GP")
    report_lines.append("")
    report_lines.append(table_to_markdown(key_evidence_df, ["dataset", "stage", "source", "low_bin_mean_error_mm", "high_bin_mean_error_mm", "mean_lead_slope", "max_lead_bias_mm", "min_lead_bias_mm"], round_digits=3))
    report_lines.append("")
    report_lines.append("Interpretation from the saved outputs:")
    report_lines.extend([
        "- In both datasets, the attention head already shows positive mean error in low-SWE bins and negative mean error in high-SWE bins, so the regression-to-the-mean behavior starts before the GP.",
        "- The GP mean usually keeps the same sign pattern and often tightens the slope further toward 1 only weakly, which is consistent with the GP acting as a smooth multitask shrinkage layer rather than fixing the conditional bias.",
        "- Because the attention loss is unweighted raw-scale MSE, the model is optimized for average squared error over a highly imbalanced target distribution with many zeros and many low-SWE samples. That setup gives much more statistical leverage to the abundant small-to-medium targets than to the rare large-SWE cases.",
        "- The output head is linear and unconstrained, so the evidence points more strongly to loss/distribution effects than to a hard architectural saturation.",
    ])
    report_lines.append("")
    report_lines.append("## Likely causes ranked by evidence")
    report_lines.extend([
        "1. Unweighted raw-scale MSE on a heavily imbalanced SWE target distribution.",
        "2. Shared single head over all stations and leads encourages average behavior across many low-SWE and zero-SWE cases.",
        "3. Multitask variational GP smoothing preserves and can slightly reinforce the attention head's mean-regression pattern.",
        "4. Lack of target transform or anomaly decomposition means high-end SWE is learned on the same linear scale as the dominant low-SWE mass.",
        "5. The shared dataloader's rain/snow overwrite bug likely reduces the quality of accumulation-related predictors and can amplify conservative predictions, although the saved outputs alone do not isolate how much of the conditional bias it explains.",
    ])
    report_lines.append("")
    report_lines.append("## Recommended next model change")
    report_lines.append("")
    report_lines.append("Most justified next change from the code and saved outputs: **Option 1 plus Option 2, starting with a weighted loss on the attention head**.")
    report_lines.extend([
        "- Weighted loss is the smallest code change because it only touches the attention training objective and leaves the data interface and GP stage unchanged.",
        "- The saved outputs already show the sign pattern expected from underweighting rare large-SWE targets and overweighting abundant low-SWE targets under plain MSE.",
        "- A second step worth testing immediately after weighted MSE is a `log1p(SWE)` or anomaly-style target, because the current training never rescales the raw target distribution.",
    ])
    report_lines.append("")
    report_lines.append("## Exact next experiment command")
    report_lines.append("")
    report_lines.append("```bash")
    report_lines.append("cd /global/homes/h/hyvchen/Snow-Predication-at-Sierra")
    report_lines.append("# Recommended next change: weighted attention loss before the GP")
    report_lines.append("python experiments/foreswe_planB_24week/planB_24week_att_model.py")
    report_lines.append("```")
    report_lines.append("")
    report_lines.append("Suggested implementation detail for that next experiment: replace `nn.MSELoss()` with unreduced MSE and multiply by a weight such as `1 + alpha * clamp(actual / 300, 0, cap)` before averaging.")
    (OUT_DIR / "bias_diagnosis_report.md").write_text("\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
