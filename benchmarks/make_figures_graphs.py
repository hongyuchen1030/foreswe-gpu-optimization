"""Figures for the CUDA-graph / torch.compile study and the fused-attention study (CPU only).

    python -m benchmarks.make_figures_graphs
"""
import csv
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from foreswe_bench import paths  # noqa: E402

CG = paths.REPO / "engineering_audit" / "optimization" / "cuda_graphs"
FA = paths.REPO / "engineering_audit" / "optimization" / "fused_attention"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {"G0": "#2a78d6", "G1": "#eb6834", "G1L": "#f2a77f", "G2": "#1baf7a", "G2L": "#7fd3b2", "G3": "#6250d6",
       "G3f": "#a79ce9", "G4": "#e34948", "G0c": "#7fb0eb"}
LABEL = {"G0": "G0 E4 eager", "G0c": "G0c eager, capturable AdamW", "G1": "G1 compile default", "G1L": "G1L compile layers",
         "G2": "G2 compile reduce-overhead", "G2L": "G2L reduce-overhead layers", "G3f": "G3f graph fwd+bwd",
         "G3": "G3 graph full step", "G4": "G4 compiled kernels in graph"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130, "legend.frameon": False})


def save(fig, d, name):
    os.makedirs(d / "figures", exist_ok=True)
    fig.tight_layout()
    fig.savefig(d / "figures" / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", d / "figures" / name)


def fig_step_times():
    s = json.load(open(CG / "benchmark_results" / "nodeA" / "matrix_summary.json"))["summary"]
    cfgs = [c for c in ("G0", "G0c", "G1", "G1L", "G2", "G2L", "G3f", "G3", "G4") if c in s and not s[c].get("failed")]
    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    y = np.arange(len(cfgs))
    e2e = np.array([s[c]["e2e_ms"] for c in cfgs])
    gpu = np.array([s[c]["gpu_region_ms"] for c in cfgs])
    prep = np.array([s[c]["prep_ms"] for c in cfgs])
    ax.barh(y + 0.2, e2e, height=0.38, color=[COL[c] for c in cfgs], edgecolor="white", linewidth=1.5,
            label="B: end-to-end step (incl. data prep, input copy, loss sync)")
    ax.barh(y - 0.2, gpu, height=0.38, color=[COL[c] for c in cfgs], alpha=0.45, edgecolor="white", linewidth=1.5,
            label="A: accelerated region only (staged inputs)")
    for i, c in enumerate(cfgs):
        ax.text(e2e[i] + 0.6, i + 0.2, "{:.1f} ms  ({:.2f}x)".format(e2e[i], s["G0"]["e2e_ms"] / e2e[i]), va="center",
                fontsize=7.5, color=INK)
        ax.text(gpu[i] + 0.6, i - 0.2, "{:.1f}".format(gpu[i]), va="center", fontsize=7, color=MUTED)
    ax.set_yticks(y, [LABEL[c] for c in cfgs])
    ax.invert_yaxis()
    ax.set_xlim(0, e2e.max() * 1.3)
    ax.set_xlabel("median ms per training step (A100-40GB, TF32 + fused AdamW, 120 steps); data prep ~{:.1f} ms in every B bar".format(np.median(prep)))
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("E4 training step: torch.compile and CUDA Graphs (speedup vs G0 = E4 eager)", loc="left", color=INK)
    save(fig, CG, "step_times.png")


def fig_launches():
    rows = {}
    for c in ("G0", "G1", "G2", "G3", "G4"):
        f = CG / "profiler_summaries" / "nsys" / "nsys_{}.json".format(c)
        if f.exists():
            rows[c] = json.load(open(f))["median"]
    if not rows:
        return
    cfgs = list(rows)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))
    x = np.arange(len(cfgs))
    axes[0].bar(x - 0.2, [rows[c]["kernels"] for c in cfgs], width=0.38, color="#52514e", label="GPU kernels executed")
    axes[0].bar(x + 0.2, [rows[c]["api_kernel_launches"] + rows[c]["api_graph_launches"] for c in cfgs], width=0.38,
                color=[COL[c] for c in cfgs], label="host launch submissions (kernel + graph)")
    axes[0].set_xticks(x, cfgs)
    axes[0].set_title("work executed vs work submitted (per step)", loc="left", color=INK)
    axes[0].legend(fontsize=7)
    axes[1].bar(x, [rows[c]["busy_pct"] for c in cfgs], color=[COL[c] for c in cfgs])
    for i, c in enumerate(cfgs):
        axes[1].text(i, rows[c]["busy_pct"] + 1, "{:.0f}%\nidle {:.1f} ms".format(rows[c]["busy_pct"], rows[c]["idle_ms"]),
                     ha="center", fontsize=7, color=INK)
    axes[1].set_ylim(0, 110)
    axes[1].set_xticks(x, cfgs)
    axes[1].set_title("GPU busy within the step (Nsight Systems)", loc="left", color=INK)
    save(fig, CG, "launches_and_busy.png")


def fig_timeline():
    cfgs = [c for c in ("G0", "G3", "G4") if (CG / "profiler_summaries" / "timelines" / "timeline_{}.csv".format(c)).exists()]
    if not cfgs:
        return
    fig, axes = plt.subplots(len(cfgs), 1, figsize=(8.4, 1.25 * len(cfgs) + 0.6), sharex=True)
    for ax, c in zip(np.atleast_1d(axes), cfgs):
        rows = list(csv.DictReader(open(CG / "profiler_summaries" / "timelines" / "timeline_{}.csv".format(c))))
        k = [(float(r["start_us"]) / 1e3, float(r["dur_us"]) / 1e3) for r in rows if r["track"] in ("gpu_kernel", "gpu_memcpy")]
        h = [float(r["start_us"]) / 1e3 for r in rows if r["track"] == "host_api" and "Launch" in r["category"]]
        g = [float(r["start_us"]) / 1e3 for r in rows if r["track"] == "host_api" and "GraphLaunch" in r["category"]]
        ax.broken_barh(k, (0.55, 0.4), facecolors=COL[c], linewidth=0)
        ax.vlines(h, 0.05, 0.4, color=MUTED, linewidth=0.3)
        if g:
            ax.vlines(g, 0.0, 0.45, color="#e34948", linewidth=2)
        end = max(s + d for s, d in k)
        ax.set_yticks([0.22, 0.75], ["host launches", "GPU"])
        ax.set_title("{}: step window {:.1f} ms under nsys, {} GPU kernels, {} host launch calls{}".format(
            LABEL[c], end, sum(1 for r in rows if r["track"] == "gpu_kernel"), len(h),
            " (red = cudaGraphLaunch)" if g else ""), loc="left", fontsize=8, color=INK)
        ax.grid(False)
    np.atleast_1d(axes)[-1].set_xlabel("ms from start of the training step (data prep first, then the step)")
    save(fig, CG, "timeline_G0_G3_G4.png")


def fig_convergence():
    d = CG / "benchmark_results" / "convergence"
    fig, ax = plt.subplots(figsize=(7.4, 2.8))
    for f in sorted(glob.glob(str(d / "G*_3297steps_trace.csv"))) or sorted(glob.glob(str(d / "G*_300steps_trace.csv"))):
        c = os.path.basename(f).split("_")[0]
        rows = list(csv.DictReader(open(f)))
        L = np.array([float(r["loss"]) for r in rows])
        k = max(1, len(L) // 30)
        ax.plot(np.arange(k, len(L) + 1), np.convolve(L, np.ones(k) / k, mode="valid"), color=COL.get(c, INK),
                linewidth=1.5, linestyle="--" if c != "G0" else "-", label=LABEL.get(c, c))
    ax.set_yscale("log")
    ax.set_xlabel("optimizer step (real Plan B data, production order, one epoch)")
    ax.set_title("training loss, moving mean", loc="left", color=INK)
    ax.legend(fontsize=7)
    save(fig, CG, "convergence_epoch.png")


def fig_attention():
    f = FA / "benchmark_results" / "attention_study.json"
    if not f.exists():
        return
    p = json.load(open(f))["perf_tf32"]
    impls = ["eager", "compiled", "fused"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.6))
    for ax, key, title in ((axes[0], "fwd_ms", "forward"), (axes[1], "fwd_bwd_ms", "forward + backward")):
        v = [p[i][key]["median"] for i in impls]
        ax.barh(range(3), v, color=["#2a78d6", "#1baf7a", "#e34948"])
        for i, x in enumerate(v):
            ax.text(x + max(v) * 0.02, i, "{:.2f} ms".format(x), va="center", fontsize=8, color=INK)
        ax.set_yticks(range(3), ["eager (production)", "torch.compile", "fused Triton prototype"])
        ax.invert_yaxis()
        ax.set_xlim(0, max(v) * 1.35)
        ax.set_title("geo-attention core, one layer, TF32: " + title, loc="left", color=INK, fontsize=8.5)
    save(fig, FA, "attention_core_latency.png")


if __name__ == "__main__":
    for fn in (fig_step_times, fig_launches, fig_timeline, fig_convergence, fig_attention):
        fn()
