"""Render report figures from the git-safe JSON/CSV results (CPU only; seconds).

    python -m benchmarks.make_figures
Encoding: precision -> hue (FP32 blue, TF32 orange, BF16 aqua), optimizer -> line style / hatch
(foreach solid, fused dashed / hatched).
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

R = paths.RESULTS_DIR
FIG = paths.REPO / "engineering_audit" / "optimization" / "figures"
HUE = {"fp32": "#2a78d6", "tf32": "#eb6834", "bf16": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
EXPS = {"E0": ("fp32", "foreach"), "E1": ("tf32", "foreach"), "E2": ("bf16", "foreach"),
        "E3": ("fp32", "fused"), "E4": ("tf32", "fused"), "E5": ("bf16", "fused")}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130,
                     "legend.frameon": False})


def label(e):
    p, o = EXPS[e]
    return "{} {}+{}".format(e, p.upper(), o)


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / name)


def fig_step_breakdown():
    for tag in ("node1_synthetic", "node2_synthetic"):
        p = R / tag / "matrix_summary.json"
        if not p.exists():
            continue
        s = json.load(open(p))["summary"]
        exps = [e for e in EXPS if e in s]
        fig, ax = plt.subplots(figsize=(7.2, 3.2))
        parts = [("prep_ms", "data prep", "#c3c2b7"), ("forward_ms", "forward+loss", None),
                 ("backward_ms", "backward", None), ("optimizer_ms", "AdamW step", None)]
        y = np.arange(len(exps))
        left = np.zeros(len(exps))
        shades = {"forward_ms": 1.0, "backward_ms": 0.7, "optimizer_ms": 0.4}
        for key, name, col in parts:
            vals = np.array([s[e][key] for e in exps])
            colors = [col or HUE[EXPS[e][0]] for e in exps]
            alphas = shades.get(key, 1.0)
            ax.barh(y, vals, left=left, color=colors, alpha=alphas, edgecolor="white", linewidth=2,
                    hatch=None, label=name)
            left += vals
        for i, e in enumerate(exps):
            ax.text(s[e]["step_ms"]["median"] + 2, i, "{:.1f} ms  ({:.2f}x)".format(s[e]["step_ms"]["median"],
                    s[e].get("speedup_vs_E0", 1)), va="center", color=INK, fontsize=8)
        ax.set_yticks(y, [label(e) for e in exps])
        ax.invert_yaxis()
        ax.set_xlabel("median training-step time (ms); segments = forward, backward, AdamW (light->dark = later)")
        ax.set_xlim(0, max(s[e]["step_ms"]["median"] for e in exps) * 1.35)
        ax.set_title("Plan B training step by phase - {} ({})".format(tag.split("_")[0], s["E0"]["env"]["gpu"]),
                     loc="left", color=INK)
        save(fig, "step_breakdown_{}.png".format(tag))


def fig_throughput():
    tags = [t for t in ("node1_synthetic", "node2_synthetic") if (R / t / "matrix_summary.json").exists()]
    if not tags:
        return
    fig, axes = plt.subplots(1, len(tags), figsize=(3.6 * len(tags), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, tag in zip(axes, tags):
        s = json.load(open(R / tag / "matrix_summary.json"))["summary"]
        exps = [e for e in EXPS if e in s]
        thr = [1e3 / s[e]["step_ms"]["median"] for e in exps]
        lo = [1e3 / max(s[e]["step_ms_repeat_medians"]) for e in exps]
        hi = [1e3 / min(s[e]["step_ms_repeat_medians"]) for e in exps]
        x = np.arange(len(exps))
        for i, e in enumerate(exps):
            ax.bar(i, thr[i], color=HUE[EXPS[e][0]], hatch="//" if EXPS[e][1] == "fused" else None,
                   edgecolor="white", linewidth=2, width=0.7)
            ax.plot([i, i], [lo[i], hi[i]], color=INK, linewidth=1)
            ax.text(i, thr[i] + 0.3, "{:.1f}".format(thr[i]), ha="center", fontsize=8, color=INK)
        ax.set_xticks(x, exps)
        ax.set_title(tag.replace("_", " "), loc="left", color=INK)
    axes[0].set_ylabel("training throughput (issue dates / s)\n1 issue date = 512 station samples")
    fig.suptitle("Throughput by configuration (hatched = fused AdamW; whisker = range over repeats)", x=0.01, ha="left",
                 fontsize=9, color=INK)
    save(fig, "throughput.png")


def fig_numerics():
    p = R / "numerics_precision" / "numerics_precision.csv"
    if not p.exists():
        return
    rows = list(csv.DictReader(open(p)))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    cases = [("tf32", "fp32"), ("bf16", "fp32"), ("fp32", "fp64"), ("tf32", "fp64"), ("bf16", "fp64")]
    for ax, metric, title in zip(axes, ("out_rel_l2", "grad_rel_l2"), ("output", "gradient (live params)")):
        for j, w in enumerate(("init", "planB")):
            vals = [np.median([float(r[metric]) for r in rows if r["weights"] == w and r["precision"] == p and
                               r["reference"] == ref]) for p, ref in cases]
            x = np.arange(len(cases)) + (j - 0.5) * 0.36
            ax.bar(x, vals, width=0.34, color=[HUE[p] for p, _ in cases], alpha=1.0 if w == "planB" else 0.5,
                   edgecolor="white", linewidth=2, label="trained Plan B weights" if w == "planB" else "seed-0 init")
        ax.set_yscale("log")
        ax.set_xticks(np.arange(len(cases)), ["{}\nvs {}".format(p.upper(), r.upper()) for p, r in cases], fontsize=8)
        ax.set_title("relative L2 error: " + title, loc="left", color=INK)
    axes[0].legend(fontsize=8)
    fig.suptitle("Median over 9 real issue dates (4 train, 5 Feb-1 test); eval mode, before any update", x=0.01,
                 ha="left", fontsize=9, color=INK)
    save(fig, "numerical_error.png")


def fig_error_propagation():
    p = R / "error_origin" / "error_origin.json"
    if not p.exists():
        return
    d = json.load(open(p))["modulewise_vs_fp64"]
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    for prec in ("fp32", "tf32", "bf16"):
        recs = [r for r in d[prec] if r["module"].count(".") <= 1 and not r["module"].endswith(("W_q", "W_k", "W_v"))]
        ax.plot(range(len(recs)), [r["rel_l2"] for r in recs], color=HUE[prec], linewidth=2, marker="o", markersize=3,
                label=prec.upper())
    ticks = [r["module"].replace("encoder_layers.", "enc").replace("historical_", "").replace("_att", "")
             for r in recs]
    ax.set_xticks(range(len(ticks)), ticks, rotation=70, fontsize=6)
    ax.set_yscale("log")
    ax.set_ylabel("rel. L2 error vs FP64")
    ax.set_title("Error along the forward pass (Plan B weights, Feb-1-2017 issue)", loc="left", color=INK)
    ax.legend(ncol=3, fontsize=8)
    save(fig, "error_propagation.png")


def fig_convergence():
    files = sorted(glob.glob(str(R / "convergence" / "E*_trace.csv")))
    for steps in sorted({os.path.basename(f).split("_")[1] for f in files}):
        sel = [f for f in files if "_{}_".format(steps) in os.path.basename(f)]
        if not sel:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
        for f in sel:
            e = os.path.basename(f).split("_")[0]
            rows = list(csv.DictReader(open(f)))
            st = np.array([int(r["step"]) for r in rows])
            loss = np.array([float(r["loss"]) for r in rows])
            g = np.array([float(r["grad_norm"]) for r in rows])
            k = max(1, len(st) // 30)
            sm = np.convolve(loss, np.ones(k) / k, mode="valid")
            ls = "--" if EXPS[e][1] == "fused" else "-"
            axes[0].plot(st[k - 1:], sm, color=HUE[EXPS[e][0]], linestyle=ls, linewidth=1.6, label=label(e))
            axes[1].plot(st[k - 1:], np.convolve(g, np.ones(k) / k, mode="valid"), color=HUE[EXPS[e][0]],
                         linestyle=ls, linewidth=1.6)
        axes[0].set_yscale("log")
        axes[0].set_title("training loss (MSE, {}-step moving mean)".format(k), loc="left", color=INK)
        axes[1].set_title("gradient L2 norm (moving mean)", loc="left", color=INK)
        axes[1].set_yscale("log")
        for ax in axes:
            ax.set_xlabel("optimizer step (real Plan B data, production order)")
        axes[0].legend(fontsize=7)
        save(fig, "convergence_{}.png".format(steps))


def fig_kernels():
    files = sorted(glob.glob(str(paths.PROFILING_SUMMARY_DIR / "torch_profiler" / "profile_E*.json")))
    if not files:
        return
    data = {json.load(open(f))["exp"]: json.load(open(f)) for f in files}
    exps = [e for e in EXPS if e in data]
    cats = []
    for e in exps:
        for c, *_ in data[e]["categories"]:
            if c not in cats:
                cats.append(c)
    cat_col = {"GEMM (Tensor Core)": "#eb6834", "GEMM (SIMT FP32)": "#2a78d6",
               "AdamW foreach (multi_tensor_apply)": "#6250d6", "AdamW fused": "#9085e9",
               "dtype casts (to/copy)": "#e34948", "index gather": "#eda100"}
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    y = np.arange(len(exps))
    left = np.zeros(len(exps))
    for c in cats:
        vals = np.array([next((r[1] for r in data[e]["categories"] if r[0] == c), 0.0) for e in exps])
        if vals.max() < 0.3:
            continue
        ax.barh(y, vals, left=left, color=cat_col.get(c, "#c3c2b7"), edgecolor="white", linewidth=1.5, label=c)
        left += vals
    for i, e in enumerate(exps):
        ax.text(left[i] + 1, i, "{:.0f} ms, {:.0f} launches, TC {:.0f}% of GEMM".format(
            data[e]["gpu_kernel_ms_per_step"], data[e]["launches_per_step"],
            100 * data[e]["tensor_core_share_of_gemm_time"]), va="center", fontsize=7, color=INK)
    ax.set_yticks(y, [label(e) for e in exps])
    ax.invert_yaxis()
    ax.set_xlim(0, left.max() * 1.6)
    ax.set_xlabel("GPU kernel time per training step (ms), torch.profiler")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Kernel-time breakdown and Tensor-Core usage", loc="left", color=INK)
    save(fig, "kernel_breakdown.png")


def fig_optimizer():
    p = R / "fused_adamw" / "fused_adamw_study.json"
    if not p.exists():
        return
    t = json.load(open(p))["timing"]
    impls = ["single_tensor", "foreach", "fused"]
    fig, ax = plt.subplots(figsize=(5.4, 2.4))
    vals = [t[i]["step_ms"]["median"] for i in impls]
    ax.barh(range(3), vals, color=["#c3c2b7", "#6250d6", "#9085e9"], edgecolor="white", linewidth=2)
    for i, impl in enumerate(impls):
        ax.text(vals[i] + 0.5, i, "{:.1f} ms, {} kernel launches".format(vals[i], t[impl]["kernel_launches"]),
                va="center", fontsize=8, color=INK)
    ax.set_yticks(range(3), ["AdamW " + i for i in impls])
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.7)
    ax.set_xlabel("optimizer.step() time, 426M live FP32 params (ms)")
    ax.set_title("AdamW implementation cost (fixed gradients)", loc="left", color=INK)
    save(fig, "optimizer_timing.png")


if __name__ == "__main__":
    for f in (fig_step_breakdown, fig_throughput, fig_numerics, fig_error_propagation, fig_convergence, fig_kernels,
              fig_optimizer):
        f()
