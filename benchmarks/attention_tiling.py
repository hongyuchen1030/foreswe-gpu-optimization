"""Task 1: D=192 tiling of the fused geo-attention kernel - bounded per-kernel sweep and final comparison.

Production shape [1, 16, 512, 192], real layer-0 activations (Plan B checkpoint), TF32 (+ FP32 timings).
Every swept config must first pass the correctness gate (FP32 vs FP64 within 10x eager FP32 error on
random inputs, the same criterion as tests/test_gpu_graphs.py).

    python -m benchmarks.attention_tiling [--quick]
"""
import argparse
import csv
import itertools
import os

import torch

from benchmarks.attention_study import capture_real_qkv, compare, kernel_profile, make_inputs, time_fn
from foreswe_bench import data as fdata
from foreswe_bench import geo_attention as v1
from foreswe_bench import geo_attention_v2 as v2
from foreswe_bench import paths
from foreswe_bench.geo_attention import eager_geo_attention
from foreswe_bench.model import load_namespace
from foreswe_bench.precision import matmul_flags
from foreswe_bench.runtime import env_info, setup_logging, write_json

OUT = paths.REPO / "engineering_audit" / "optimization" / "fused_attention" / "benchmark_results" / "tiling"
A100 = dict(regs=65536, smem=167936, max_warps=64, max_blocks=32)
GRADS = ("dQ", "dK", "dV", "dalpha", "dbeta", "dgamma")


def occupancy(info):
    w = info["num_warps"]
    by_regs = A100["regs"] // max(1, info["n_regs"] * 32 * w)
    by_smem = A100["smem"] // max(1, info["shared_bytes"]) if info["shared_bytes"] else A100["max_blocks"]
    blocks = min(by_regs, by_smem, A100["max_warps"] // w, A100["max_blocks"])
    return dict(blocks_per_sm=blocks, theoretical_occupancy=blocks * w / A100["max_warps"],
                limiter=min((("registers", by_regs), ("shared", by_smem), ("warps", A100["max_warps"] // w)),
                            key=lambda t: t[1])[0])


def gate(fn, inputs):
    """FP32 errors vs FP64 within 10x eager FP32's on the same inputs (+1e-7 floor)."""
    f, e = compare(fn, inputs, "fp32"), compare(eager_geo_attention, inputs, "fp32")
    worst = max([f[k]["rel_l2"] / (e[k]["rel_l2"] * 10 + 1e-7) for k in GRADS] +
                [f["output"]["rel_l2"] / (e["output"]["rel_l2"] * 10 + 1e-7)])
    return worst <= 1.0, worst


def timings(fn, inputs, prec, warm=3, iters=15):
    g = torch.Generator(device="cpu").manual_seed(99)
    do = torch.randn(inputs[0].shape, generator=g).to(inputs[0].device)
    with matmul_flags(prec):
        x = [t.detach().clone().requires_grad_(i in (0, 1, 2, 5, 6, 7)) for i, t in enumerate(inputs)]

        def fwd():
            with torch.no_grad():
                return fn(*x)
        o = fn(*x)

        def bwd():
            for t in x:
                t.grad = None
            o.backward(do, retain_graph=True)

        def fb():
            for t in x:
                t.grad = None
            fn(*x).backward(do)
        r = dict(fwd_ms=time_fn(fwd, warm, iters)["median"], bwd_ms=time_fn(bwd, warm, iters)["median"],
                 fwd_bwd_ms=time_fn(fb, warm, iters)["median"])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        fb()
        torch.cuda.synchronize()
        r["fwd_bwd_peak_extra_mib"] = (torch.cuda.max_memory_allocated() - base) / 2**20
        r["kernels_fwd"] = kernel_profile(fwd)["kernels"]
        r["kernels_fwd_bwd"] = kernel_profile(fb)["kernels"]
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="tiny sweep (smoke test)")
    a = ap.parse_args()
    log = setup_logging()
    ns = load_namespace("planB24", "cuda")
    ds = fdata.real(ns)
    real0 = capture_real_qkv(ns, ds, layers=(0,))[0]
    rnd = make_inputs(1, 16, 512, 192, seed=5)
    default = dict(v2.CFG)
    rows = []

    def trial(kind, cfg):
        v2.CFG.update(default)
        v2.CFG[kind] = cfg
        try:
            ok, worst = gate(v2.fused_geo_attention_v2, rnd)
            t = timings(v2.fused_geo_attention_v2, real0, "tf32", warm=2, iters=10)
            info = {k: dict(v, **occupancy(v)) for k, v in v2.LAUNCH_INFO.items()}
            rec = dict(kernel=kind, cfg=list(cfg), correct=ok, worst_err_ratio=worst, **t, launch_info=info)
        except Exception as exc:  # out of resources etc.
            rec = dict(kernel=kind, cfg=list(cfg), error="{}: {}".format(type(exc).__name__, str(exc)[:120]))
        rows.append(rec)
        log.info("%s %s -> %s", kind, cfg, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()
                                             if k not in ("launch_info",)})
        return rec

    BM = (16, 32) if a.quick else (16, 32, 64)
    BN = (32,) if a.quick else (32, 64)
    W = (4,) if a.quick else (4, 8)
    ST = (2,) if a.quick else (1, 2)
    for cfg in itertools.product(BM, BN, W, ST):
        trial("fwd", cfg)
    best_fwd = min((r for r in rows if r["kernel"] == "fwd" and r.get("correct")), key=lambda r: r["fwd_ms"])["cfg"]
    default["fwd"] = tuple(best_fwd)
    for cfg in itertools.product((16, 32), BN, W, ST):
        trial("rows", cfg)
    best_rows = min((r for r in rows if r["kernel"] == "rows" and r.get("correct")), key=lambda r: r["bwd_ms"])["cfg"]
    default["rows"] = tuple(best_rows)
    for cfg in itertools.product(BM, (16, 32) if not a.quick else (32,), W, ST):
        trial("cols", cfg)
    best_cols = min((r for r in rows if r["kernel"] == "cols" and r.get("correct")), key=lambda r: r["bwd_ms"])["cfg"]
    default["cols"] = tuple(best_cols)
    v2.CFG.update(default)
    best = dict(fwd=best_fwd, rows=best_rows, cols=best_cols)
    log.info("best v2 config: %s", best)

    # final comparison (TF32 + FP32 timings), real layer-0 activations
    compiled = torch.compile(eager_geo_attention)
    final = {}
    for name, fn in (("eager", eager_geo_attention), ("inductor", compiled), ("triton_v1_padded", v1.fused_geo_attention),
                     ("triton_v2_tuned", v2.fused_geo_attention_v2)):
        final[name] = dict(tf32=timings(fn, real0, "tf32"), fp32=timings(fn, real0, "fp32"))
        if name.startswith("triton_v2"):
            final[name]["launch_info"] = {k: dict(v, **occupancy(v)) for k, v in v2.LAUNCH_INFO.items()}
        log.info("%s: %s", name, final[name]["tf32"])
    os.makedirs(OUT, exist_ok=True)
    write_json(dict(env=env_info(), best_v2_cfg=best, sweep=rows, final=final), OUT / "tiling_results.json")
    with open(OUT / "tiling_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kernel", "block_m", "block_n", "warps", "stages", "correct", "fwd_ms", "bwd_ms", "fwd_bwd_ms",
                    "regs", "spills", "shared_bytes", "occupancy", "error"])
        for r in rows:
            li = r.get("launch_info", {})
            key = {"fwd": "_fwd_kernel", "rows": "_bwd_rows_kernel_mode2", "cols": "_bwd_cols_kernel"}[r["kernel"]]
            k = li.get(key, {})
            w.writerow([r["kernel"], *r["cfg"], r.get("correct"), r.get("fwd_ms"), r.get("bwd_ms"), r.get("fwd_bwd_ms"),
                        k.get("n_regs"), k.get("n_spills"), k.get("shared_bytes"), k.get("theoretical_occupancy"),
                        r.get("error", "")])


if __name__ == "__main__":
    main()
