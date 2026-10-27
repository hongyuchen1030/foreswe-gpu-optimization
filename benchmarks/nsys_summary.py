"""Summarise an nsys SQLite export of `bench_one --nvtx`: per training step GPU busy/idle, kernel
launches, Tensor-Core GEMM share, H2D/D2H bytes and time.

Step window = [train/prep push of step i, train/prep push of step i+1): NVTX ranges are CPU-side and
the harness synchronises at the end of each step, so the window contains all of that step's GPU work.

    python -m benchmarks.nsys_summary REPORT.sqlite --exp E1 --warmup 3 --out summary.json
"""
import argparse
import sqlite3

import numpy as np

from foreswe_bench.kernels import categorize, is_tensor_core_gemm
from foreswe_bench.runtime import write_json

COPY_KIND = {1: "H2D", 2: "D2H", 8: "D2D", 10: "P2P"}


def union_len(iv):
    tot, cs, ce = 0, None, None
    for s, e in sorted(iv):
        if ce is None or s > ce:
            if ce is not None:
                tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return tot + (ce - cs if ce is not None else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sqlite")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    con = sqlite3.connect(a.sqlite)
    q = lambda s: con.execute(s).fetchall()
    strings = dict(q("SELECT id, value FROM StringIds"))
    nvtx = [(s, e, t if t else strings.get(tid)) for s, e, t, tid in
            q("SELECT start, end, text, textId FROM NVTX_EVENTS WHERE end IS NOT NULL")]
    kern = [(s, e, strings.get(n, "?")) for s, e, n in q("SELECT start, end, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL")]
    copies = q("SELECT start, end, bytes, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY")
    sets = q("SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMSET")
    starts = sorted(s for s, e, n in nvtx if n == "train/prep")
    try:  # host-side CUDA runtime/driver API calls (launch submissions vs graph launches)
        api = [(s, strings.get(n, "?")) for s, n in q("SELECT start, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME")]
    except sqlite3.OperationalError:
        api = []
    steps = []
    for s, e in zip(starts[:-1], starts[1:]):
        ks = [(a_, b_, n) for a_, b_, n in kern if a_ >= s and b_ <= e]
        calls = [n for t, n in api if s <= t < e]
        cs = [(a_, b_, by, k) for a_, b_, by, k in copies if a_ >= s and b_ <= e]
        busy = union_len([(x, y) for x, y, _ in ks] + [(x, y) for x, y, _, _ in cs] +
                         [(x, y) for x, y in sets if x >= s and y <= e])
        gemm = sum(b_ - a_ for a_, b_, n in ks if categorize(n).startswith("GEMM"))
        tc = sum(b_ - a_ for a_, b_, n in ks if is_tensor_core_gemm(n))
        rec = dict(window_ms=(e - s) / 1e6, gpu_busy_ms=busy / 1e6, busy_pct=100 * busy / (e - s),
                   kernels=len(ks), gemm_ms=gemm / 1e6, tensor_core_gemm_ms=tc / 1e6,
                   triton_kernels=sum(1 for _, _, n in ks if n.startswith("triton_")),
                   api_kernel_launches=sum(1 for n in calls if "LaunchKernel" in n),
                   api_graph_launches=sum(1 for n in calls if "GraphLaunch" in n),
                   api_memcpy_calls=sum(1 for n in calls if "Memcpy" in n),
                   api_sync_calls=sum(1 for n in calls if "Synchronize" in n))
        for kind in ("H2D", "D2H", "D2D"):
            sel = [(a_, b_, by) for a_, b_, by, k in cs if COPY_KIND.get(k) == kind]
            rec[kind + "_bytes"] = int(sum(by for _, _, by in sel))
            rec[kind + "_ms"] = sum(b_ - a_ for a_, b_, _ in sel) / 1e6
        steps.append(rec)
    meas = steps[a.warmup:] or steps
    med = {k: float(np.median([r[k] for r in meas])) for k in meas[0]}
    med["idle_ms"] = med["window_ms"] - med["gpu_busy_ms"]
    med["tensor_core_share_of_gemm_time"] = med["tensor_core_gemm_ms"] / med["gemm_ms"] if med["gemm_ms"] else 0.0
    write_json(dict(exp=a.exp, sqlite=a.sqlite, measured_steps=len(meas), median=med, steps=steps), a.out)
    print(a.exp, {k: round(v, 3) for k, v in med.items()})


if __name__ == "__main__":
    main()
