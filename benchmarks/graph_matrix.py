"""G-matrix driver: each (repeat, config) in a fresh process, rotated order.

Repeat 0 uses an empty Inductor cache (cold compile); later repeats reuse it (warm compile).

    python -m benchmarks.graph_matrix --tag nodeA --cfgs G0 G0c G1 G1L G2 G2L G3f G3 G4 --repeats 3
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys

import numpy as np

from foreswe_bench import paths
from foreswe_bench.runtime import setup_logging, write_json

OUT_ROOT = paths.REPO / "engineering_audit" / "optimization" / "cuda_graphs" / "benchmark_results"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cfgs", nargs="+", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--data", default="synthetic")
    a = ap.parse_args()
    log = setup_logging()
    out = OUT_ROOT / a.tag
    os.makedirs(out, exist_ok=True)
    cache_root = paths.SCRATCH_ROOT / "inductor_cache" / a.tag
    shutil.rmtree(cache_root, ignore_errors=True)
    for r in range(a.repeats):
        order = a.cfgs[r % len(a.cfgs):] + a.cfgs[: r % len(a.cfgs)]
        for c in order:
            env = dict(os.environ)
            env.pop("CUDA_LAUNCH_BLOCKING", None)
            env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root / c)   # per config: cold in repeat 0, warm after
            cmd = [sys.executable, "-m", "benchmarks.graph_bench", "--cfg", c, "--data", a.data, "--warmup",
                   str(a.warmup), "--iters", str(a.iters), "--out", str(out / "{}_r{}.json".format(c, r))]
            if r == 0:
                cmd.append("--profile")
            log.info("repeat %d %s", r, c)
            p = subprocess.run(cmd, env=env)
            if p.returncode != 0:
                log.error("%s repeat %d failed (exit %d)", c, r, p.returncode)

    summ, rows = {}, []
    for c in a.cfgs:
        runs = []
        for r in range(a.repeats):
            f = out / "{}_r{}.json".format(c, r)
            if f.exists():
                runs.append(json.load(open(f)))
        if not runs:
            summ[c] = dict(failed=True)
            continue
        med = lambda path: float(np.median([eval("run" + path, {"run": run}) for run in runs]))  # noqa: E731
        s = dict(cfg=c, n_runs=len(runs),
                 e2e_ms=med('["end_to_end"]["step_wall_ms"]["median"]'),
                 e2e_ms_repeat_medians=[r["end_to_end"]["step_wall_ms"]["median"] for r in runs],
                 e2e_iqr_ms=med('["end_to_end"]["step_wall_ms"]["iqr"]'),
                 prep_ms=med('["end_to_end"]["prep_ms"]["median"]'),
                 step_after_prep_ms=med('["end_to_end"]["step_after_prep_ms"]["median"]'),
                 gpu_region_ms=med('["gpu_region"]["event_ms"]["median"]'),
                 gpu_region_wall_ms=med('["gpu_region"]["wall_ms"]["median"]'),
                 infer_ms=med('["inference"]["forward_ms"]["median"]'),
                 peak_alloc_gib=max(r["end_to_end"]["peak_allocated_gib"] for r in runs),
                 peak_reserved_gib=max(r["end_to_end"]["peak_reserved_gib"] for r in runs),
                 setup_s_cold=runs[0]["setup_s"], setup_s_warm=[r["setup_s"] for r in runs[1:]],
                 phases_ms={k: v["median"] for k, v in runs[0]["phases_ms"].items()},
                 profiler_counts=runs[0].get("profiler_counts"), dynamo_counters=runs[0]["dynamo_counters"],
                 gpu=runs[0]["env"]["gpu"], host=runs[0]["env"]["host"])
        summ[c] = s
    base = summ.get("G0")
    for c, s in summ.items():
        if s.get("failed"):
            rows.append(dict(cfg=c, failed=True))
            continue
        if base and not base.get("failed"):
            s["speedup_e2e_vs_G0"] = base["e2e_ms"] / s["e2e_ms"]
            s["speedup_gpu_region_vs_G0"] = base["gpu_region_ms"] / s["gpu_region_ms"]
        rows.append({k: s.get(k) for k in ("cfg", "e2e_ms", "e2e_iqr_ms", "speedup_e2e_vs_G0", "prep_ms",
                                            "step_after_prep_ms", "gpu_region_ms", "speedup_gpu_region_vs_G0",
                                            "infer_ms", "peak_alloc_gib", "peak_reserved_gib", "setup_s_cold", "gpu")})
    write_json(dict(tag=a.tag, summary=summ), out / "matrix_summary.json")
    keys = sorted({k for r in rows for k in r})
    with open(out / "matrix_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        log.info("%s", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
