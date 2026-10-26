"""Run the E0-E5 timing matrix: every (repeat, experiment) in its own process, order rotated per repeat.

    python -m benchmarks.bench_matrix --tag node1_synthetic --repeats 3 [--exps E0 E1 ...] [--data real]

Writes <results>/<tag>/{E*_r*.json, matrix_summary.json, matrix_summary.csv}.
"""
import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np

from foreswe_bench import paths
from foreswe_bench.precision import EXPERIMENTS
from foreswe_bench.runtime import setup_logging, stats, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", required=True)
    p.add_argument("--exps", nargs="+", default=sorted(EXPERIMENTS))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--data", default="synthetic", choices=["synthetic", "real"])
    p.add_argument("--variant", default="planB24")
    p.add_argument("--out-root", default=str(paths.RESULTS_DIR))
    a = p.parse_args()
    log = setup_logging()
    out_dir = os.path.join(a.out_root, a.tag)
    os.makedirs(out_dir, exist_ok=True)
    env = dict(os.environ)
    env.pop("CUDA_LAUNCH_BLOCKING", None)

    for r in range(a.repeats):
        order = a.exps[r % len(a.exps):] + a.exps[: r % len(a.exps)]  # rotate to spread drift
        for e in order:
            path = os.path.join(out_dir, "{}_r{}.json".format(e, r))
            cmd = [sys.executable, "-m", "benchmarks.bench_one", "--exp", e, "--variant", a.variant,
                   "--data", a.data, "--warmup", str(a.warmup), "--iters", str(a.iters), "--infer",
                   "--out", path]
            log.info("repeat %d: %s", r, " ".join(cmd[2:]))
            subprocess.run(cmd, check=True, env=env)

    summary, rows = {}, []
    for e in a.exps:
        runs = [json.load(open(os.path.join(out_dir, "{}_r{}.json".format(e, r)))) for r in range(a.repeats)]
        pooled = [x for run in runs for x in run["raw_step_wall_ms"]]
        rep_meds = [run["train"]["step_wall"]["median"] for run in runs]
        t = runs[0]["train"]
        s = dict(exp=e, precision=runs[0]["precision"], optimizer=runs[0]["optimizer"],
                 step_ms=stats(pooled), step_ms_repeat_medians=rep_meds,
                 forward_ms=float(np.median([r["train"]["forward"]["median"] for r in runs])),
                 backward_ms=float(np.median([r["train"]["backward"]["median"] for r in runs])),
                 optimizer_ms=float(np.median([r["train"]["optimizer"]["median"] for r in runs])),
                 prep_ms=float(np.median([r["train"]["prep"]["median"] for r in runs])),
                 peak_allocated_gib=max(r["train"]["peak_allocated_gib"] for r in runs),
                 peak_reserved_gib=max(r["train"]["peak_reserved_gib"] for r in runs),
                 infer_forward_ms=float(np.median([r["infer"]["forward"]["median"] for r in runs])),
                 infer_step_ms=float(np.median([r["infer"]["step_wall"]["median"] for r in runs])),
                 infer_peak_allocated_gib=max(r["infer"]["peak_allocated_gib"] for r in runs),
                 env=dict((k, runs[0]["env"].get(k)) for k in ("host", "gpu", "gpu_uuid", "torch", "cuda_runtime",
                                                                 "driver", "slurm_job_id", "git_commit")),
                 hosts=sorted({r["env"]["host"] for r in runs}))
        summary[e] = s
    base = summary.get("E0")
    for e, s in summary.items():
        if base:
            s["speedup_vs_E0"] = base["step_ms"]["median"] / s["step_ms"]["median"]
            s["speedup_vs_E0_repeat_range"] = [
                min(b / x for b, x in zip(base["step_ms_repeat_medians"], s["step_ms_repeat_medians"])),
                max(b / x for b, x in zip(base["step_ms_repeat_medians"], s["step_ms_repeat_medians"]))]
            s["infer_speedup_vs_E0"] = base["infer_forward_ms"] / s["infer_forward_ms"]
        rows.append(dict(exp=e, precision=s["precision"], optimizer=s["optimizer"],
                         step_ms_median=s["step_ms"]["median"], step_ms_iqr=s["step_ms"]["iqr"],
                         speedup_vs_E0=s.get("speedup_vs_E0"), forward_ms=s["forward_ms"],
                         backward_ms=s["backward_ms"], optimizer_ms=s["optimizer_ms"], prep_ms=s["prep_ms"],
                         peak_allocated_gib=s["peak_allocated_gib"], peak_reserved_gib=s["peak_reserved_gib"],
                         throughput_issue_dates_per_s=1e3 / s["step_ms"]["median"],
                         infer_forward_ms=s["infer_forward_ms"], infer_speedup_vs_E0=s.get("infer_speedup_vs_E0"),
                         gpu=s["env"]["gpu"], hosts=";".join(s["hosts"])))
    write_json(dict(tag=a.tag, data=a.data, repeats=a.repeats, warmup=a.warmup, iters=a.iters, summary=summary),
               os.path.join(out_dir, "matrix_summary.json"))
    with open(os.path.join(out_dir, "matrix_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        log.info("%s %-4s %-7s step %.2f ms x%.2f fwd %.2f bwd %.2f opt %.2f peak %.2f GiB", r["exp"], r["precision"],
                 r["optimizer"], r["step_ms_median"], r["speedup_vs_E0"] or 0, r["forward_ms"], r["backward_ms"],
                 r["optimizer_ms"], r["peak_allocated_gib"])


if __name__ == "__main__":
    main()
