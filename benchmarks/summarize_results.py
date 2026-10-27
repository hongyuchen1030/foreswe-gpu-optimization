"""Assemble the final E0-E5 table (timing + memory + numerics + convergence) from the result files.

    python -m benchmarks.summarize_results [--timing-tag node1_synthetic]
Writes benchmark_results/final_table.{csv,json} and convergence/divergence_vs_E0.json.
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

from foreswe_bench import paths
from foreswe_bench.precision import EXPERIMENTS
from foreswe_bench.runtime import write_json

R = paths.RESULTS_DIR


def rel_l2(a, b):
    return float(np.linalg.norm((a - b).ravel()) / np.linalg.norm(b.ravel()))


def divergence(steps):
    base = R / "convergence" / "E0_{}steps_test_predictions.npz".format(steps)
    if not base.exists():
        return {}
    b = np.load(base)
    out = {}
    for e in EXPERIMENTS:
        f = R / "convergence" / "{}_{}steps_test_predictions.npz".format(e, steps)
        if not f.exists():
            continue
        d = np.load(f)
        prec = EXPERIMENTS[e][0]
        own = np.stack([d["{}_{}".format(prec, y)] for y in range(2015, 2020)])
        fp32 = np.stack([d["fp32_{}".format(y)] for y in range(2015, 2020)])
        ref = np.stack([b["fp32_{}".format(y)] for y in range(2015, 2020)])
        tgt = np.stack([b["target_{}".format(y)] for y in range(2015, 2020)]).reshape(own.shape)
        out[e] = dict(own_precision_vs_E0_rel_l2=rel_l2(own, ref), fp32_eval_vs_E0_rel_l2=rel_l2(fp32, ref),
                      own_precision_vs_E0_max_abs_mm=float(np.abs(own - ref).max()),
                      test_rmse_mm_own_precision=float(np.sqrt(np.mean((own - tgt) ** 2))))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timing-tag", default="node1_synthetic")
    a = ap.parse_args()
    timing = json.load(open(R / a.timing_tag / "matrix_summary.json"))["summary"]
    num = list(csv.DictReader(open(R / "numerics_precision" / "numerics_precision.csv")))
    div = {s: divergence(s) for s in ("300", "3297")}
    write_json(div, R / "convergence" / "divergence_vs_E0.json")
    rows = []
    for e, (prec, impl) in EXPERIMENTS.items():
        t = timing[e]
        sel = [r for r in num if r["weights"] == "planB" and r["precision"] == prec and r["reference"] == "fp32"]
        conv = {}
        for s in ("300", "3297"):
            f = R / "convergence" / "{}_{}steps_summary.json".format(e, s)
            if os.path.exists(f):
                conv[s] = json.load(open(f))
        rows.append(dict(
            exp=e, precision=prec, optimizer=impl,
            step_ms=round(t["step_ms"]["median"], 2), step_ms_iqr=round(t["step_ms"]["iqr"], 2),
            speedup=round(t["speedup_vs_E0"], 3),
            forward_ms=round(t["forward_ms"], 2), backward_ms=round(t["backward_ms"], 2),
            optimizer_ms=round(t["optimizer_ms"], 2), prep_ms=round(t["prep_ms"], 2),
            throughput_issue_dates_per_s=round(1e3 / t["step_ms"]["median"], 2),
            peak_allocated_gib=round(t["peak_allocated_gib"], 2), peak_reserved_gib=round(t["peak_reserved_gib"], 2),
            infer_forward_ms=round(t["infer_forward_ms"], 2),
            output_rel_l2_vs_fp32=float(np.median([float(r["out_rel_l2"]) for r in sel])) if sel else 0.0,
            grad_rel_l2_vs_fp32=float(np.median([float(r["grad_rel_l2"]) for r in sel])) if sel else 0.0,
            final_loss_300=conv.get("300", {}).get("final_loss"),
            mean_loss_last50_300=conv.get("300", {}).get("mean_loss_last50"),
            mean_loss_last50_epoch=conv.get("3297", {}).get("mean_loss_last50"),
            epoch_wall_s=conv.get("3297", {}).get("elapsed_s"),
            test_pred_rel_l2_vs_E0_300=div["300"].get(e, {}).get("own_precision_vs_E0_rel_l2"),
            test_pred_rel_l2_vs_E0_epoch=div["3297"].get(e, {}).get("own_precision_vs_E0_rel_l2"),
            gpu=t["env"]["gpu"]))
    write_json(dict(timing_tag=a.timing_tag, rows=rows), R / "final_table.json")
    with open(R / "final_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    print(json.dumps(div, indent=1))


if __name__ == "__main__":
    main()
