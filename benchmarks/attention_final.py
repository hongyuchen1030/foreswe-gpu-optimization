"""Final comparison: eager / Inductor / original Triton / v2 fast / v2 accurate, plus the regression table.

    python -m benchmarks.attention_final
Writes benchmark_results/final/{final_comparison.json, final_comparison.csv, regression_table.csv}.
Per-kernel registers / spills / shared memory: benchmarks.attention_resources (launch_resources.json).
"""
import csv
import os

import torch

from benchmarks.attention_study import capture_real_qkv
from benchmarks.attention_tiling import timings
from foreswe_bench import attn_check
from foreswe_bench import data as fdata
from foreswe_bench import geo_attention as v1
from foreswe_bench import geo_attention_v2 as v2
from foreswe_bench import paths
from foreswe_bench.geo_attention import eager_geo_attention
from foreswe_bench.model import load_namespace
from foreswe_bench.runtime import env_info, setup_logging, write_json

OUT = paths.REPO / "engineering_audit" / "optimization" / "fused_attention" / "benchmark_results" / "final"
CASES = ("random", "near_one_hot", "uniform", "moderate", "wide_signed", "large_alpha")


def v2_with(preset):
    def fn(*a):
        v2.use_preset(preset)
        return v2.fused_geo_attention_v2(*a)
    return fn


def main():
    log = setup_logging()
    ns = load_namespace("planB24", "cuda")
    real = capture_real_qkv(ns, fdata.real(ns))
    impls = {"eager": eager_geo_attention, "inductor": torch.compile(eager_geo_attention),
             "triton_original": v1.fused_geo_attention, "triton_v2_fast": v2_with("fast_experimental"),
             "triton_v2_accurate": v2_with("accurate_experimental"),
             "triton_v2_accurate_x3": v2_with("accurate_x3_experimental")}
    res = dict(env=env_info(), tiles=v2.TUNED_TILES, presets=v2.PRESETS, impls={})
    rows = []
    for name, fn in impls.items():
        r = dict(perf={p: timings(fn, real[0], p) for p in ("tf32", "fp32")})
        r["errors"] = {"L{}_{}".format(L, p): attn_check.evaluate(fn, inp, p) for L, inp in real.items()
                       for p in ("fp32", "tf32")}
        res["impls"][name] = r
        for p in ("tf32", "fp32"):
            e0 = r["errors"]["L0_" + p]
            rows.append(dict(impl=name, prec=p, fwd_ms=r["perf"][p]["fwd_ms"], bwd_ms=r["perf"][p]["bwd_ms"],
                             fwd_bwd_ms=r["perf"][p]["fwd_bwd_ms"], peak_extra_mib=r["perf"][p]["fwd_bwd_peak_extra_mib"],
                             kernels_fwd_bwd=r["perf"][p]["kernels_fwd_bwd"],
                             L0_dQ_abs=e0["dQ"]["candidate"]["max_abs"], L0_dK_abs=e0["dK"]["candidate"]["max_abs"],
                             L0_dV_abs=e0["dV"]["candidate"]["max_abs"],
                             L0_dQ_x_eager=e0["dQ"]["ratio_to_eager"], L0_dK_x_eager=e0["dK"]["ratio_to_eager"],
                             L0_all_pass=all(v["passed"] for v in e0.values()),
                             L7_all_pass=all(v["passed"] for v in r["errors"]["L7_" + p].values())))
        log.info("%s: %s", name, rows[-2])
    # regression table: every synthetic case x v2 preset x precision
    table = []
    for preset in ("fast_experimental", "accurate_experimental", "accurate_x3_experimental"):
        for case in CASES:
            for p in ("fp32", "tf32"):
                ev = attn_check.evaluate(v2_with(preset), attn_check.case_inputs(case), p)
                for k, v in ev.items():
                    table.append(dict(preset=preset, case=case, prec=p, quantity=k,
                                      abs_err=v["candidate"]["max_abs"], eager_abs_err=v["eager"]["max_abs"],
                                      ratio_to_eager=v["ratio_to_eager"], rel_l2=v["candidate"]["rel_l2"],
                                      ref_norm=v["candidate"]["ref_norm"], cosine=v["candidate"]["cosine"],
                                      passed=v["passed"]))
    v2.use_preset("accurate_experimental")
    os.makedirs(OUT, exist_ok=True)
    write_json(res, OUT / "final_comparison.json")
    for fname, data in (("final_comparison.csv", rows), ("regression_table.csv", table)):
        with open(OUT / fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    for preset in ("fast_experimental", "accurate_experimental", "accurate_x3_experimental"):
        for p in ("fp32", "tf32"):
            sel = [t for t in table if t["preset"] == preset and t["prec"] == p]
            log.info("regression %s %s: %d/%d pass", preset, p, sum(t["passed"] for t in sel), len(sel))


if __name__ == "__main__":
    main()
