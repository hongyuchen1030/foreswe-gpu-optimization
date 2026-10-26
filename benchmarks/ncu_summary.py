"""Condense `ncu --csv --page raw` output (one row per profiled kernel) into a small JSON table with
duration, SM / DRAM throughput and Tensor-pipe utilisation.

    python -m benchmarks.ncu_summary ncu_E1.csv --exp E1 --out summary.json
"""
import argparse
import csv

from foreswe_bench.kernels import is_tensor_core_gemm
from foreswe_bench.runtime import write_json

METRICS = {
    "gpu__time_duration.sum": "duration_ns",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_throughput_pct",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_throughput_pct",
    "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active": "tensor_pipe_active_pct",
    "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active": "fma_pipe_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "achieved_occupancy_pct",
}


def num(x):
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.csv) as f:
        lines = [l for l in f if l.startswith('"')]
    reader = csv.reader(lines)
    header = next(reader)
    rows = list(reader)
    rows = [r for r in rows if r and r[0] not in ("", "ID") and len(r) == len(header)]
    col = {h: i for i, h in enumerate(header)}
    out = []
    for r in rows:
        rec = dict(id=r[col["ID"]], kernel=r[col["Kernel Name"]][:120])
        for m, k in METRICS.items():
            if m in col:
                rec[k] = num(r[col[m]])
        rec["tensor_core_kernel_by_name"] = is_tensor_core_gemm(rec["kernel"])
        out.append(rec)
    write_json(dict(exp=a.exp, kernels=out), a.out)
    for rec in out:
        print(a.exp, rec)


if __name__ == "__main__":
    main()
