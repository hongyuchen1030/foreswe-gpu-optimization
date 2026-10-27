"""Extract one measured training step's GPU activity and host launch calls from an nsys SQLite export
into a small CSV (git-safe) for before/after timeline figures.

    python -m benchmarks.nsys_timeline G0.sqlite --step 5 --out timeline_G0.csv
"""
import argparse
import csv
import sqlite3

from foreswe_bench.kernels import categorize


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sqlite")
    ap.add_argument("--step", type=int, default=5, help="index of the end-to-end step window")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    con = sqlite3.connect(a.sqlite)
    q = lambda s: con.execute(s).fetchall()
    strings = dict(q("SELECT id, value FROM StringIds"))
    nvtx = [(s, e, t if t else strings.get(tid)) for s, e, t, tid in
            q("SELECT start, end, text, textId FROM NVTX_EVENTS WHERE end IS NOT NULL")]
    starts = sorted(s for s, e, n in nvtx if n == "train/prep")
    s0, s1 = starts[a.step], starts[a.step + 1]
    rows = []
    for s, e, n in q("SELECT start, end, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        if s >= s0 and e <= s1:
            name = strings.get(n, "?")
            rows.append(("gpu_kernel", (s - s0) / 1e3, (e - s) / 1e3, categorize(name), name[:80]))
    for s, e in q("SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY"):
        if s >= s0 and e <= s1:
            rows.append(("gpu_memcpy", (s - s0) / 1e3, (e - s) / 1e3, "memcpy", ""))
    for s, e, n in q("SELECT start, end, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
        name = strings.get(n, "?")
        if s0 <= s < s1 and ("Launch" in name or "Memcpy" in name or "Synchronize" in name):
            rows.append(("host_api", (s - s0) / 1e3, (e - s) / 1e3, name.split("_v")[0], ""))
    for s, e, n in nvtx:
        if s0 <= s < s1 and n and n.startswith("train/"):
            rows.append(("nvtx", (s - s0) / 1e3, (e - s) / 1e3, n, ""))
    rows.sort(key=lambda r: (r[0], r[1]))
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track", "start_us", "dur_us", "category", "name"])
        w.writerows(rows)
    print(a.out, len(rows), "rows, window", (s1 - s0) / 1e6, "ms")


if __name__ == "__main__":
    main()
