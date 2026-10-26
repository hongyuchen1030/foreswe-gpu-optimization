"""Condense an `ncu --csv --page details` export into one row per profiled kernel launch."""
import csv
import json
import sys

WANT = {"Duration": "duration", "Compute (SM) Throughput": "sm_pct", "Memory Throughput": "mem_pct",
        "DRAM Throughput": "dram_pct", "Achieved Occupancy": "achieved_occ_pct", "Theoretical Occupancy": "theor_occ_pct",
        "Registers Per Thread": "regs", "Grid Size": "grid", "Block Size": "block", "Waves Per SM": "waves_per_sm"}


def main(path, out):
    with open(path) as f:
        lines = [l for l in f if l.startswith('"')]
    rows = list(csv.DictReader(lines))
    by_launch = {}
    for r in rows:
        key = (r["ID"], r["Kernel Name"])
        d = by_launch.setdefault(key, {"id": int(r["ID"]), "kernel": r["Kernel Name"][:120]})
        name = r["Metric Name"]
        if name in WANT:
            d[WANT[name]] = r["Metric Value"]
            if name == "Duration":
                d["duration_unit"] = r["Metric Unit"]
    recs = sorted(by_launch.values(), key=lambda d: d["id"])
    with open(out, "w") as f:
        json.dump(recs, f, indent=1)
    for d in recs:
        print("{id:>4} {dur:>9} {u:<5} SM {sm:>6}% MEM {mem:>6}% DRAM {dram:>6}% occ {occ:>6}/{tocc:<6} grid {grid:<14} {k}".format(
            id=d["id"], dur=d.get("duration", "?"), u=d.get("duration_unit", ""), sm=d.get("sm_pct", "?"), mem=d.get("mem_pct", "?"),
            dram=d.get("dram_pct", "?"), occ=d.get("achieved_occ_pct", "?"), tocc=d.get("theor_occ_pct", "?"), grid=d.get("grid", "?"), k=d["kernel"][:70]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
