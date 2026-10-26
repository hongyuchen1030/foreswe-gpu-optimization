"""GPU busy/idle per NVTX phase from an nsys SQLite export (run on a compute node).

For each measured step (NVTX ranges emitted by profile_foreswe.py --nvtx), the
step window is [first range start, last range end]; the harness synchronises
after every step, so all GPU work of the step falls inside its window. GPU busy
time is the union of kernel + memcpy + memset intervals inside the window.
"""
import json
import sqlite3
import sys


def union_len(intervals):
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def main(db_path, out_path):
    con = sqlite3.connect(db_path)
    q = lambda sql: con.execute(sql).fetchall()
    strings = dict(q("SELECT id, value FROM StringIds"))
    nvtx = [(s, e, t if t else strings.get(tid)) for s, e, t, tid in
            q("SELECT start, end, text, textId FROM NVTX_EVENTS WHERE end IS NOT NULL")]
    gpu = q("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL")
    kernels = list(gpu)
    for tbl in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        gpu += q("SELECT start, end FROM {}".format(tbl))
    result = {}
    for phase, first, last in (("train", "train/prep", None), ("infer", "infer/prep", None)):
        # NVTX ranges are CPU-side: the GPU tail of a step runs after the last pop, until the
        # harness's cuda.synchronize(). So a step window is [this step's first push, next step's
        # first push); the final step of each phase has no successor and is dropped.
        starts = sorted(s for s, e, n in nvtx if n == first)
        steps = []
        for s, e in zip(starts[:-1], starts[1:]):
            inside = [(max(a, s), min(b, e)) for a, b in gpu if b > s and a < e]
            nk = sum(1 for a, b in kernels if a >= s and b <= e)
            busy = union_len(inside)
            steps.append(dict(window_ms=(e - s) / 1e6, gpu_busy_ms=busy / 1e6, gpu_idle_ms=(e - s - busy) / 1e6,
                              busy_pct=100.0 * busy / (e - s), kernels=nk))
        result[phase] = dict(all_steps=steps, measured_steps=steps[2:])  # nsys run: --warmup 2
        m = steps[2:]
        if m:
            result[phase]["median"] = {k: sorted(x[k] for x in m)[len(m) // 2] for k in m[0]}
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({p: result[p].get("median") for p in result}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
