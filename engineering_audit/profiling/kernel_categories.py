"""Bucket GPU kernel/memcpy time from a torch.profiler chrome trace into categories (per profiled step)."""
import json
import re
import sys

CATS = [
    ("GEMM fp32 (sgemm/gemv)", r"sgemm|gemv|gemm"),
    ("AdamW foreach (multi_tensor_apply)", r"multi_tensor_apply"),
    ("index gather (model_inputs[:, loc, :])", r"index_elementwise_kernel|indexing_backward|index_put"),
    ("softmax fwd/bwd", r"softmax"),
    ("layer_norm fwd/bwd", r"layer_norm|LayerNorm"),
    ("reductions (sum/mean)", r"reduce_kernel"),
    ("concat/stack", r"CatArrayBatched|cat_"),
    ("copy / contiguous / to()", r"direct_copy|copy_kernel"),
    ("other elementwise", r"elementwise|Functor|gelu|Gelu|threshold"),
    ("memcpy/memset", r"Memcpy|Memset|memcpy|memset"),
]


def main(trace, steps):
    with open(trace) as f:
        ev = json.load(f)["traceEvents"]
    gpu = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e]
    tot = sum(e["dur"] for e in gpu)
    acc = {c: [0.0, 0] for c, _ in CATS}
    acc["uncategorized"] = [0.0, 0]
    for e in gpu:
        name = e["name"]
        for c, rx in CATS:
            if re.search(rx, name):
                break
        else:
            c = "uncategorized"
        acc[c][0] += e["dur"]
        acc[c][1] += 1
    rows = sorted(((c, t / steps / 1e3, n / steps, 100 * t / tot) for c, (t, n) in acc.items() if n), key=lambda r: -r[1])
    out = dict(trace=trace, steps=steps, gpu_ms_per_step=tot / steps / 1e3, launches_per_step=len(gpu) / steps,
               categories=[dict(category=c, ms_per_step=ms, launches_per_step=n, pct=p) for c, ms, n, p in rows])
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
