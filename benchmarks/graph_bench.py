"""Benchmark one G-configuration (E4 + torch.compile / CUDA Graphs) in a fresh process.

Two timing scopes (both reported):
  A  gpu_region : the accelerated region on already-staged inputs (no host prep, no host sync inside)
  B  end_to_end : production data prep (ds.example) + copy into static buffers + step + loss.item()

    python -m benchmarks.graph_bench --cfg G3 --out r.json [--profile] [--data real]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

from foreswe_bench import data as fdata
from foreswe_bench.graphs import CONFIGS, dynamo_counters, make_step
from foreswe_bench.model import build_model, load_namespace
from foreswe_bench.precision import set_matmul_flags
from foreswe_bench.runtime import Nvtx, env_info, setup_logging, stats, write_json

API_LAUNCH = ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC", "cudaGraphLaunch", "cuGraphLaunch")


def profile_counts(step, ds, idx):
    """torch.profiler over 3 end-to-end steps: host launch API calls vs device kernels observed."""
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i in idx:
            step.train_step(ds.example(int(i)))
        torch.cuda.synchronize()
    api, kern, triton = {}, 0, 0
    for e in prof.events():
        dt = str(getattr(e, "device_type", ""))
        if "CUDA" in dt:
            if e.name and not e.name.startswith(("train/", "Optimizer.", "ProfilerStep")):
                kern += 1
                triton += e.name.startswith("triton_")
        elif e.name in API_LAUNCH or e.name.startswith(("cudaMemcpy", "cudaGraph")):
            api[e.name] = api.get(e.name, 0) + 1
    n = len(idx)
    return dict(steps=n, host_api_calls_per_step={k: v / n for k, v in api.items()},
                host_kernel_launch_calls_per_step=sum(v for k, v in api.items() if "Launch" in k and "Graph" not in k) / n,
                graph_launches_per_step=sum(v for k, v in api.items() if "GraphLaunch" in k) / n,
                device_kernels_seen_by_profiler_per_step=kern / n, triton_kernels_per_step=triton / n,
                note="kernels executed inside CUDA graphs may be invisible to torch.profiler; nsys --cuda-graph-trace=node is authoritative")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cfg", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--data", default="synthetic", choices=["synthetic", "real"])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--nvtx", action="store_true")
    ap.add_argument("--attn", default="eager", choices=["eager", "fused"],
                    help="geo-attention core: production eager or fused Triton prototype")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    log = setup_logging()
    if os.environ.get("CUDA_LAUNCH_BLOCKING", "0") not in ("", "0"):
        sys.exit("CUDA_LAUNCH_BLOCKING must be unset for performance runs")
    set_matmul_flags("tf32")                     # E4 precision
    Nvtx.enabled = a.nvtx
    ns = load_namespace("planB24", "cuda")
    ds = fdata.synthetic(ns) if a.data == "synthetic" else fdata.real(ns)
    model = build_model(ns, "planB24", seed=0).train()
    if a.attn == "fused":
        from foreswe_bench.geo_attention import patch_model
        patch_model(model, "fused")
    mem0 = torch.cuda.memory_allocated()
    seq = ds.train_indices[: 1 + a.warmup + a.iters]

    with dynamo_counters() as dyn:
        step = make_step(a.cfg, model)
        ex0 = ds.example(int(seq[0]))
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        step.setup(ex0)
        torch.cuda.synchronize()
        setup_s = time.perf_counter() - t0
    res = dict(cfg=a.cfg, attn=a.attn, config=CONFIGS[a.cfg], data=a.data, env=env_info(), warmup=a.warmup, iters=a.iters,
               setup_s=setup_s, setup_inner_s=getattr(step, "setup_s", None), dynamo_counters=dyn,
               inductor_cache_dir=os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
               mem_after_setup_gib=torch.cuda.memory_allocated() / 2**30,
               peak_during_setup_gib=torch.cuda.max_memory_allocated() / 2**30, model_gib=mem0 / 2**30)

    # ---- scope A: accelerated region on staged inputs
    torch.cuda.synchronize()
    ra, wa = [], []
    for i in range(a.warmup + a.iters):
        if i == a.warmup:
            torch.cuda.reset_peak_memory_stats()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        w0 = time.perf_counter()
        e0.record()
        step.run_staged()
        e1.record()
        torch.cuda.synchronize()
        if i >= a.warmup:
            ra.append(e0.elapsed_time(e1))
            wa.append((time.perf_counter() - w0) * 1e3)
    res["gpu_region"] = dict(event_ms=stats(ra), wall_ms=stats(wa),
                             peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                             peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)

    # ---- phase split where the variant allows it
    ph = [step.phases(step.static.buf) for _ in range(a.iters // 2)]
    res["phases_ms"] = {k: stats([p[k] for p in ph]) for k in ph[0]}

    # ---- scope B: end to end
    torch.cuda.reset_peak_memory_stats()
    rb, prep, rest, losses = [], [], [], []
    for i, ix in enumerate(seq[1:]):
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        torch.cuda.synchronize()
        w0 = time.perf_counter()
        ev[0].record()
        with Nvtx("train/prep"):
            ex = ds.example(int(ix))
        ev[1].record()
        with Nvtx("train/step"):
            losses.append(step.train_step(ex))
        ev[2].record()
        torch.cuda.synchronize()
        if i >= a.warmup:
            rb.append((time.perf_counter() - w0) * 1e3)
            prep.append(ev[0].elapsed_time(ev[1]))
            rest.append(ev[1].elapsed_time(ev[2]))
    res["end_to_end"] = dict(step_wall_ms=stats(rb), prep_ms=stats(prep), step_after_prep_ms=stats(rest),
                             throughput_issue_dates_per_s=1e3 / float(np.median(rb)),
                             peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                             peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                             losses_first_last=[losses[0], losses[-1]], loss_finite=bool(np.all(np.isfinite(losses))))

    # ---- inference (forward only, eval mode; training state stays resident)
    exi = ds.example(int(ds.test_issue[2017]))
    t0 = time.perf_counter()
    step.infer(exi)
    torch.cuda.synchronize()
    infer_setup = time.perf_counter() - t0
    ri = []
    for i in range(a.warmup + a.iters):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        e0.record()
        step.infer(exi)
        e1.record()
        torch.cuda.synchronize()
        if i >= a.warmup:
            ri.append(e0.elapsed_time(e1))
    res["inference"] = dict(forward_ms=stats(ri), first_call_s=infer_setup)

    if a.profile:
        res["profiler_counts"] = profile_counts(step, ds, seq[-3:])
    write_json(res, a.out)
    log.info("%s setup %.1fs | A gpu-region %.2f ms | B end-to-end %.2f ms (prep %.2f) | infer %.2f ms | peak %.2f GiB",
             a.cfg, setup_s, res["gpu_region"]["event_ms"]["median"], res["end_to_end"]["step_wall_ms"]["median"],
             res["end_to_end"]["prep_ms"]["median"], res["inference"]["forward_ms"]["median"],
             res["end_to_end"]["peak_allocated_gib"])


if __name__ == "__main__":
    main()
