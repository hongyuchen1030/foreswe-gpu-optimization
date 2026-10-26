"""torch.profiler kernel breakdown for one E0-E5 configuration (training steps, steady state).

Writes a compact JSON summary (categories, Tensor-Core share, launches/step, top kernels) to the repo and
the Chrome trace to $FSW_ARTIFACT_DIR (never into git).

    python -m benchmarks.profile_config --exp E4 --steps 3
"""
import argparse
import os

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from foreswe_bench import data as fdata
from foreswe_bench import paths
from foreswe_bench.kernels import categorize, is_tensor_core_gemm
from foreswe_bench.model import VARIANTS, build_model, load_namespace
from foreswe_bench.precision import EXPERIMENTS, make_optimizer, set_matmul_flags
from foreswe_bench.runtime import env_info, forward_loss, setup_logging, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", required=True, choices=sorted(EXPERIMENTS))
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", default=str(paths.PROFILING_SUMMARY_DIR / "torch_profiler"))
    a = ap.parse_args()
    log = setup_logging()
    precision, impl = EXPERIMENTS[a.exp]
    set_matmul_flags(precision)
    ns = load_namespace("planB24", "cuda")
    ds = fdata.synthetic(ns)
    model = build_model(ns, "planB24", seed=0).train()
    opt = make_optimizer(model.parameters(), impl, VARIANTS["planB24"]["lr"], 1e-4)

    def step(i):
        ex = ds.example(int(ds.train_indices[i]))
        with record_function("phase/forward"):
            _, _, loss = forward_loss(model, ex, precision)
        loss.item()
        opt.zero_grad()
        with record_function("phase/backward"):
            loss.backward()
        with record_function("phase/optimizer"):
            opt.step()

    for i in range(a.warmup):
        step(i)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        for i in range(a.steps):
            step(a.warmup + i)
        torch.cuda.synchronize()
    os.makedirs(paths.ARTIFACT_DIR / "torch_traces", exist_ok=True)
    trace = paths.ARTIFACT_DIR / "torch_traces" / "trace_{}.json".format(a.exp)
    prof.export_chrome_trace(str(trace))

    gpu = [e for e in prof.events() if "CUDA" in str(getattr(e, "device_type", "")) and e.name
           and not e.name.startswith(("phase/", "Optimizer.", "ProfilerStep"))]
    total = sum(e.device_time_total for e in gpu) or 1.0
    cats, kern = {}, {}
    tc_time = gemm_time = 0.0
    for e in gpu:
        c = categorize(e.name)
        t = e.device_time_total
        cats.setdefault(c, [0.0, 0])
        cats[c][0] += t
        cats[c][1] += 1
        k = kern.setdefault(e.name[:140], [0.0, 0, c])
        k[0] += t
        k[1] += 1
        if c.startswith("GEMM"):
            gemm_time += t
            if is_tensor_core_gemm(e.name):
                tc_time += t
    summary = dict(exp=a.exp, precision=precision, optimizer=impl, steps=a.steps, env=env_info(),
                   gpu_kernel_ms_per_step=total / a.steps / 1e3, launches_per_step=len(gpu) / a.steps,
                   tensor_core_share_of_gemm_time=tc_time / gemm_time if gemm_time else 0.0,
                   tensor_core_share_of_gpu_time=tc_time / total,
                   categories=sorted(([c, v[0] / a.steps / 1e3, v[1] / a.steps, 100 * v[0] / total]
                                      for c, v in cats.items()), key=lambda r: -r[1]),
                   top_kernels=sorted(([n, v[0] / a.steps / 1e3, v[1] / a.steps, v[2]] for n, v in kern.items()),
                                      key=lambda r: -r[1])[:20],
                   chrome_trace=str(trace))
    write_json(summary, os.path.join(a.out, "profile_{}.json".format(a.exp)))
    log.info("%s: %.1f ms GPU/step, %.0f launches/step, Tensor-Core share of GEMM time %.1f%%", a.exp,
             summary["gpu_kernel_ms_per_step"], summary["launches_per_step"], 100 * summary["tensor_core_share_of_gemm_time"])


if __name__ == "__main__":
    main()
