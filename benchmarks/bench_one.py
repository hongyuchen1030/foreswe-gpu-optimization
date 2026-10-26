"""Time one E0-E5 configuration in a fresh process (training step regions + optional inference).

    python -m benchmarks.bench_one --exp E1 --out result.json [--data real] [--infer]
"""
import argparse
import os
import sys
import time

import torch

from foreswe_bench import data as fdata
from foreswe_bench.model import VARIANTS, build_model, load_namespace
from foreswe_bench.precision import EXPERIMENTS, make_optimizer, set_matmul_flags
from foreswe_bench.runtime import (Nvtx, env_info, setup_logging, stats, timed_infer_step, timed_train_step,
                                   write_json, TRAIN_REGIONS)


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp", required=True, choices=sorted(EXPERIMENTS))
    p.add_argument("--variant", default="planB24", choices=sorted(VARIANTS))
    p.add_argument("--data", default="synthetic", choices=["synthetic", "real"])
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--infer", action="store_true", help="also time inference (forward only)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nvtx", action="store_true")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    a = parse()
    log = setup_logging()
    if os.environ.get("CUDA_LAUNCH_BLOCKING", "0") not in ("", "0"):
        sys.exit("CUDA_LAUNCH_BLOCKING must be 0/unset for performance runs")
    assert torch.cuda.is_available(), "GPU required"
    precision, impl = EXPERIMENTS[a.exp]
    set_matmul_flags(precision)
    Nvtx.enabled = a.nvtx
    cfg = VARIANTS[a.variant]

    ns = load_namespace(a.variant, "cuda")
    ds = fdata.synthetic(ns, a.variant) if a.data == "synthetic" else fdata.real(ns, a.variant)
    t0 = time.perf_counter()
    model = build_model(ns, a.variant, seed=a.seed)
    torch.cuda.synchronize()
    init_s = time.perf_counter() - t0
    mem_model = torch.cuda.memory_allocated()
    opt = make_optimizer(model.parameters(), impl, cfg["lr"], cfg["weight_decay"])
    model.train()

    idx = ds.train_indices[: a.warmup + a.iters]
    recs, mem_after_first = [], None
    for i, ix in enumerate(idx):
        if i == a.warmup:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        r = timed_train_step(model, opt, ds, ix, precision)
        if i == 0:
            mem_after_first = torch.cuda.memory_allocated()
        if i >= a.warmup:
            recs.append(r)
    train = {k: stats([r[k] for r in recs]) for k in TRAIN_REGIONS + ("step_gpu", "step_wall")}
    train["loss_first_measured"] = recs[0]["loss"]
    train["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    train["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    train["allocated_after_first_step_gib"] = mem_after_first / 2**30
    train["model_params_gib"] = mem_model / 2**30
    train["optimizer_state_gib"] = sum(t.numel() * t.element_size() for s in opt.state.values()
                                       for t in s.values() if torch.is_tensor(t)) / 2**30
    med = train["step_wall"]["median"]
    train["throughput_issue_dates_per_s"] = 1e3 / med
    train["throughput_station_samples_per_s"] = 512 * 1e3 / med
    out = dict(exp=a.exp, precision=precision, optimizer=impl, variant=a.variant, data=a.data, seed=a.seed,
               warmup=a.warmup, iters=a.iters, env=env_info(), model_init_s=init_s, train=train,
               raw_step_wall_ms=[r["step_wall"] for r in recs])

    if a.infer:
        model.eval()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        issue = ds.test_issue[2017]
        irecs = [timed_infer_step(model, ds, issue, precision) for _ in range(a.warmup + a.iters)][a.warmup:]
        out["infer"] = {k: stats([r[k] for r in irecs]) for k in ("prep", "forward", "d2h", "step_wall")}
        out["infer"]["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        out["infer"]["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    write_json(out, a.out)
    log.info("%s %s/%s step %.2f ms (fwd %.2f bwd %.2f opt %.2f) peak %.2f GiB", a.exp, precision, impl, med,
             train["forward"]["median"], train["backward"]["median"], train["optimizer"]["median"],
             train["peak_allocated_gib"])


if __name__ == "__main__":
    main()
