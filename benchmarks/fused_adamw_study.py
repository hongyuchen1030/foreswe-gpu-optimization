"""Phase 2: AdamW foreach vs fused, isolated from everything else.

Correctness: two bit-identical FP32 models (A: foreach, B: fused). Each step the gradient is computed
once on A and copied bit-for-bit into B, so the only difference between A and B is the optimizer
kernel implementation. Reports one-step update/state differences and multi-step parameter drift.
Optional `--free-running` also lets each model compute its own gradients (feedback drift).

Performance: optimizer.step() alone, on fixed gradients, with CUDA events; kernel launches from
torch.profiler; optimizer-state and transient memory.

    python -m benchmarks.fused_adamw_study --steps 50 --out DIR
"""
import argparse
import copy
import os

import numpy as np
import torch

from foreswe_bench import data as fdata
from foreswe_bench import metrics, paths
from foreswe_bench.model import build_model, live_named_parameters, load_namespace
from foreswe_bench.precision import make_optimizer, set_matmul_flags
from foreswe_bench.runtime import env_info, forward_loss, setup_logging, stats, write_json

LR, WD = 5e-6, 1e-4


def live(model):
    return [p for _, p in live_named_parameters(model)]


def compare_params(pa, pb, a0=None):
    out = metrics.grad_errors(pb, pa)  # rel-L2 / cosine of B params vs A params
    out["max_abs"] = max((x - y).abs().max().item() for x, y in zip(pa, pb))
    out["bitwise_equal_fraction"] = float(np.average([metrics.bits_equal_fraction(x, y) for x, y in zip(pa, pb)],
                                                     weights=[x.numel() for x in pa]))
    if a0 is not None:  # compare the updates themselves
        ua = [x - x0 for x, x0 in zip(pa, a0)]
        ub = [y - x0 for y, x0 in zip(pb, a0)]
        u = metrics.grad_errors(ub, ua)
        out["update_rel_l2"], out["update_cosine"] = u["rel_l2"], u["cosine"]
        out["update_max_abs"] = max((x - y).abs().max().item() for x, y in zip(ua, ub))
        out["update_ref_l2"] = u["ref_l2"]
    return out


def compare_state(oa, ob, params_a, params_b):
    res = {}
    for key in ("exp_avg", "exp_avg_sq"):
        sa = [oa.state[p][key] for p in params_a if p in oa.state]
        sb = [ob.state[p][key] for p in params_b if p in ob.state]
        e = metrics.grad_errors(sb, sa)
        e["max_abs"] = max((x - y).abs().max().item() for x, y in zip(sa, sb))
        res[key] = e
    return res


def grad_step(model, ds, index):
    model.zero_grad(set_to_none=True)
    _, _, loss = forward_loss(model, ds.example(index), "fp32")
    loss.backward()
    return loss.item()


def time_optimizer(model, impl, iters=50, warmup=10):
    """Time opt.step() on fixed (already populated) gradients; returns stats + kernel count + memory."""
    opt = make_optimizer(model.parameters(), impl, LR, WD)
    base_alloc = torch.cuda.memory_allocated()
    opt.step()  # creates state
    torch.cuda.synchronize()
    state_gib = (torch.cuda.memory_allocated() - base_alloc) / 2**30
    times = []
    for i in range(warmup + iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        a.record()
        opt.step()
        b.record()
        torch.cuda.synchronize()
        if i >= warmup:
            times.append(a.elapsed_time(b))
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    opt.step()
    torch.cuda.synchronize()
    transient_gib = (torch.cuda.max_memory_allocated() - before) / 2**30
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        opt.step()
        torch.cuda.synchronize()
    kernels = [e for e in prof.events() if "CUDA" in str(getattr(e, "device_type", "")) and e.name and
               not e.name.startswith("Optimizer.")]
    names = {}
    for e in kernels:
        names[e.name[:90]] = names.get(e.name[:90], 0) + 1
    del opt
    torch.cuda.empty_cache()
    return dict(step_ms=stats(times), state_gib=state_gib, transient_peak_gib=transient_gib,
                kernel_launches=len(kernels), distinct_kernels=len(names),
                top_kernels=sorted(names.items(), key=lambda kv: -kv[1])[:8])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--free-running", action="store_true")
    ap.add_argument("--data", default="real", choices=["real", "synthetic"])
    ap.add_argument("--with-adam-algorithm-compare", action="store_true")
    ap.add_argument("--out", default=str(paths.RESULTS_DIR / "fused_adamw"))
    a = ap.parse_args()
    log = setup_logging()
    set_matmul_flags("fp32")
    ns = load_namespace("planB24", "cuda")
    ds = fdata.real(ns) if a.data == "real" else fdata.synthetic(ns)
    res = dict(env=env_info(), data=ds.meta, lr=LR, weight_decay=WD)

    model_a = build_model(ns, "planB24", seed=0).eval()  # eval: deterministic graph (no dropout)
    model_b = copy.deepcopy(model_a)
    pa, pb = live(model_a), live(model_b)
    opt_a = make_optimizer(model_a.parameters(), "foreach", LR, WD)
    opt_b = make_optimizer(model_b.parameters(), "fused", LR, WD)
    drift = []
    for s in range(a.steps):
        idx = int(ds.train_indices[s])
        loss = grad_step(model_a, ds, idx)
        for x, y in zip(pa, pb):
            y.grad = x.grad.detach().clone()
        a0 = [x.detach().clone() for x in pa] if s == 0 else None
        opt_a.step()
        opt_b.step()
        if s == 0:
            res["one_step"] = dict(params=compare_params(pa, pb, a0), state=compare_state(opt_a, opt_b, pa, pb),
                                   loss=loss)
            del a0
        if s in (0, 1, 4, 9, 19, 49, 99, 199) or s == a.steps - 1:
            d = compare_params(pa, pb)
            d["step"] = s + 1
            drift.append(d)
            log.info("step %d: param rel-L2 %.3e max-abs %.3e bitwise-equal %.4f", s + 1, d["rel_l2"], d["max_abs"],
                     d["bitwise_equal_fraction"])
    res["shared_gradient_drift"] = drift
    res["final_state"] = compare_state(opt_a, opt_b, pa, pb)

    if a.free_running:
        ma = build_model(ns, "planB24", seed=0).eval()
        mb = copy.deepcopy(ma)
        oa, ob = make_optimizer(ma.parameters(), "foreach", LR, WD), make_optimizer(mb.parameters(), "fused", LR, WD)
        fr = []
        for s in range(a.steps):
            idx = int(ds.train_indices[s])
            la = grad_step(ma, ds, idx)
            lb = grad_step(mb, ds, idx)
            oa.step()
            ob.step()
            if s in (0, 9, 49) or s == a.steps - 1:
                d = compare_params(live(ma), live(mb))
                d.update(step=s + 1, loss_a=la, loss_b=lb)
                fr.append(d)
        res["free_running_drift"] = fr
        del ma, mb, oa, ob

    del opt_a, opt_b, model_b
    torch.cuda.empty_cache()
    grad_step(model_a, ds, int(ds.train_indices[0]))  # fixed gradients for timing
    res["timing"] = {impl: time_optimizer(model_a, impl) for impl in ("foreach", "fused", "single_tensor")}
    for impl, t in res["timing"].items():
        log.info("%s: step %.2f ms, %d launches, state %.2f GiB, transient %.2f GiB", impl, t["step_ms"]["median"],
                 t["kernel_launches"], t["state_gib"], t["transient_peak_gib"])

    if a.with_adam_algorithm_compare:  # different ALGORITHM (L2 in grad vs decoupled decay): labelled separately
        m1 = build_model(ns, "planB24", seed=0).eval()
        m2 = copy.deepcopy(m1)
        grad_step(m1, ds, int(ds.train_indices[0]))
        for x, y in zip(live(m1), live(m2)):
            y.grad = x.grad.detach().clone()
        p0 = [x.detach().clone() for x in live(m1)]
        torch.optim.AdamW(m1.parameters(), lr=LR, weight_decay=WD, fused=True).step()
        torch.optim.Adam(m2.parameters(), lr=LR, weight_decay=WD, fused=True).step()
        res["algorithm_compare_AdamW_vs_Adam_fused"] = compare_params(live(m1), live(m2), p0)
    os.makedirs(a.out, exist_ok=True)
    write_json(res, os.path.join(a.out, "fused_adamw_study.json"))
    log.info("wrote %s", a.out)


if __name__ == "__main__":
    main()
