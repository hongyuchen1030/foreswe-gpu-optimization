"""Phase 4: numerical and training correctness of G-configurations vs E4 eager (G0).

Deterministic comparisons use model.eval() (dropout off) on real ForeSWE batches, starting every
configuration from the identical seed-0 weights and a fresh AdamW state:
  1. forward output on one batch (inference path)
  2. one training step: loss, gradients (live params), updated parameters, AdamW moments
  3. 20-step trajectory on 20 different batches (inputs must be refreshed each step)
Graph-specific safety checks:
  - capture warm-up leaves weights bitwise unchanged and AdamW state zero
  - train-mode replays draw fresh dropout masks (no frozen RNG)
  - memory does not grow over 200 replays
Acceptance (fixed before running; see COMPILE_AND_CUDAGRAPH_REPORT.md):
  - manual graphs (same kernels as G0c): bitwise identical to G0c
  - compiled variants: output / gradient error vs the FP32 reference no more than 10 % above G0's,
    and rel-L2 vs G0 below the TF32-vs-FP32 error itself (3.9e-4 output, 2.2e-4 gradient)
  - no NaN/Inf; parameters actually change

    python -m benchmarks.graph_correctness --cfgs G0c G1 G2 G3f G3 G4
"""
import argparse
import gc
import os

import numpy as np
import torch

from foreswe_bench import data as fdata
from foreswe_bench import metrics
from foreswe_bench.graphs import CONFIGS, GraphStep, forward_loss, make_step
from foreswe_bench.model import build_model, is_dead, load_namespace
from foreswe_bench.precision import matmul_flags, set_matmul_flags
from foreswe_bench.runtime import env_info, setup_logging, write_json

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engineering_audit", "optimization",
                   "cuda_graphs", "benchmark_results", "correctness")


def live(model):
    return [(n, p) for n, p in model.named_parameters() if not is_dead(n)]


def run_cfg(name, ns, ds, batches, traj):
    torch._dynamo.reset()
    model = build_model(ns, "planB24", seed=0).eval()
    init = [p.detach().clone() for _, p in live(model)]
    step = make_step(name, model)
    step.setup(ds.example(batches[0]))
    rec = dict(cfg=name)
    rec["warmup_restored_bitwise"] = all(torch.equal(p, q) for (_, p), q in zip(live(model), init))
    rec["optimizer_state_zero_after_setup"] = all(float(v.abs().max()) == 0.0 for st in step.opt.state.values()
                                                   for v in st.values() if torch.is_tensor(v)) if step.opt.state else True
    ex = ds.example(batches[0])
    out = step.infer(ex).float().clone()
    loss = step.train_step(ex)
    grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for _, p in live(model)]
    upd = [p.detach() - q for (_, p), q in zip(live(model), init)]
    moms = [(step.opt.state[p]["exp_avg"].clone(), step.opt.state[p]["exp_avg_sq"].clone())
            for _, p in live(model) if p in step.opt.state]
    for b in traj:
        step.train_step(ds.example(b))
    final = [p.detach().clone() for _, p in live(model)]
    rec["param_change_l2"] = float(torch.sqrt(sum(((f - q) ** 2).sum() for f, q in zip(final, init))).item())
    data = dict(out=out.cpu(), loss=loss, grads=[g.cpu() for g in grads], upd=[u.cpu() for u in upd],
                m=[m.cpu() for m, _ in moms], v=[v.cpu() for _, v in moms], final=[f.cpu() for f in final])
    extra = {}
    if isinstance(step, GraphStep):
        r0 = torch.cuda.memory_reserved()
        for i in range(200):
            step.train_step(ds.example(traj[i % len(traj)]))
        torch.cuda.synchronize()
        extra["reserved_growth_200_replays_mib"] = (torch.cuda.memory_reserved() - r0) / 2**20
    rec.update(extra)
    del step, model
    gc.collect()
    torch.cuda.empty_cache()
    return rec, data


def rng_check(ns, ds, b):
    """Train-mode fwd+bwd graph replayed twice on the same input must give different losses (fresh dropout)."""
    model = build_model(ns, "planB24", seed=0).train()
    step = make_step("G3f", model)
    step.setup(ds.example(b))
    step.static.load(ds.example(b))
    step.graph.replay()
    l1 = step.static_loss.item()
    step.graph.replay()
    l2 = step.static_loss.item()
    model.eval()
    step_e = make_step("G3f", build_model(ns, "planB24", seed=0).eval())
    step_e.setup(ds.example(b))
    step_e.graph.replay()
    e1 = step_e.static_loss.item()
    step_e.graph.replay()
    e2 = step_e.static_loss.item()
    return dict(train_mode_replays_differ=l1 != l2, train_losses=[l1, l2], eval_mode_replays_equal=e1 == e2,
                eval_losses=[e1, e2])


def compare(d, ref):
    e = dict(output=metrics.tensor_errors(d["out"], ref["out"]),
             loss_abs=abs(d["loss"] - ref["loss"]), loss_rel=abs(d["loss"] - ref["loss"]) / abs(ref["loss"]),
             grad=metrics.grad_errors(d["grads"], ref["grads"]),
             update=metrics.grad_errors(d["upd"], ref["upd"]),
             exp_avg=metrics.grad_errors(d["m"], ref["m"]), exp_avg_sq=metrics.grad_errors(d["v"], ref["v"]))
    num = sum(float(((f - g) ** 2).sum()) for f, g in zip(d["final"], ref["final"]))
    den = sum(float(((g) ** 2).sum()) for g in ref["final"])
    e["params_after_traj_rel_l2"] = (num / den) ** 0.5
    e["bitwise_output"] = bool(torch.equal(d["out"], ref["out"]))
    e["bitwise_grads"] = all(torch.equal(a, b) for a, b in zip(d["grads"], ref["grads"]))
    e["bitwise_final_params"] = all(torch.equal(a, b) for a, b in zip(d["final"], ref["final"]))
    return e


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cfgs", nargs="+", default=["G0c", "G1", "G2", "G3f", "G3", "G4"])
    ap.add_argument("--traj", type=int, default=20)
    a = ap.parse_args()
    log = setup_logging()
    set_matmul_flags("tf32")
    ns = load_namespace("planB24", "cuda")
    ds = fdata.real(ns)
    batches = [int(x) for x in ds.train_indices[:1]]
    traj = [int(x) for x in ds.train_indices[1:1 + a.traj]]
    res = dict(env=env_info(), batches=batches, traj_len=len(traj), configs={})

    # FP32 reference (TF32 off, eager) for context: output + grads at seed-0 weights
    with matmul_flags("fp32"):
        m32 = build_model(ns, "planB24", seed=0).eval()
        ex = ds.example(batches[0])
        with torch.no_grad():
            o32 = m32(ex["daily"], ex["weekly"], ex["yearly"], ex["prompts"], ex["dist"], ex["ang"])[0].float().cpu()
        m32.zero_grad()
        _, l32 = forward_loss(m32, ex)
        l32.backward()
        g32 = [p.grad.detach().cpu() for _, p in live(m32)]
        del m32
        torch.cuda.empty_cache()

    rec0, ref = run_cfg("G0", ns, ds, batches, traj)
    res["configs"]["G0"] = dict(rec0, vs_fp32=dict(output=metrics.tensor_errors(ref["out"], o32),
                                                   grad=metrics.grad_errors(ref["grads"], g32)))
    refs = {"G0": ref}
    for c in a.cfgs:
        try:
            rec, d = run_cfg(c, ns, ds, batches, traj)
        except Exception as exc:  # report, do not hide
            log.exception("%s failed", c)
            res["configs"][c] = dict(cfg=c, error="{}: {}".format(type(exc).__name__, exc))
            continue
        rec["vs_G0"] = compare(d, ref)
        rec["vs_fp32"] = dict(output=metrics.tensor_errors(d["out"], o32), grad=metrics.grad_errors(d["grads"], g32))
        if "G0c" in refs and CONFIGS[c]["kind"] in ("graph", "compile_graph"):
            rec["vs_G0c"] = compare(d, refs["G0c"])
        if c == "G0c":
            refs["G0c"] = d
        res["configs"][c] = rec
        log.info("%s vs G0: out %.2e grad %.2e upd %.2e traj %.2e bitwise(out,grad,params)=%s,%s,%s", c,
                 rec["vs_G0"]["output"]["rel_l2"], rec["vs_G0"]["grad"]["rel_l2"], rec["vs_G0"]["update"]["rel_l2"],
                 rec["vs_G0"]["params_after_traj_rel_l2"], rec["vs_G0"]["bitwise_output"],
                 rec["vs_G0"]["bitwise_grads"], rec["vs_G0"]["bitwise_final_params"])
        del d
        gc.collect()
    res["rng_check"] = rng_check(ns, ds, batches[0])
    log.info("RNG: %s", res["rng_check"])
    os.makedirs(OUT, exist_ok=True)
    write_json(res, os.path.join(OUT, "graph_correctness.json"))


if __name__ == "__main__":
    main()
