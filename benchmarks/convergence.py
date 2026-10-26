"""Phase 3: matched-budget short training on real ForeSWE data for one E0-E5 configuration.

Identical across configurations: seeded initial weights, data order (production order of the Plan B
training issue dates), dropout RNG seed, AdamW hyper-parameters, step count, loss. Train mode
(dropout on) as in production. No checkpoints are written.

Per step: loss, global grad L2 norm (live params), parameter-update L2 norm, NaN/Inf flags.
After training: predictions on the five Feb-1 test issue dates, saved for cross-config divergence.

    python -m benchmarks.convergence --exp E5 --steps 300 --out DIR
"""
import argparse
import csv
import math
import os
import time

import numpy as np
import torch

from foreswe_bench import data as fdata
from foreswe_bench import paths
from foreswe_bench.model import build_model, live_named_parameters, load_namespace
from foreswe_bench.precision import EXPERIMENTS, autocast, make_optimizer, matmul_flags, set_matmul_flags
from foreswe_bench.runtime import env_info, forward_loss, setup_logging, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", required=True, choices=sorted(EXPERIMENTS))
    ap.add_argument("--steps", type=int, default=300, help="0 = one full epoch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dropout-seed", type=int, default=1234)
    ap.add_argument("--out", default=str(paths.RESULTS_DIR / "convergence"))
    a = ap.parse_args()
    log = setup_logging()
    precision, impl = EXPERIMENTS[a.exp]
    set_matmul_flags(precision)
    ns = load_namespace("planB24", "cuda")
    ds = fdata.real(ns)
    model = build_model(ns, "planB24", seed=a.seed).train()
    opt = make_optimizer(model.parameters(), impl, 5e-6, 1e-4)
    live = [p for _, p in live_named_parameters(model)]
    steps = a.steps or len(ds.train_indices)
    torch.manual_seed(a.dropout_seed)
    torch.cuda.manual_seed_all(a.dropout_seed)
    rows, t0 = [], time.time()
    prev = [p.detach().clone() for p in live]
    for s in range(steps):
        idx = int(ds.train_indices[s])
        ex = ds.example(idx)
        _, _, loss = forward_loss(model, ex, precision)
        loss_v = loss.item()
        opt.zero_grad()
        loss.backward()
        gnorm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm([p.grad for p in live]))).item()
        opt.step()
        with torch.no_grad():
            torch._foreach_sub_(prev, live)  # prev <- prev - new = -update
            unorm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(prev))).item()
            torch._foreach_copy_(prev, live)
        bad = not (math.isfinite(loss_v) and math.isfinite(gnorm) and math.isfinite(unorm))
        rows.append(dict(step=s + 1, issue_index=idx, loss=loss_v, grad_norm=gnorm, update_norm=unorm, nonfinite=int(bad)))
        if (s + 1) % 100 == 0 or s == 0:
            log.info("%s step %d loss %.4g grad %.4g upd %.3e (%.1fs)", a.exp, s + 1, loss_v, gnorm, unorm, time.time() - t0)
    elapsed = time.time() - t0

    model.eval()
    preds = {}
    with torch.no_grad():
        for mode in (precision, "fp32"):  # own precision and FP32 evaluation of the trained weights
            with matmul_flags(mode):
                for y, idx in sorted(ds.test_issue.items()):
                    ex = ds.example(int(idx))
                    with autocast(mode):
                        out, _ = model(ex["daily"], ex["weekly"], ex["yearly"], ex["prompts"], ex["dist"], ex["ang"])
                    preds["{}_{}".format(mode, y)] = out.float().cpu().numpy()
                    preds["target_{}".format(y)] = ex["target"].float().cpu().numpy()
    tag = "{}_{}steps".format(a.exp, steps)
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, tag + "_test_predictions.npz"), **preds)
    with open(os.path.join(a.out, tag + "_trace.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    losses = np.array([r["loss"] for r in rows])
    write_json(dict(exp=a.exp, precision=precision, optimizer=impl, steps=steps, seed=a.seed,
                    dropout_seed=a.dropout_seed, elapsed_s=elapsed, env=env_info(),
                    final_loss=float(losses[-1]), mean_loss_last50=float(losses[-50:].mean()),
                    mean_loss_first50=float(losses[:50].mean()), nonfinite_steps=int(sum(r["nonfinite"] for r in rows)),
                    final_grad_norm=rows[-1]["grad_norm"]), os.path.join(a.out, tag + "_summary.json"))
    log.info("%s done: %d steps in %.1fs, last-50 mean loss %.4g", a.exp, steps, elapsed, losses[-50:].mean())


if __name__ == "__main__":
    main()
