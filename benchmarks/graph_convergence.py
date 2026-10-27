"""Bounded real-data training for one G-configuration (same protocol as benchmarks/convergence.py:
seed-0 weights, production data order, train mode, dropout seed 1234, TF32 + fused AdamW).

    python -m benchmarks.graph_convergence --cfg G3 --steps 300     # --steps 0 = one epoch
"""
import argparse
import csv
import math
import os
import time

import numpy as np
import torch

from foreswe_bench import data as fdata
from foreswe_bench.graphs import make_step
from foreswe_bench.model import build_model, is_dead, load_namespace
from foreswe_bench.precision import set_matmul_flags
from foreswe_bench.runtime import env_info, setup_logging, write_json

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engineering_audit", "optimization",
                   "cuda_graphs", "benchmark_results", "convergence")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--dropout-seed", type=int, default=1234)
    a = ap.parse_args()
    log = setup_logging()
    set_matmul_flags("tf32")
    ns = load_namespace("planB24", "cuda")
    ds = fdata.real(ns)
    model = build_model(ns, "planB24", seed=0).train()
    live = [p for n, p in model.named_parameters() if not is_dead(n)]
    step = make_step(a.cfg, model)
    t0 = time.time()
    step.setup(ds.example(int(ds.train_indices[0])))
    setup_s = time.time() - t0
    torch.manual_seed(a.dropout_seed)
    torch.cuda.manual_seed_all(a.dropout_seed)
    steps = a.steps or len(ds.train_indices)
    prev = [p.detach().clone() for p in live]
    rows, t0 = [], time.time()
    for s in range(steps):
        idx = int(ds.train_indices[s])
        loss = step.train_step(ds.example(idx))
        with torch.no_grad():
            g = torch.linalg.vector_norm(torch.stack(torch._foreach_norm([p.grad for p in live]))).item()
            torch._foreach_sub_(prev, live)
            u = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(prev))).item()
            torch._foreach_copy_(prev, live)
        rows.append(dict(step=s + 1, issue_index=idx, loss=loss, grad_norm=g, update_norm=u,
                         nonfinite=int(not (math.isfinite(loss) and math.isfinite(g) and math.isfinite(u)))))
        if (s + 1) % 500 == 0 or s == 0:
            log.info("%s step %d loss %.4g grad %.4g upd %.3e (%.1fs)", a.cfg, s + 1, loss, g, u, time.time() - t0)
    elapsed = time.time() - t0
    preds = {}
    for y, idx in sorted(ds.test_issue.items()):
        preds["out_{}".format(y)] = step.infer(ds.example(int(idx))).float().cpu().numpy()
    tag = "{}_{}steps".format(a.cfg, steps)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, tag + "_test_predictions.npz"), **preds)
    with open(os.path.join(OUT, tag + "_trace.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    L = np.array([r["loss"] for r in rows])
    write_json(dict(cfg=a.cfg, steps=steps, setup_s=setup_s, elapsed_s=elapsed, env=env_info(),
                    final_loss=float(L[-1]), mean_loss_last50=float(L[-50:].mean()),
                    nonfinite_steps=int(sum(r["nonfinite"] for r in rows)), final_grad_norm=rows[-1]["grad_norm"]),
               os.path.join(OUT, tag + "_summary.json"))
    log.info("%s done: %d steps in %.1fs (+%.1fs setup), last-50 mean loss %.6g", a.cfg, steps, elapsed, setup_s,
             L[-50:].mean())


if __name__ == "__main__":
    main()
