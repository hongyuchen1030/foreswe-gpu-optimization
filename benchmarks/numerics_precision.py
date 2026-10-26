"""Phase 1: FP32 vs TF32 vs BF16 output/loss/gradient error on identical inputs, before any update.

Baseline = FP32 with TF32 disabled (the spec's practical reference). Every precision is also compared
with an FP64 copy of the same weights, which shows how far FP32 itself is from exact arithmetic.
Caveat: production CrossVarTransformer.forward hard-casts the stacked variable embeddings to
float32 (`.to(dtype=torch.float32)`), so the FP64 model rounds once to FP32 at that point.

Dropout is disabled (model.eval()) so all precisions see the identical computational graph;
the gradients are still the full backward pass.

    python -m benchmarks.numerics_precision --weights init planB --out DIR
"""
import argparse
import copy
import csv
import os

import numpy as np
import torch

from foreswe_bench import data as fdata
from foreswe_bench import metrics, paths
from foreswe_bench.model import build_model, live_named_parameters, load_checkpoint_weights, load_namespace, param_group
from foreswe_bench.precision import matmul_flags
from foreswe_bench.runtime import env_info, forward_loss, setup_logging, write_json

PRECISIONS = ("fp32", "tf32", "bf16")


def as64(ex):
    return {k: v.double() for k, v in ex.items()}


def run(model, ex, precision):
    """Forward + loss + backward; returns output, loss, live grads (cloned, fp32/fp64 as computed)."""
    model.zero_grad(set_to_none=True)
    with matmul_flags("fp32" if precision == "fp64" else precision):
        out, _, loss = forward_loss(model, ex, "fp32" if precision == "fp64" else precision)
        loss.backward()
    torch.cuda.synchronize()
    named = [(n, p.grad.detach().clone()) for n, p in live_named_parameters(model)]
    model.zero_grad(set_to_none=True)  # keep only the cloned copy alive (FP64 grads are 3.4 GB)
    return out.detach().float() if precision != "fp64" else out.detach(), loss.detach(), named


def activation_dtypes(model, ex):
    """dtype of each module type's output under BF16 autocast (first occurrence per type/name)."""
    seen, hooks = {}, []
    for name, m in model.named_modules():
        if name.startswith("historical_") and ".encoder_layers." in name:
            continue
        def hook(mod, inp, out, name=name):
            o = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(o):
                key = type(mod).__name__
                seen.setdefault(key, dict(module=name, dtype=str(o.dtype)))
        hooks.append(m.register_forward_hook(hook))
    with torch.no_grad(), matmul_flags("bf16"):
        out, _, loss = forward_loss(model, ex, "bf16")
    for h in hooks:
        h.remove()
    seen["<model output>"] = dict(module="SWETransformer.forward", dtype=str(out.dtype))
    seen["<MSE loss>"] = dict(module="nn.MSELoss", dtype=str(loss.dtype))
    return seen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", nargs="+", default=["init", "planB"], choices=["init", "planB"])
    ap.add_argument("--n-train", type=int, default=4, help="real training issue dates (spread over the period)")
    ap.add_argument("--data", default="real", choices=["real", "synthetic"])
    ap.add_argument("--out", default=str(paths.RESULTS_DIR / "numerics_precision"))
    a = ap.parse_args()
    log = setup_logging()
    ns = load_namespace("planB24", "cuda")
    ds = fdata.real(ns) if a.data == "real" else fdata.synthetic(ns)
    tr = ds.train_indices
    batches = [int(tr[i]) for i in np.linspace(0, len(tr) - 1, a.n_train).astype(int)] + \
              [int(ds.test_issue[y]) for y in sorted(ds.test_issue)]
    results, rows = dict(env=env_info(), data=ds.meta, batches=batches), []

    for wname in a.weights:
        model = build_model(ns, "planB24", seed=0)
        if wname == "planB":
            results["planB_checkpoint_meta"] = load_checkpoint_weights(model, paths.CHECKPOINTS["planB24"])
        model.eval()
        model64 = copy.deepcopy(model).double().eval()
        per_batch = []
        for b in batches:
            ex = ds.example(b)
            ref64 = run(model64, as64(ex), "fp64")
            runs = {p: run(model, ex, p) for p in PRECISIONS}
            entry = dict(batch=b, kind="test_feb1" if b in ds.test_issue.values() else "train")
            for p, (out, loss, grads) in runs.items():
                for refname, (rout, rloss, rgrads) in (("fp32", runs["fp32"]), ("fp64", ref64)):
                    if p == "fp32" and refname == "fp32":
                        continue
                    e = dict(output=metrics.tensor_errors(out, rout),
                             loss=dict(value=loss.item(), ref=rloss.item(), abs=abs(loss.item() - rloss.item()),
                                       rel=abs(loss.item() - rloss.item()) / abs(rloss.item())),
                             grad=metrics.grad_errors([g for _, g in grads], [g for _, g in rgrads]),
                             grad_by_group=metrics.grouped_grad_errors(grads, rgrads, param_group))
                    entry["{}_vs_{}".format(p, refname)] = e
                    rows.append(dict(weights=wname, batch=b, kind=entry["kind"], precision=p, reference=refname,
                                     out_max_abs=e["output"]["max_abs"], out_rel_l2=e["output"]["rel_l2"],
                                     loss_rel=e["loss"]["rel"], grad_rel_l2=e["grad"]["rel_l2"],
                                     grad_cosine=e["grad"]["cosine"], nan=e["output"]["nan"] + e["grad"]["nan"],
                                     inf=e["output"]["inf"] + e["grad"]["inf"]))
            per_batch.append(entry)
            log.info("%s batch %d: out rel-L2 vs fp32  tf32 %.2e  bf16 %.2e | grad rel-L2 tf32 %.2e bf16 %.2e | fp32 vs fp64 out %.2e",
                     wname, b, entry["tf32_vs_fp32"]["output"]["rel_l2"], entry["bf16_vs_fp32"]["output"]["rel_l2"],
                     entry["tf32_vs_fp32"]["grad"]["rel_l2"], entry["bf16_vs_fp32"]["grad"]["rel_l2"],
                     entry["fp32_vs_fp64"]["output"]["rel_l2"])
            del runs, ref64
        results[wname] = dict(per_batch=per_batch, activation_dtypes_bf16=activation_dtypes(model, ds.example(batches[0])))
        del model, model64
        torch.cuda.empty_cache()

    os.makedirs(a.out, exist_ok=True)
    write_json(results, os.path.join(a.out, "numerics_precision.json"))
    with open(os.path.join(a.out, "numerics_precision.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    log.info("wrote %s", a.out)


if __name__ == "__main__":
    main()
