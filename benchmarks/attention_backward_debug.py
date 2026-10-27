"""Task 2 diagnostic: locate the first backward intermediate where the fused kernel diverges from eager FP32.

Real layer-0 / layer-7 Q, K, V from the trained Plan B checkpoint (benchmarks.attention_study.capture_real_qkv),
one fixed upstream gradient dO. For every intermediate the script reports max-abs and rel-L2 error vs an
explicit FP64 computation, for (a) explicit FP32 math with eager's ops and (b) the fused v2 kernel
(intermediates read back through geo_attention_v2.DEBUG). Also splits errors by row saturation of P1.

    python -m benchmarks.attention_backward_debug [--acc 0] [--prec fp32|tf32] --out DIR
"""
import argparse
import math
import os

import torch

from benchmarks.attention_study import capture_real_qkv
from foreswe_bench import data as fdata
from foreswe_bench import geo_attention_v2 as v2
from foreswe_bench import paths
from foreswe_bench.geo_attention import eager_geo_attention
from foreswe_bench.model import load_namespace
from foreswe_bench.precision import matmul_flags
from foreswe_bench.runtime import env_info, setup_logging, write_json

OUT = paths.REPO / "engineering_audit" / "optimization" / "fused_attention" / "benchmark_results" / "backward_debug"


def explicit(q, k, v, dist, ang, a, b, g, do):
    """All intermediates of the forward and backward, written out (same dtype as the inputs)."""
    d = q.shape[-1]
    S = q @ k.transpose(-2, -1) / math.sqrt(d)
    m1 = S.max(-1).values
    l1 = torch.exp(S - m1[..., None]).sum(-1)
    P1 = torch.softmax(S, -1)
    Dp, Ap = 1 / (1 + dist), 1 / (1 + ang)
    Z = a * P1 + b * Dp + g * Ap
    P2 = torch.softmax(Z, -1)
    O = P2 @ v
    dP2 = do @ v.transpose(-2, -1)
    delta2 = (dP2 * P2).sum(-1)
    dZ = P2 * (dP2 - delta2[..., None])
    dP1 = a * dZ
    delta1 = (dP1 * P1).sum(-1)
    dS = P1 * (dP1 - delta1[..., None])
    return dict(S=S, m1=m1, l1=l1, P1=P1, Z=Z, P2=P2, O=O, dP2=dP2, delta2=delta2, delta2_doO=(do * O).sum(-1),
                dZ=dZ, dP1=dP1, delta1=delta1, dS=dS, dQ=dS @ k / math.sqrt(d), dK=dS.transpose(-2, -1) @ q / math.sqrt(d),
                dV=P2.transpose(-2, -1) @ do, dalpha=(dZ * P1).sum(), dbeta=(dZ * Dp).sum(), dgamma=(dZ * Ap).sum())


def err(x, ref):
    x64, r64 = x.double(), ref.double()
    diff = (x64 - r64).abs()
    n = torch.linalg.vector_norm(r64).item()
    return dict(max_abs=diff.max().item(), rel_l2=torch.linalg.vector_norm(x64 - r64).item() / n if n else float("nan"),
                ref_norm=n, ref_max_abs=r64.abs().max().item())


def run_fused(inp, do):
    x = [t.detach().clone().requires_grad_(i in (0, 1, 2, 5, 6, 7)) for i, t in enumerate(inp)]
    v2.DEBUG.clear()
    v2.DEBUG["on"] = True
    o = v2.fused_geo_attention_v2(*x)
    o.backward(do)
    v2.DEBUG["on"] = False
    B, H, N, D = inp[0].shape
    shp = (B, H, N)
    dbg = {k: t.reshape(B, H, *t.shape[1:]) for k, t in v2.DEBUG.items() if torch.is_tensor(t)}
    return dict(O=o.detach(), m1=dbg["m1"].reshape(shp), l1=dbg["l1"].reshape(shp), delta2_doO=dbg["delta2"].reshape(shp),
                delta1=dbg["delta1"].reshape(shp), P1=dbg["P1"], dZ=dbg["dZ"], dS=dbg["dS"],
                dQ=x[0].grad, dK=x[1].grad, dV=x[2].grad, dalpha=x[5].grad.reshape(()), dbeta=x[6].grad.reshape(()),
                dgamma=x[7].grad.reshape(()), l2=dbg["l2"].reshape(shp), rs=dbg["rs"].reshape(shp))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="fast_experimental", choices=sorted(v2.PRESETS),
                    help="numerical preset (fast = the original defect, accurate = the fix)")
    ap.add_argument("--acc", type=int, default=0, help="dS formulation in the v2 kernel")
    ap.add_argument("--prec", default="fp32", choices=["fp32", "tf32"])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    log = setup_logging()
    v2.use_preset(a.preset)
    v2.CFG["acc"] = a.acc
    ns = load_namespace("planB24", "cuda")
    ds = fdata.real(ns)
    real = capture_real_qkv(ns, ds)
    res = dict(env=env_info(), preset=a.preset, cfg={k: v for k, v in v2.CFG.items()}, acc=a.acc, prec=a.prec, layers={})
    order = ["S", "m1", "l1", "P1", "Z", "P2", "O", "dP2", "delta2", "delta2_doO", "dZ", "dP1", "delta1", "dS",
             "dQ", "dK", "dV", "dalpha", "dbeta", "dgamma"]
    for L, inp in real.items():
        torch.manual_seed(1234)
        do = torch.randn_like(inp[0])
        ref = explicit(*[t.double() for t in inp], do.double())
        with matmul_flags(a.prec):
            e32 = explicit(*inp, do)
            # sanity: explicit FP32 gradients == autograd eager FP32
            xe = [t.detach().clone().requires_grad_(i in (0, 1, 2)) for i, t in enumerate(inp)]
            eager_geo_attention(*xe).backward(do)
            fused = run_fused(inp, do)
        sat = ref["P1"].max(-1).values > 1 - 1e-6
        rows = {}
        for key in order:
            r = dict(eager_fp32=err(e32[key], ref[key]))
            if key in fused:
                r["fused"] = err(fused[key], ref[key])
            rows[key] = r
        # where does the dS / dQ error live? saturated vs non-saturated rows
        split = {}
        for name, mask in (("saturated_rows", sat), ("other_rows", ~sat)):
            if mask.any():
                split[name] = dict(n_rows=int(mask.sum()),
                                   dS_eager=err(e32["dS"][mask], ref["dS"][mask]),
                                   dS_fused=err(fused["dS"][mask], ref["dS"][mask]),
                                   dZ_fused=err(fused["dZ"][mask], ref["dZ"][mask]),
                                   P1_fused=err(fused["P1"][mask], ref["P1"][mask]),
                                   delta1_fused=err(fused["delta1"][mask], ref["delta1"][mask]),
                                   delta1_eager=err(e32["delta1"][mask], ref["delta1"][mask]))
        # normalisation consistency of P1: the softmax-Jacobian cancellation requires rowsum(P1) == 1
        rowsum = dict(fp64=(ref["P1"].sum(-1) - 1).abs().max().item(),
                      eager_fp32=(e32["P1"].double().sum(-1) - 1).abs().max().item(),
                      fused_recomputed_P1=(fused["P1"].double().sum(-1) - 1).abs().max().item(),
                      fused_B1_rowsum_minus_1=(fused["rs"].double() - 1).abs().max().item())
        # does the backward recompute S (and therefore P1's peak) differently from the forward's m1?
        peak = fused["P1"].double().max(-1).values
        rowsum["fused_P1_peak_minus_1_on_saturated_rows"] = (peak[sat] - 1).abs().max().item() if sat.any() else None
        rowsum["m1_fused_minus_m1_fp64_max_abs"] = (fused["m1"].double() - ref["m1"]).abs().max().item()
        rowsum["m1_eager32_minus_fp64_max_abs"] = (e32["m1"].double() - ref["m1"]).abs().max().item()
        log.info("P1 normalisation |rowsum-1|: %s", {k: "%.3e" % v if v is not None else None for k, v in rowsum.items()})
        res["layers"][L] = dict(rows=rows, split=split, p1_normalisation=rowsum, autograd_vs_explicit_dQ=err(xe[0].grad, e32["dQ"]),
                                alpha_beta_gamma=[t.item() for t in inp[5:]],
                                p1_rowmax_quantiles=torch.quantile(ref["P1"].max(-1).values.flatten().float(),
                                                                   torch.tensor([0.0, 0.01, 0.5, 0.99, 1.0], device="cuda")).tolist(),
                                saturated_fraction=sat.double().mean().item())
        log.info("=== layer %d  (alpha, beta, gamma) = %s, saturated rows %.4f", L, res["layers"][L]["alpha_beta_gamma"],
                 res["layers"][L]["saturated_fraction"])
        for key in order:
            r = rows[key]
            f = r.get("fused")
            log.info("%-11s ref|max| %9.3e | eager32 abs %9.3e rel %9.3e | fused abs %s rel %s", key,
                     r["eager_fp32"]["ref_max_abs"], r["eager_fp32"]["max_abs"], r["eager_fp32"]["rel_l2"],
                     "%9.3e" % f["max_abs"] if f else "    -    ", "%9.3e" % f["rel_l2"] if f else "    -    ")
        for name, sp in split.items():
            log.info("%s (%d rows): dS abs eager %.3e fused %.3e | dZ fused %.3e | P1 fused %.3e | delta1 eager %.3e fused %.3e",
                     name, sp["n_rows"], sp["dS_eager"]["max_abs"], sp["dS_fused"]["max_abs"], sp["dZ_fused"]["max_abs"],
                     sp["P1_fused"]["max_abs"], sp["delta1_eager"]["max_abs"], sp["delta1_fused"]["max_abs"])
    os.makedirs(a.out, exist_ok=True)
    write_json(res, os.path.join(a.out, "backward_debug_{}_acc{}_{}.json".format(a.preset, a.acc, a.prec)))


if __name__ == "__main__":
    main()
