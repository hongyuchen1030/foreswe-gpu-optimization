"""Numerical acceptance checks for the fused geo-attention kernels (shared by tests and reports).

Acceptance rule, fixed before evaluating any candidate (SATURATED_SOFTMAX_BACKWARD_DEBUG.md):
for the output and every gradient (dQ, dK, dV, dalpha, dbeta, dgamma), the candidate's max-abs error
against an FP64 reference must be <= 10 x eager FP32's max-abs error on the same inputs and the same dO,
plus a floor of 1e-7 * max|reference| (about one FP32 ulp) for the rare case where eager happens to be exact.
The relative-error denominator is never inflated and no rows are excluded.
"""
import math

import torch

from .geo_attention import eager_geo_attention
from .precision import matmul_flags

KEYS = ("output", "dQ", "dK", "dV", "dalpha", "dbeta", "dgamma")
FACTOR, FLOOR = 10.0, 1e-7


def case_inputs(name, B=1, H=4, N=128, D=192, seed=0, device="cuda"):
    """Synthetic regimes: random, near one-hot, uniform, moderate, wide signed ranges, large alpha."""
    g = torch.Generator().manual_seed(seed)
    q, k, v = (torch.randn(B, H, N, D, generator=g) for _ in range(3))
    dist = torch.rand(N, N, generator=g) * 1500
    ang = torch.rand(N, N, generator=g) * 20
    abg = [0.31, -0.21, 0.20]
    if name == "random":
        pass
    elif name == "near_one_hot":          # score std ~ 400: rows saturate like real layer 0
        q, k = q * 20, k * 20
    elif name == "uniform":               # Q = 0: every score is 0, P1 exactly uniform (like real layer 7)
        q = torch.zeros_like(q)
    elif name == "moderate":              # score std ~ 4: concentrated but not saturated
        q, k = q * 2, k * 2
    elif name == "wide_signed":           # scores spanning +-thousands, with a large positive offset per row
        q, k = q * 12, k * 12
        q[..., 0] += 50.0
        k[..., 0] += 50.0
    elif name == "large_alpha":
        q, k = q * 8, k * 8
        abg = [3.0, -1.5, 2.5]
    else:
        raise ValueError(name)
    a, b, c = (torch.tensor([x]) for x in abg)
    return [t.to(device) for t in (q, k, v, dist, ang, a, b, c)]


def grads(fn, inputs, do, dtype, prec="fp32"):
    x = [t.detach().to(dtype).requires_grad_(i in (0, 1, 2, 5, 6, 7)) for i, t in enumerate(inputs)]
    with matmul_flags(prec):
        o = fn(*x)
        o.backward(do.to(dtype))
    return dict(output=o.detach(), dQ=x[0].grad, dK=x[1].grad, dV=x[2].grad, dalpha=x[5].grad,
                dbeta=x[6].grad, dgamma=x[7].grad)


def err(a, ref):
    a64, r64 = a.double(), ref.double()
    rn = torch.linalg.vector_norm(r64).item()
    an = torch.linalg.vector_norm(a64).item()
    dot = float((a64 * r64).sum())
    return dict(max_abs=(a64 - r64).abs().max().item(),
                rel_l2=torch.linalg.vector_norm(a64 - r64).item() / rn if rn else math.nan,
                ref_norm=rn, ref_max_abs=r64.abs().max().item(),
                cosine=dot / (an * rn) if an and rn else math.nan)


def evaluate(fn, inputs, prec="fp32", seed=1234):
    """Errors of fn (fp32 storage, `prec` matmul mode) and of eager FP32 against the FP64 reference."""
    g = torch.Generator().manual_seed(seed)
    do = torch.randn(inputs[0].shape, generator=g).to(inputs[0].device)
    ref = grads(eager_geo_attention, inputs, do, torch.float64)
    cand = grads(fn, inputs, do, torch.float32, prec)
    eag = grads(eager_geo_attention, inputs, do, torch.float32, prec)
    out = {}
    for k in KEYS:
        ec, ee = err(cand[k], ref[k]), err(eag[k], ref[k])
        bound = FACTOR * ee["max_abs"] + FLOOR * ec["ref_max_abs"]
        out[k] = dict(candidate=ec, eager=ee, bound=bound, ratio_to_eager=ec["max_abs"] / ee["max_abs"]
                      if ee["max_abs"] else math.inf, passed=ec["max_abs"] <= bound)
    return out
