"""Numerical comparison metrics (all reductions in float64)."""
import math

import torch


def _f64(t):
    return t.detach().to(torch.float64)


def tensor_errors(x, ref):
    """max |x-ref|, relative L2 ||x-ref||/||ref||, and NaN/Inf counts of x."""
    x64, r64 = _f64(x), _f64(ref)
    diff = x64 - r64
    ref_norm = torch.linalg.vector_norm(r64).item()
    return dict(max_abs=diff.abs().max().item(),
                rel_l2=torch.linalg.vector_norm(diff).item() / ref_norm if ref_norm > 0 else math.nan,
                ref_l2=ref_norm, nan=int(torch.isnan(x).sum().item()), inf=int(torch.isinf(x).sum().item()))


def grad_errors(grads, ref_grads):
    """Global relative L2 and cosine over a list of gradient tensors (concatenated)."""
    num = den = dot = nx = 0.0
    nan = inf = 0
    for g, r in zip(grads, ref_grads):
        g64, r64 = _f64(g), _f64(r)
        num += torch.sum((g64 - r64) ** 2).item()
        den += torch.sum(r64 ** 2).item()
        dot += torch.sum(g64 * r64).item()
        nx += torch.sum(g64 ** 2).item()
        nan += int(torch.isnan(g).sum().item())
        inf += int(torch.isinf(g).sum().item())
    return dict(rel_l2=math.sqrt(num / den) if den > 0 else math.nan,
                cosine=dot / math.sqrt(nx * den) if nx > 0 and den > 0 else math.nan,
                ref_l2=math.sqrt(den), nan=nan, inf=inf)


def grouped_grad_errors(named_grads, named_ref, group_fn):
    groups = {}
    for (n, g), (n2, r) in zip(named_grads, named_ref):
        assert n == n2, (n, n2)
        groups.setdefault(group_fn(n), ([], []))
        groups[group_fn(n)][0].append(g)
        groups[group_fn(n)][1].append(r)
    return {k: grad_errors(gs, rs) for k, (gs, rs) in groups.items()}


def bits_equal_fraction(x, ref):
    return (x == ref).float().mean().item()
