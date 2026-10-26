"""GPU regression tests on the full-size Plan B model (run on a Perlmutter GPU node).

Bounds are set from the measured values in engineering_audit/optimization (about 3x headroom), so a
change in kernels, flags or library versions that degrades accuracy is caught.
"""
import copy

import pytest
import torch

from foreswe_bench import data as fdata
from foreswe_bench import metrics
from foreswe_bench.model import build_model, live_named_parameters, load_namespace
from foreswe_bench.precision import make_optimizer, matmul_flags
from foreswe_bench.runtime import forward_loss

pytestmark = pytest.mark.gpu

# measured on A100-40GB, torch 2.6.0, synthetic batch, seed-0 weights (see NUMERICAL_ERROR_ANALYSIS.md)
BOUNDS = {"tf32": dict(out=3e-3, grad=3e-2), "bf16": dict(out=3e-2, grad=2e-1)}


@pytest.fixture(scope="module")
def setup():
    ns = load_namespace("planB24", "cuda")
    ds = fdata.synthetic(ns)
    model = build_model(ns, "planB24", seed=0).eval()
    return ds, model


def _run(model, ex, p):
    model.zero_grad(set_to_none=True)
    with matmul_flags(p):
        out, _, loss = forward_loss(model, ex, p)
        loss.backward()
    return out.float().detach(), [q.grad.detach().clone() for _, q in live_named_parameters(model)]


def test_fp32_forward_backward_is_deterministic(setup):
    ds, model = setup
    ex = ds.example(int(ds.train_indices[0]))
    o1, g1 = _run(model, ex, "fp32")
    o2, g2 = _run(model, ex, "fp32")
    assert torch.equal(o1, o2)
    assert metrics.grad_errors(g2, g1)["rel_l2"] == 0.0


@pytest.mark.parametrize("p", ["tf32", "bf16"])
def test_reduced_precision_error_bounds(setup, p):
    ds, model = setup
    ex = ds.example(int(ds.train_indices[0]))
    o_ref, g_ref = _run(model, ex, "fp32")
    o, g = _run(model, ex, p)
    e_out = metrics.tensor_errors(o, o_ref)
    e_grad = metrics.grad_errors(g, g_ref)
    assert e_out["nan"] == 0 and e_out["inf"] == 0 and e_grad["nan"] == 0
    assert e_out["rel_l2"] < BOUNDS[p]["out"], e_out
    assert e_grad["rel_l2"] < BOUNDS[p]["grad"], e_grad


def test_fused_adamw_matches_foreach(setup):
    ds, model = setup
    mb = copy.deepcopy(model)
    ex = ds.example(int(ds.train_indices[0]))
    _run(model, ex, "fp32")
    for (_, pa), (_, pb) in zip(live_named_parameters(model), live_named_parameters(mb)):
        pb.grad = pa.grad.clone()
    p0 = [p.detach().clone() for _, p in live_named_parameters(model)]
    make_optimizer(model.parameters(), "foreach").step()
    make_optimizer(mb.parameters(), "fused").step()
    ua = [p - x for (_, p), x in zip(live_named_parameters(model), p0)]
    ub = [p - x for (_, p), x in zip(live_named_parameters(mb), p0)]
    e = metrics.grad_errors(ub, ua)
    assert e["rel_l2"] < 1e-5 and e["cosine"] > 1 - 1e-8, e
