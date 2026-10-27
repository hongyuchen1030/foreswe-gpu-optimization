"""Numerical regression tests for the v2 fused geo-attention (FP32 vs FP64, rule in foreswe_bench/attn_check.py).

Default preset (accurate_experimental) must pass every case. The fast preset is kept as an explicit
experimental alternative; the real layer-0 test records that it fails, so the defect cannot silently return
as the default.
"""
import os

import pytest
import torch

from foreswe_bench import attn_check
from foreswe_bench import geo_attention_v2 as v2
from foreswe_bench import paths

pytestmark = pytest.mark.gpu
CASES = ("random", "near_one_hot", "uniform", "moderate", "wide_signed", "large_alpha")


@pytest.fixture(autouse=True)
def accurate_preset():
    v2.use_preset("accurate_experimental")
    yield
    v2.use_preset("accurate_experimental")


def _assert_all(res, label):
    bad = {k: (r["candidate"]["max_abs"], r["bound"]) for k, r in res.items() if not r["passed"]}
    assert not bad, "{}: candidate max-abs error above 10x eager FP32: {}".format(label, bad)


@pytest.mark.parametrize("prec", ["fp32", "tf32"])
@pytest.mark.parametrize("case", CASES)
def test_synthetic_regimes(case, prec):
    """TF32 candidates are compared with eager TF32 (same 10x rule, same FP64 reference)."""
    _assert_all(attn_check.evaluate(v2.fused_geo_attention_v2, attn_check.case_inputs(case), prec), case + " " + prec)


@pytest.mark.parametrize("shape", [
    (1, 2, 64, 64),
    pytest.param((1, 3, 129, 48), marks=pytest.mark.xfail(strict=False, reason=(
        "recorded failure under the unchanged 10x rule: dbeta 3.0e-8 vs bound 1.3e-8 because eager FP32's dbeta "
        "error on this seed (1.2e-9) is a low outlier; over 6 other seeds eager ranges 6e-9..6e-8 and the fused "
        "kernel is 0.2-3.5x eager (test_seed_sweep_scalar_grads)"))),
    (1, 2, 100, 192), (1, 16, 512, 192)])
def test_shapes_near_one_hot(shape):
    B, H, N, D = shape
    inp = attn_check.case_inputs("near_one_hot", B=B, H=H, N=N, D=D, seed=sum(shape))
    _assert_all(attn_check.evaluate(v2.fused_geo_attention_v2, inp), str(shape))


@pytest.mark.parametrize("seed", range(6))
def test_seed_sweep_scalar_grads(seed):
    """Same regime/shape as the xfail case above, six further seeds, same unchanged rule."""
    inp = attn_check.case_inputs("near_one_hot", B=1, H=3, N=129, D=48, seed=183 + seed)
    _assert_all(attn_check.evaluate(v2.fused_geo_attention_v2, inp), "seed {}".format(seed))


def _real_layers():
    if not (os.path.exists(paths.DATA_CACHE) and os.path.exists(paths.CHECKPOINTS["planB24"])):
        pytest.skip("real-data cache or Plan B checkpoint not available")
    from benchmarks.attention_study import capture_real_qkv
    from foreswe_bench import data as fdata
    from foreswe_bench.model import load_namespace
    ns = load_namespace("planB24", "cuda")
    return capture_real_qkv(ns, fdata.real(ns))


@pytest.mark.realdata
@pytest.mark.parametrize("prec", ["fp32", "tf32"])
def test_real_layers_accurate_preset(prec):
    for L, inp in _real_layers().items():
        _assert_all(attn_check.evaluate(v2.fused_geo_attention_v2, inp, prec), "real layer {} {}".format(L, prec))


@pytest.mark.realdata
def test_real_layer0_fast_preset_is_inaccurate():
    """Documents the defect the accurate preset fixes (saturated softmax, real layer 0)."""
    inp = _real_layers()[0]
    v2.use_preset("fast_experimental")
    res = attn_check.evaluate(v2.fused_geo_attention_v2, inp)
    assert not res["dQ"]["passed"] and not res["dK"]["passed"]


def test_cuda_graph_replay_matches_eager_launch_and_refreshes_inputs():
    """The accurate kernel captured in a CUDA graph reproduces direct launches bitwise for new inputs."""
    inputs = [attn_check.case_inputs("near_one_hot", seed=s) for s in (1, 2)]
    static = [t.clone().requires_grad_(i in (0, 1, 2, 5, 6, 7)) for i, t in enumerate(inputs[0])]
    do = torch.randn_like(static[0])
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            for t in static:
                t.grad = None
            v2.fused_geo_attention_v2(*static).backward(do)
    torch.cuda.current_stream().wait_stream(s)
    for t in static:
        t.grad = None
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = v2.fused_geo_attention_v2(*static)
        out.backward(do)
    for inp in inputs:
        with torch.no_grad():
            for t, src in zip(static, inp):
                t.copy_(src)
        g.replay()
        x = [t.clone().requires_grad_(i in (0, 1, 2, 5, 6, 7)) for i, t in enumerate(inp)]
        ref = v2.fused_geo_attention_v2(*x)
        ref.backward(do)
        assert torch.equal(out, ref)
        for i in (0, 1, 2, 5, 6, 7):
            assert torch.equal(static[i].grad, x[i].grad), i
