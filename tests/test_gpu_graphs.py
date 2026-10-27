"""GPU regression tests: manual CUDA-graph training step and the fused geo-attention prototype."""
import pytest
import torch

from foreswe_bench import data as fdata
from foreswe_bench import metrics
from foreswe_bench.geo_attention import eager_geo_attention, fused_geo_attention
from foreswe_bench.graphs import make_step
from foreswe_bench.model import build_model, is_dead, load_namespace
from foreswe_bench.precision import matmul_flags

pytestmark = pytest.mark.gpu


def _run(cfg, ns, ds, steps=3):
    model = build_model(ns, "planB24", seed=0).eval()  # dropout off: deterministic graph
    st = make_step(cfg, model)
    st.setup(ds.example(int(ds.train_indices[0])))
    losses = [st.train_step(ds.example(int(ds.train_indices[i]))) for i in range(steps)]
    return losses, [p.detach().clone() for n, p in model.named_parameters() if not is_dead(n)]


def test_full_step_graph_is_bitwise_identical_to_eager():
    with matmul_flags("tf32"):
        ns = load_namespace("planB24", "cuda")
        ds = fdata.synthetic(ns)
        le, pe = _run("G0c", ns, ds)
        lg, pg = _run("G3", ns, ds)
    assert le == lg
    assert all(torch.equal(a, b) for a, b in zip(pe, pg))
    assert le[0] != le[-1]  # parameters actually trained


def test_graph_replay_draws_fresh_dropout_masks():
    with matmul_flags("tf32"):
        ns = load_namespace("planB24", "cuda")
        ds = fdata.synthetic(ns)
        model = build_model(ns, "planB24", seed=0).train()
        st = make_step("G3f", model)
        ex = ds.example(int(ds.train_indices[0]))
        st.setup(ex)
        st.static.load(ex)
        st.graph.replay()
        a = st.static_loss.item()
        st.graph.replay()
        assert st.static_loss.item() != a


@pytest.mark.parametrize("shape", [(1, 2, 64, 64), (1, 3, 129, 48), (1, 16, 512, 192)])
def test_fused_attention_matches_fp64_reference_in_fp32(shape):
    g = torch.Generator().manual_seed(0)
    B, H, N, D = shape
    q, k, v = (torch.randn(B, H, N, D, generator=g).cuda() for _ in range(3))
    dist = (torch.rand(N, N, generator=g) * 1500).cuda()
    ang = (torch.rand(N, N, generator=g) * 20).cuda()
    abg = [((torch.rand(1, generator=g) - 0.5) * 2).cuda() for _ in range(3)]
    do = torch.randn(B, H, N, D, generator=g).cuda()
    inputs = [q, k, v, dist, ang] + abg

    def run(fn, dtype):
        x = [t.to(dtype).requires_grad_(i in (0, 1, 2, 5, 6, 7)) for i, t in enumerate(inputs)]
        o = fn(*x)
        o.backward(do.to(dtype))
        return o.detach(), [x[i].grad for i in (0, 1, 2, 5, 6, 7)]
    o64, g64 = run(eager_geo_attention, torch.float64)
    with matmul_flags("fp32"):
        o32, g32 = run(fused_geo_attention, torch.float32)
        oe, ge = run(eager_geo_attention, torch.float32)
    # The fused kernel must be as accurate as the production eager operator in the same precision
    # (both measured against float64). dalpha/dbeta/dgamma are sums of signed terms whose cancellation
    # inflates *relative* error for any fp32 implementation, so the bound is relative to eager, not absolute.
    err = lambda a, b: metrics.tensor_errors(a, b)["rel_l2"]  # noqa: E731
    assert err(o32, o64) <= 10 * err(oe, o64) + 1e-7
    for name, a, e, b in zip(("dQ", "dK", "dV", "dalpha", "dbeta", "dgamma"), g32, ge, g64):
        assert err(a, b) <= 10 * err(e, b) + 1e-7, (name, err(a, b), err(e, b))
