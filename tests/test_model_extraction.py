"""The production model classes are extracted verbatim and behave as documented (CPU)."""
import ast

import torch

from conftest import small_batch
from foreswe_bench import paths
from foreswe_bench.model import extract_source, is_dead, live_named_parameters, param_group
from foreswe_bench.runtime import forward_loss


def _class_src(variant, name):
    mod = extract_source(paths.SOURCES[variant], (name,))
    return ast.dump(mod.body[0])


def test_planB_architecture_matches_upstream_weekly():
    # Plan B reformatted the class code; the parameter layout must be identical except the head width.
    from foreswe_bench.model import build_model, load_namespace
    shapes = {}
    for v in ("planB24", "weekly4"):
        ns = load_namespace(v, "cpu")
        m = build_model(ns, v, seed=0, device="cpu", model_dim=64, nhead=4, enc_layers_count=1, n_output_heads=24)
        shapes[v] = {k: tuple(t.shape) for k, t in m.state_dict().items()}
    assert shapes["planB24"] == shapes["weekly4"]
    assert "CustomEncoderLayer" in _class_src("planB24", "SWETransformer")


def test_small_model_forward_backward(small_model_factory):
    ns, model = small_model_factory()
    ex = small_batch()
    out, rep, loss = forward_loss(model, ex, "fp32")
    assert out.shape == (1, 24, 24) and rep.shape == (1, 24, 8)
    loss.backward()
    for n, p in model.named_parameters():
        if is_dead(n):
            assert p.grad is None, n  # CrossVarTransformer.encoder_layers are never called
    assert all(p.grad is not None for _, p in live_named_parameters(model))


def test_seeded_init_is_deterministic(small_model_factory):
    _, m1 = small_model_factory(seed=3)
    _, m2 = small_model_factory(seed=3)
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert n1 == n2 and torch.equal(p1, p2)


def test_param_groups_cover_live_params(small_model_factory):
    _, model = small_model_factory()
    groups = {param_group(n) for n, _ in live_named_parameters(model)}
    assert "other" not in groups
    assert {"crossvar_daily", "crossvar_weekly", "crossvar_yearly", "output_mlp", "final_head"} <= groups
