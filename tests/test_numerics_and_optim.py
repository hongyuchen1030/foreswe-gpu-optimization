"""Metric helpers, precision-flag hygiene, and optimizer-implementation equivalence (CPU)."""
import copy
import math

import pytest
import torch

from conftest import small_batch
from foreswe_bench import metrics
from foreswe_bench.kernels import categorize, is_tensor_core_gemm
from foreswe_bench.model import live_named_parameters
from foreswe_bench.precision import matmul_flags, make_optimizer
from foreswe_bench.runtime import forward_loss


def test_tensor_errors_basic():
    ref = torch.tensor([3.0, 4.0])
    e = metrics.tensor_errors(torch.tensor([3.0, 4.5]), ref)
    assert e["max_abs"] == pytest.approx(0.5) and e["rel_l2"] == pytest.approx(0.5 / 5.0)
    e = metrics.tensor_errors(torch.tensor([float("nan"), float("inf")]), ref)
    assert e["nan"] == 1 and e["inf"] == 1


def test_grad_errors_cosine():
    a = [torch.tensor([1.0, 0.0]), torch.tensor([0.0])]
    b = [torch.tensor([0.0, 1.0]), torch.tensor([0.0])]
    assert metrics.grad_errors(a, a)["cosine"] == pytest.approx(1.0)
    assert metrics.grad_errors(a, b)["cosine"] == pytest.approx(0.0)
    assert metrics.grad_errors(a, b)["rel_l2"] == pytest.approx(math.sqrt(2))


def test_matmul_flags_restored():
    before = (torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision())
    with matmul_flags("tf32"):
        assert torch.backends.cuda.matmul.allow_tf32
    assert (torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision()) == before


def test_kernel_classification():
    assert not is_tensor_core_gemm("ampere_sgemm_128x64_tn")
    assert is_tensor_core_gemm("ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_tn")
    assert is_tensor_core_gemm("sm80_xmma_gemm_f32f32_tf32f32_f32_tn_n_tilesize128x128x32_stage3_warpsize2x2x1_tensor16x8x8_kernel")
    assert categorize("ampere_sgemm_128x64_tn") == "GEMM (SIMT FP32)"
    assert categorize("void at::native::(anonymous namespace)::multi_tensor_apply_kernel<...>") .startswith("AdamW (multi_tensor_apply")


@pytest.mark.parametrize("impl", ["fused"])
def test_fused_adamw_matches_foreach_one_step_cpu(small_model_factory, impl):
    _, ma = small_model_factory()
    mb = copy.deepcopy(ma)
    ex = small_batch()
    _, _, loss = forward_loss(ma.eval(), ex, "fp32")
    loss.backward()
    for (_, pa), (_, pb) in zip(live_named_parameters(ma), live_named_parameters(mb)):
        pb.grad = pa.grad.clone()
    p0 = [p.detach().clone() for _, p in live_named_parameters(ma)]
    make_optimizer(ma.parameters(), "single_tensor").step()  # reference: per-tensor loop (CPU-safe)
    make_optimizer(mb.parameters(), impl).step()
    ua = [p - x for (_, p), x in zip(live_named_parameters(ma), p0)]
    ub = [p - x for (_, p), x in zip(live_named_parameters(mb), p0)]
    e = metrics.grad_errors(ub, ua)
    assert e["rel_l2"] < 1e-5 and e["cosine"] > 1 - 1e-9
