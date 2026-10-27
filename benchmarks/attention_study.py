"""Part II: geo-biased attention core - correctness and performance of eager / torch.compile / fused Triton.

    python -m benchmarks.attention_study [--skip-model]
Writes engineering_audit/optimization/fused_attention/benchmark_results/attention_study.json
"""
import argparse
import copy
import math
import os

import torch

from foreswe_bench import data as fdata
from foreswe_bench import metrics, paths
from foreswe_bench.geo_attention import eager_geo_attention, fused_geo_attention, patch_model
from foreswe_bench.model import build_model, is_dead, load_checkpoint_weights, load_namespace
from foreswe_bench.precision import matmul_flags
from foreswe_bench.runtime import env_info, forward_loss, setup_logging, stats, write_json

OUT = paths.REPO / "engineering_audit" / "optimization" / "fused_attention" / "benchmark_results"


def make_inputs(B, H, N, D, seed, scale=1.0, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q, k, v = (torch.randn(B, H, N, D, generator=g) * scale for _ in range(3))
    dist = torch.rand(N, N, generator=g) * 1500
    ang = torch.rand(N, N, generator=g) * 20
    abg = (torch.rand(3, generator=g) - 0.5) * 2  # signed coefficients
    return [t.to(device) for t in (q, k, v, dist, ang)] + [abg[i:i + 1].to(device) for i in range(3)]


def grads_of(fn, inputs, do, dtype):
    x = [t.detach().to(dtype).requires_grad_(i in (0, 1, 2, 5, 6, 7)) for i, t in enumerate(inputs)]
    o = fn(*x)
    o.backward(do.to(dtype))
    return o.detach(), [x[i].grad.detach() for i in (0, 1, 2, 5, 6, 7)]


NAMES = ("dQ", "dK", "dV", "dalpha", "dbeta", "dgamma")


def compare(fn, inputs, prec, seed=0):
    """fn (in fp32 under `prec`) vs eager in float64."""
    torch.manual_seed(seed)
    do = torch.randn_like(inputs[0])
    o64, g64 = grads_of(eager_geo_attention, inputs, do, torch.float64)
    with matmul_flags(prec):
        o, g = grads_of(fn, inputs, do, torch.float32)
    res = dict(output=metrics.tensor_errors(o, o64))
    for n, a, b in zip(NAMES, g, g64):
        e = metrics.tensor_errors(a, b)
        e["cosine"] = metrics.grad_errors([a], [b])["cosine"]
        res[n] = e
    return res


def reference_gradcheck():
    """torch.autograd.gradcheck (float64 finite differences) of the eager reference operator, incl. alpha,
    beta, gamma. The fused kernel is then validated against this reference in compare(). A finite-difference
    test directly on the fp32 fused kernel is not meaningful (fp32 cannot resolve the perturbation)."""
    inp = make_inputs(1, 2, 8, 16, seed=11)
    x = [t.double().requires_grad_(i in (0, 1, 2, 5, 6, 7)) for i, t in enumerate(inp)]
    ok = torch.autograd.gradcheck(eager_geo_attention, tuple(x), eps=1e-6, atol=1e-8, rtol=1e-6)
    return dict(passed=bool(ok), shape=[1, 2, 8, 16], dtype="float64")


def time_fn(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return stats(ts)


def kernel_profile(fn):
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if "CUDA" in str(getattr(e, "device_type", "")) and e.name]
    launches = sum(1 for e in prof.events() if e.name in ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC"))
    return dict(kernels=len(names), host_launches=launches, names=sorted(set(n[:90] for n in names)))


def perf(inputs, prec):
    """Isolated fwd / fwd+bwd latency, memory and kernel counts at the production shape."""
    torch.manual_seed(0)
    do = torch.randn_like(inputs[0])
    compiled = torch.compile(eager_geo_attention)
    res = {}
    with matmul_flags(prec):
        for name, fn in (("eager", eager_geo_attention), ("compiled", compiled), ("fused", fused_geo_attention)):
            x = [t.detach().clone().requires_grad_(i in (0, 1, 2, 5, 6, 7)) for i, t in enumerate(inputs)]

            def fwd():
                with torch.no_grad():
                    return fn(*x)

            def fwdbwd():
                for t in x:
                    t.grad = None
                fn(*x).backward(do)
            r = dict(fwd_ms=time_fn(fwd), fwd_bwd_ms=time_fn(fwdbwd))
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            o = fn(*x)
            r["saved_for_backward_mib"] = (torch.cuda.memory_allocated() - base) / 2**20 - o.numel() * 4 / 2**20
            del o
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            fwdbwd()
            torch.cuda.synchronize()
            r["fwd_bwd_peak_extra_mib"] = (torch.cuda.max_memory_allocated() - base) / 2**20
            r["fwd_kernels"] = kernel_profile(fwd)
            r["fwd_bwd_kernels"] = kernel_profile(fwdbwd)
            res[name] = r
    B, H, N, D = inputs[0].shape
    f = res["fused"]["fwd_ms"]["median"] * 1e-3
    min_bytes = 4 * (4 * B * H * N * D + 2 * N * N)       # Q, K, V, O once + dist, ang once
    res["fused_fwd_effective"] = dict(
        min_bytes_mb=min_bytes / 1e6, achieved_gb_s_on_min_bytes=min_bytes / f / 1e9,
        flops_g=(3 * 2 * B * H * N * N * D) / 1e9,     # QK^T twice (two passes) + P2 V
        tflop_s=(3 * 2 * B * H * N * N * D) / f / 1e12)
    return res


def capture_real_qkv(ns, ds, layers=(0, 7)):
    """Real Q/K/V (after split_heads) of encoder layers, Plan B weights, Feb-1-2017 issue."""
    model = build_model(ns, "planB24", seed=0)
    load_checkpoint_weights(model, paths.CHECKPOINTS["planB24"])
    model.eval()
    ex = ds.example(int(ds.test_issue[2017]))
    got = {}
    hooks = []
    for L in layers:
        att = model.encoder_layers[L].self_attn
        orig = att.scaled_dot_product_attention

        def cap(Q, K, V, distances, angularities, mask=None, _L=L, _orig=orig, _att=att):
            got[_L] = [Q.detach().clone(), K.detach().clone(), V.detach().clone(), distances.detach().clone(),
                       angularities.detach().clone(), _att.alpha.detach().clone(), _att.beta.detach().clone(),
                       _att.gamma.detach().clone()]
            return _orig(Q, K, V, distances, angularities, mask)
        att.scaled_dot_product_attention = cap
        hooks.append(att)
    with torch.no_grad(), matmul_flags("fp32"):
        model(ex["daily"], ex["weekly"], ex["yearly"], ex["prompts"], ex["dist"], ex["ang"])
    del model
    torch.cuda.empty_cache()
    return got


def model_level(ns, ds):
    """Full Plan B model, E4 (TF32): eager vs fused attention - output, gradients, one training step time."""
    res = {}
    with matmul_flags("tf32"):
        ex = ds.example(int(ds.train_indices[0]))
        base = build_model(ns, "planB24", seed=0).eval()
        out = {}
        for impl in ("eager", "fused"):
            m = copy.deepcopy(base)
            patch_model(m, impl)
            m.zero_grad()
            o, _, loss = forward_loss(m, ex, "tf32")
            loss.backward()
            out[impl] = (o.detach().float(), [p.grad.detach().clone() for n, p in m.named_parameters() if not is_dead(n)])
            del m
        res["fused_vs_eager_tf32"] = dict(output=metrics.tensor_errors(out["fused"][0], out["eager"][0]),
                                          grad=metrics.grad_errors(out["fused"][1], out["eager"][1]))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-model", action="store_true")
    a = ap.parse_args()
    log = setup_logging()
    res = dict(env=env_info())
    # 1. correctness on assorted shapes (incl. N not a multiple of the block, D not a power of two)
    shapes = [(1, 2, 64, 64), (1, 2, 100, 192), (1, 3, 129, 48), (1, 16, 512, 192)]
    res["shapes"] = {}
    for sh in shapes:
        inp = make_inputs(*sh, seed=sum(sh))
        res["shapes"][str(sh)] = {"fused_fp32": compare(fused_geo_attention, inp, "fp32"),
                                  "eager_fp32": compare(eager_geo_attention, inp, "fp32"),
                                  "fused_tf32": compare(fused_geo_attention, inp, "tf32"),
                                  "eager_tf32": compare(eager_geo_attention, inp, "tf32")}
        r = res["shapes"][str(sh)]
        log.info("%s out rel-L2 vs fp64: fused32 %.2e eager32 %.2e fusedTF %.2e eagerTF %.2e | dQ fused32 %.2e eager32 %.2e",
                 sh, r["fused_fp32"]["output"]["rel_l2"], r["eager_fp32"]["output"]["rel_l2"],
                 r["fused_tf32"]["output"]["rel_l2"], r["eager_tf32"]["output"]["rel_l2"],
                 r["fused_fp32"]["dQ"]["rel_l2"], r["eager_fp32"]["dQ"]["rel_l2"])
    res["reference_gradcheck"] = reference_gradcheck()
    log.info("reference gradcheck: %s", res["reference_gradcheck"])
    # 2. real activations (Plan B weights)
    ns = load_namespace("planB24", "cuda")
    ds = fdata.real(ns)
    real = capture_real_qkv(ns, ds)
    res["real_layers"] = {}
    for L, inp in real.items():
        res["real_layers"][L] = {"fused_fp32": compare(fused_geo_attention, inp, "fp32"),
                                 "eager_fp32": compare(eager_geo_attention, inp, "fp32"),
                                 "fused_tf32": compare(fused_geo_attention, inp, "tf32"),
                                 "eager_tf32": compare(eager_geo_attention, inp, "tf32")}
        r = res["real_layers"][L]
        log.info("real layer %d: out fused32 %.2e eager32 %.2e fusedTF %.2e eagerTF %.2e", L,
                 r["fused_fp32"]["output"]["rel_l2"], r["eager_fp32"]["output"]["rel_l2"],
                 r["fused_tf32"]["output"]["rel_l2"], r["eager_tf32"]["output"]["rel_l2"])
    # 3. performance at the production shape, real layer-0 activations
    res["perf_tf32"] = perf(real[0], "tf32")
    res["perf_fp32"] = perf(real[0], "fp32")
    for prec in ("tf32", "fp32"):
        p = res["perf_" + prec]
        log.info("%s: fwd eager %.3f compiled %.3f fused %.3f | fwd+bwd eager %.3f compiled %.3f fused %.3f ms", prec,
                 *(p[k]["fwd_ms"]["median"] for k in ("eager", "compiled", "fused")),
                 *(p[k]["fwd_bwd_ms"]["median"] for k in ("eager", "compiled", "fused")))
    if not a.skip_model:
        res["model"] = model_level(ns, ds)
        log.info("model fused vs eager (TF32): %s", res["model"])
    os.makedirs(OUT, exist_ok=True)
    write_json(res, OUT / "attention_study.json")


if __name__ == "__main__":
    main()
