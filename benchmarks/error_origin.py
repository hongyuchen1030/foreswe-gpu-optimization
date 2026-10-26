"""Phase 5: where do TF32/BF16 errors originate, and can error-free-transformation-style splitting recover them?

1. Module-wise forward error vs FP64 (real batch, Plan B weights), in execution order.
2. Op-level isolation on captured FP64 activations of encoder layers 0 and 7:
   - GEMM (W_q, K=3072): native FP32/TF32/BF16 vs "inputs rounded, computed in FP64" (pure input-rounding
     error, i.e. what is lost before any accumulation happens) and emulated multi-component splits
     (bf16x3 = hi*hi + hi*lo + lo*hi; tf32x3 likewise) built from stock GEMMs (analysis only: no custom kernel).
   - attention scores QK^T, the custom double-softmax attention core, softmax, LayerNorm.
3. Reductions: MSE loss sum in FP32 vs FP64.
4. Optimizer: FP32 master-weight AdamW update absorption (|update| below half an ulp of the weight).

    python -m benchmarks.error_origin --out DIR
"""
import argparse
import copy
import math
import os

import torch

from foreswe_bench import data as fdata
from foreswe_bench import paths
from foreswe_bench.metrics import tensor_errors
from foreswe_bench.model import build_model, live_named_parameters, load_checkpoint_weights, load_namespace
from foreswe_bench.precision import autocast, make_optimizer, matmul_flags
from foreswe_bench.runtime import env_info, forward_loss, setup_logging, write_json


def round_bf16(x):
    return x.to(torch.bfloat16).to(x.dtype)


def round_tf32(x):
    """Round-to-nearest (ties away) of an fp32 mantissa to 10 bits. Emulation: cuBLAS' internal rounding may differ."""
    f = x.float().contiguous()
    i = f.view(torch.int32)
    i = (i + 0x1000) & ~0x1FFF
    return i.view(torch.float32).to(x.dtype)


def rel(x, ref):
    return tensor_errors(x, ref)["rel_l2"]


def time_ms(fn, iters=20):
    for _ in range(3):
        fn()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def capture(model, ex, precision, names):
    store, hooks = {}, []
    mods = dict(model.named_modules())
    for n in names:
        def hook(m, inp, out, n=n):
            o = out[0] if isinstance(out, tuple) else out
            store[n] = (inp[0].detach() if inp and torch.is_tensor(inp[0]) else None, o.detach())
        hooks.append(mods[n].register_forward_hook(hook))
    with torch.no_grad(), matmul_flags("fp32" if precision == "fp64" else precision):
        out, _, loss = forward_loss(model, ex, "fp32" if precision == "fp64" else precision)
    for h in hooks:
        h.remove()
    return store, out, loss


def gemm_study(x64, lin):
    """x64: [1,512,3072] fp64 input of a Linear; lin: the fp32 nn.Linear."""
    W, bvec = lin.weight.detach(), lin.bias.detach()
    W64, b64, x32 = W.double(), bvec.double(), x64.float()
    ref = x64 @ W64.T + b64
    res = {}
    with matmul_flags("fp32"):
        res["fp32_native"] = rel(x32 @ W.T + bvec, ref)
        res["fp32_time_ms"] = time_ms(lambda: x32 @ W.T + bvec)
    with matmul_flags("tf32"):
        res["tf32_native"] = rel(x32 @ W.T + bvec, ref)
        res["tf32_time_ms"] = time_ms(lambda: x32 @ W.T + bvec)
    with matmul_flags("bf16"), autocast("bf16"):
        res["bf16_native"] = rel((x32 @ W.T + bvec).float(), ref)
        res["bf16_time_ms"] = time_ms(lambda: x32 @ W.T + bvec)
    # pure input rounding, exact (fp64) accumulation
    res["bf16_inputs_rounded_fp64_accum"] = rel(round_bf16(x64) @ round_bf16(W64).T + b64, ref)
    res["tf32_inputs_rounded_fp64_accum"] = rel(round_tf32(x64) @ round_tf32(W64).T + b64, ref)
    # emulated 2-term splits (Dekker/Ogita-style decomposition), 3 low-precision GEMMs, fp32 accumulation.
    # BF16 tensor-core GEMMs accumulate in fp32 but stock torch.matmul rounds the OUTPUT to bf16, which would
    # defeat the split. A bf16 x bf16 product is exact in fp32, so "bf16 inputs, fp32 accumulate, fp32 output"
    # is emulated exactly by an fp32 (TF32-off) GEMM on bf16-valued operands. Error only; no timing claim
    # (cuBLASLt can do bf16-in/fp32-out natively, stock PyTorch cannot).
    xh, Wh = x32.to(torch.bfloat16).float(), W.to(torch.bfloat16).float()
    xl, Wl = (x32 - xh).to(torch.bfloat16).float(), (W - Wh).to(torch.bfloat16).float()
    with matmul_flags("fp32"):
        res["bf16_inputs_fp32_output"] = rel(xh @ Wh.T + bvec, ref)
        res["bf16x3_split_fp32_output_emulated"] = rel(xh @ Wh.T + xh @ Wl.T + xl @ Wh.T + bvec, ref)
    xh32, Wh32 = round_tf32(x32), round_tf32(W)
    xl32, Wl32 = round_tf32(x32 - xh32), round_tf32(W - Wh32)

    def tf32x3():
        return xh32 @ Wh32.T + xh32 @ Wl32.T + xl32 @ Wh32.T + bvec
    with matmul_flags("tf32"):
        res["tf32x3_split_emulated"] = rel(tf32x3(), ref)
        res["tf32x3_time_ms_unfused"] = time_ms(tf32x3)
    return res


def attention_study(layer, x64, dist, ang):
    """Custom geo-biased attention core of one encoder layer on its real fp64 input."""
    att = layer.self_attn
    d = dist.double()
    a = ang.double()
    att64 = copy.deepcopy(att).double()
    with torch.no_grad():
        q64 = att64.split_heads(att64.W_q(x64))
        k64 = att64.split_heads(att64.W_k(x64))
        v64 = att64.split_heads(att64.W_v(x64))
        s_ref = q64 @ k64.transpose(-2, -1) / math.sqrt(att64.d_k)
        core_ref = att64.scaled_dot_product_attention(q64, k64, v64, d, a)
        q32, k32, v32 = q64.float(), k64.float(), v64.float()
        res = {}
        for p in ("fp32", "tf32", "bf16"):
            with matmul_flags(p), autocast(p):
                s = q32 @ k32.transpose(-2, -1) / math.sqrt(att.d_k)
                c = att.scaled_dot_product_attention(q32, k32, v32, dist.float(), ang.float())
            res["scores_" + p] = rel(s.float(), s_ref)
            res["core_" + p] = rel(c.float(), core_ref)
        res["scores_bf16_inputs_rounded_fp64"] = rel(round_bf16(q64) @ round_bf16(k64).transpose(-2, -1) / math.sqrt(att64.d_k), s_ref)
        # softmax given perfectly computed vs bf16-rounded scores (softmax itself runs in fp32 under autocast)
        sm_ref = torch.softmax(s_ref, -1)
        res["softmax_fp32_on_exact_scores"] = rel(torch.softmax(s_ref.float(), -1), sm_ref)
        res["softmax_fp32_on_bf16_rounded_scores"] = rel(torch.softmax(round_bf16(s_ref).float(), -1), sm_ref)
        res["alpha_beta_gamma"] = [att.alpha.item(), att.beta.item(), att.gamma.item()]
    return res


def layernorm_study(layer, x64):
    ln = layer.norm1
    ln64 = copy.deepcopy(ln).double()
    with torch.no_grad():
        ref = ln64(x64)
        return dict(fp32=rel(ln(x64.float()), ref), fp32_on_bf16_rounded_input=rel(ln(round_bf16(x64).float()), ref))


def optimizer_absorption(model, ds, idx):
    """How much of an FP32 AdamW update is lost to rounding when added to FP32 master weights."""
    model.train()
    model.zero_grad(set_to_none=True)
    _, _, loss = forward_loss(model, ds.example(idx), "fp32")
    loss.backward()
    live = [p for _, p in live_named_parameters(model)]
    p0 = [p.detach().double() for p in live]
    opt = make_optimizer(model.parameters(), "fused", 5e-6, 1e-4)
    # replicate the update in fp64 from the same grads (first AdamW step: m=g*(1-b1), v=g^2*(1-b2), bias-corrected)
    lr, wd, eps = 5e-6, 1e-4, 1e-8
    upd64 = [-(lr * wd * x) - lr * p.grad.double() / (p.grad.double().abs() + eps) for x, p in zip(p0, live)]
    opt.step()
    applied = [p.detach().double() - x for p, x in zip(live, p0)]
    num = den = 0.0
    lost = total = 0
    for u, ap, x in zip(upd64, applied, p0):
        num += torch.sum((ap - u) ** 2).item()
        den += torch.sum(u ** 2).item()
        ulp = torch.abs(torch.nextafter(x.float(), torch.full_like(x.float(), math.inf)).double() - x)
        lost += int((u.abs() < 0.5 * ulp).sum().item())
        total += u.numel()
    out = dict(update_rel_l2_error_fp32_master=math.sqrt(num / den), fraction_updates_below_half_ulp=lost / total,
               note="first AdamW step: update magnitude ~lr for every element with |g|>>eps")
    # Is decoupled weight decay (lr*wd = 5e-10 relative) representable at all in FP32 master weights?
    # Same grads + same fresh state, wd=1e-4 vs wd=0: count bit-identical results.
    for impl in ("foreach", "fused"):
        res_wd = []
        for wd in (1e-4, 0.0):
            for p, x in zip(live, p0):
                p.data.copy_(x.float())
            o = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd,
                                  foreach=impl == "foreach", fused=impl == "fused")
            o.step()
            res_wd.append([p.detach().clone() for p in live])
            del o
        same = sum(int(torch.equal(a_, b_)) for a_, b_ in zip(*res_wd))
        elems = sum(int((a_ == b_).sum().item()) for a_, b_ in zip(*res_wd))
        out["weight_decay_effect_" + impl] = dict(
            tensors_bit_identical_with_and_without_wd=same, n_tensors=len(live),
            elements_bit_identical_fraction=elems / total, lr_times_wd=lr * 1e-4,
            fp32_half_ulp_relative=2.0 ** -24)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(paths.RESULTS_DIR / "error_origin"))
    a = ap.parse_args()
    log = setup_logging()
    ns = load_namespace("planB24", "cuda")
    ds = fdata.real(ns)
    model = build_model(ns, "planB24", seed=0)
    meta = load_checkpoint_weights(model, paths.CHECKPOINTS["planB24"])
    model.eval()
    model64 = copy.deepcopy(model).double().eval()
    idx = int(ds.test_issue[2017])
    ex = ds.example(idx)
    ex64 = {k: v.double() for k, v in ex.items()}

    names = []
    for br in ("historical_daily_att", "historical_monthly_att", "historical_yearly_att"):
        names += [br + ".sp_tp_prompt_embed", br + ".prompt_embedding", br + ".variable_attn", br]
    for L in range(8):
        b = "encoder_layers.{}".format(L)
        names += [b + ".self_attn.W_q", b + ".self_attn.W_k", b + ".self_attn.W_v", b + ".self_attn",
                  b + ".norm1", b + ".feed_forward.fc1", b + ".feed_forward", b + ".norm2", b]
    names += ["output_layers", "final_output"]

    ref, out_ref, loss_ref = capture(model64, ex64, "fp64", names)
    modulewise = {}
    for p in ("fp32", "tf32", "bf16"):
        st, out, loss = capture(model, ex, p, names)
        modulewise[p] = [dict(module=n, rel_l2=rel(st[n][1].float(), ref[n][1])) for n in names]
        modulewise[p].append(dict(module="<model output>", rel_l2=rel(out.float(), out_ref)))
        modulewise[p].append(dict(module="<loss>", rel_l2=abs(loss.item() - loss_ref.item()) / abs(loss_ref.item())))
        del st
    res = dict(env=env_info(), batch=idx, checkpoint_meta=meta, modulewise_vs_fp64=modulewise, ops={})
    dist, ang = ex["dist"], ex["ang"]
    for L in (0, 7):
        layer = model.encoder_layers[L]
        x64 = ref["encoder_layers.{}.self_attn.W_q".format(L)][0]
        res["ops"]["layer{}".format(L)] = dict(gemm_Wq_K3072=gemm_study(x64, layer.self_attn.W_q),
                                                gemm_fc2_K2048=gemm_study(ref["encoder_layers.{}.feed_forward.fc1".format(L)][1].clamp(min=0),
                                                                          layer.feed_forward.fc2),
                                                attention=attention_study(layer, x64, dist, ang),
                                                layernorm=layernorm_study(layer, ref["encoder_layers.{}.norm1".format(L)][0]))
        log.info("layer %d GEMM W_q: %s", L, {k: "%.2e" % v for k, v in res["ops"]["layer{}".format(L)]["gemm_Wq_K3072"].items()})
    with torch.no_grad():
        err = (out_ref.reshape(512, -1) - ex64["target"]) ** 2
        s64 = err.sum()
        res["reductions"] = dict(mse_sum_fp32_rel_err=abs(err.float().sum().double() - s64).item() / s64.item(),
                                 n_terms=err.numel())
    del model64, ref
    torch.cuda.empty_cache()
    res["optimizer"] = optimizer_absorption(model, ds, int(ds.train_indices[100]))
    os.makedirs(a.out, exist_ok=True)
    write_json(res, os.path.join(a.out, "error_origin.json"))
    log.info("wrote %s", a.out)


if __name__ == "__main__":
    main()
