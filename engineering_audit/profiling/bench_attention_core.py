"""Isolate the cost of ForeSWE's geo-biased attention core (unchanged upstream code).

Times, at the production shapes of the main encoder (1 x 16 heads x 512 stations x d_k=192):
  core   = MultiHeadAttention.scaled_dot_product_attention  (QK^T, softmax, 1/(1+d), 1/(1+a),
           alpha/beta/gamma mix, softmax, @V)
  mha    = full MultiHeadAttention.forward (W_q/W_k/W_v/W_o projections + core)
  layer  = full EncoderLayer.forward (mha + residual/LayerNorm + FF)
forward-only (no_grad) and forward+backward, with torch.cuda.Event timing.
Must run on a GPU compute node.
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "fore_swe"))
from custom_encoder_model import EncoderLayer  # noqa: E402

assert torch.cuda.is_available()
dev = torch.device("cuda:0")
torch.manual_seed(0)
D, H, N = 3072, 16, 512
layer = EncoderLayer(d_model=D, num_heads=H).to(dev)
mha = layer.self_attn
x = torch.randn(1, N, D, device=dev)
dist = torch.rand(N, N, device=dev) * 1500
ang = torch.rand(N, N, device=dev) * 20
q = torch.randn(1, H, N, D // H, device=dev)
k, v = torch.randn_like(q), torch.randn_like(q)


def timeit(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def fwd_bwd(make_out, leaves):
    def run():
        for t in leaves:
            t.grad = None
        make_out().sum().backward()
    return run


res = {}
with torch.no_grad():
    res["core_fwd_ms"] = timeit(lambda: mha.scaled_dot_product_attention(q, k, v, dist, ang))
    res["mha_fwd_ms"] = timeit(lambda: mha(x, x, x, dist, ang))
    res["layer_fwd_ms"] = timeit(lambda: layer(x, dist, ang))
for t in (q, k, v, x):
    t.requires_grad_(True)
res["core_fwd_bwd_ms"] = timeit(fwd_bwd(lambda: mha.scaled_dot_product_attention(q, k, v, dist, ang), [q, k, v]))
res["layer_fwd_bwd_ms"] = timeit(fwd_bwd(lambda: layer(x, dist, ang), [x]))
# the [1,16,512,512] fp32 score tensor that the eager core materialises several times per call
res["score_tensor_mib"] = H * N * N * 4 / 2**20
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    base = torch.cuda.memory_allocated()
    mha.scaled_dot_product_attention(q, k, v, dist, ang)
    res["core_fwd_peak_extra_mib"] = (torch.cuda.max_memory_allocated() - base) / 2**20
res["core_share_of_layer_fwd_pct"] = 100 * res["core_fwd_ms"] / res["layer_fwd_ms"]
res["core_share_of_layer_fwd_bwd_pct"] = 100 * res["core_fwd_bwd_ms"] / res["layer_fwd_bwd_ms"]
res["gpu"] = torch.cuda.get_device_name(0)
res["torch"] = torch.__version__
res["shapes"] = dict(q=list(q.shape), x=list(x.shape), dist=list(dist.shape))
out = os.path.join(sys.argv[1], "bench_attention_core.json")
with open(out, "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps(res, indent=1))
