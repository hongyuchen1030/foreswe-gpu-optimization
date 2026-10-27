"""Fused geo-biased attention (Triton prototype) for ForeSWE's custom MultiHeadAttention.

Production operator (fore_swe/custom_encoder_model.py, MultiHeadAttention.scaled_dot_product_attention):
    S  = Q K^T / sqrt(d_k)                       [B, H, N, N]
    P1 = softmax(S, -1)
    Z  = alpha * P1 + beta * 1/(1+dist) + gamma * 1/(1+ang)     dist, ang: [N, N] (broadcast over B, H)
    P2 = softmax(Z, -1)
    O  = P2 V
alpha, beta, gamma are trainable scalars (one each per layer). mask is always None in ForeSWE.

Fusion: P1 in [0, 1] and both priors in (0, 1] (dist, ang >= 0), so Z <= c = |alpha|+|beta|+|gamma|.
softmax(Z) = exp(Z - c) / sum exp(Z - c) exactly, so the second softmax needs no row max and can be
accumulated in the same pass as P2 V. Forward = 2 passes over the keys (row stats of S, then
P1 -> Z -> exp -> accumulate O), never materialising an N x N tensor. Backward recomputes from the
saved row statistics (m1, l1, l2) with three kernels:
    B1 rows:    delta1 = alpha * sum_j P1 * dZ ; partial dalpha/dbeta/dgamma
    B2 rows:    dQ = sum_j dS K / sqrt(d)
    B3 columns: dK = sum_i dS^T Q / sqrt(d), dV = sum_i P2^T dO
with dP2 = dO V^T, delta2 = rowsum(dO * O), dZ = P2 (dP2 - delta2), dS = P1 (alpha dZ - delta1).

The eager operator stays available as the reference (`impl="eager"`).
"""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _scores(q, K, k_base, skn, skd, offs_n, offs_d, N, D, sqrt_dk, PREC: tl.constexpr):
    k = tl.load(K + k_base + offs_n[None, :] * skn + offs_d[:, None] * skd,
                mask=(offs_n[None, :] < N) & (offs_d[:, None] < D), other=0.0)
    return tl.dot(q, k, input_precision=PREC) / sqrt_dk


@triton.jit
def _fwd_kernel(Q, K, V, DIST, ANG, ABG, O, M1, L1, L2,
                sqb, sqh, sqn, sqd, skb, skh, skn, skd, svb, svh, svn, svd, sob, soh, son, sod, sdn, sdm,
                H, N, NS, D, sqrt_dk,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, PREC: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n0 = tl.arange(0, BLOCK_N)
    row_ok = offs_m < N
    q = tl.load(Q + b * sqb + h * sqh + offs_m[:, None] * sqn + offs_d[None, :] * sqd,
                mask=row_ok[:, None] & (offs_d[None, :] < D), other=0.0)
    alpha = tl.load(ABG)
    beta = tl.load(ABG + 1)
    gamma = tl.load(ABG + 2)
    c = tl.abs(alpha) + tl.abs(beta) + tl.abs(gamma)
    k_base = b * skb + h * skh
    # pass 1: row max / sum of S
    m1 = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l1 = tl.zeros([BLOCK_M], tl.float32)
    for start in range(0, N, BLOCK_N):
        offs_n = start + offs_n0
        s = _scores(q, K, k_base, skn, skd, offs_n, offs_d, N, D, sqrt_dk, PREC)
        s = tl.where(offs_n[None, :] < N, s, float("-inf"))
        m_new = tl.maximum(m1, tl.max(s, 1))
        l1 = l1 * tl.exp(m1 - m_new) + tl.sum(tl.exp(s - m_new[:, None]), 1)
        m1 = m_new
    # pass 2: P1 -> Z -> exp(Z - c) -> accumulate P2 V
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
    l2 = tl.zeros([BLOCK_M], tl.float32)
    for start in range(0, N, BLOCK_N):
        offs_n = start + offs_n0
        col_ok = offs_n < N
        s = _scores(q, K, k_base, skn, skd, offs_n, offs_d, N, D, sqrt_dk, PREC)
        p = tl.exp(s - m1[:, None]) / l1[:, None]
        tile = offs_m[:, None] * sdn + offs_n[None, :] * sdm
        tmask = row_ok[:, None] & col_ok[None, :]
        dd = tl.load(DIST + tile, mask=tmask, other=0.0)
        aa = tl.load(ANG + tile, mask=tmask, other=0.0)
        z = alpha * p + beta * (1.0 / (1.0 + dd)) + gamma * (1.0 / (1.0 + aa))
        e = tl.where(col_ok[None, :], tl.exp(z - c), 0.0)
        l2 += tl.sum(e, 1)
        v = tl.load(V + b * svb + h * svh + offs_n[:, None] * svn + offs_d[None, :] * svd,
                    mask=col_ok[:, None] & (offs_d[None, :] < D), other=0.0)
        acc += tl.dot(e, v, input_precision=PREC)
    o = acc / l2[:, None]
    tl.store(O + b * sob + h * soh + offs_m[:, None] * son + offs_d[None, :] * sod, o,
             mask=row_ok[:, None] & (offs_d[None, :] < D))
    stat = pid_bh * NS + offs_m
    tl.store(M1 + stat, m1, mask=row_ok)
    tl.store(L1 + stat, l1, mask=row_ok)
    tl.store(L2 + stat, l2, mask=row_ok)


@triton.jit
def _recompute(q, do, K, V, k_base, v_base, skn, skd, svn, svd, DIST, ANG, sdn, sdm, offs_m, offs_n, offs_d,
               m1, l1, l2, delta2, alpha, beta, gamma, c, N, D, sqrt_dk, PREC: tl.constexpr):
    """Recompute P1, prior terms, P2 and dZ for one [BLOCK_M, BLOCK_N] tile."""
    col_ok = offs_n < N
    row_ok = offs_m < N
    s = _scores(q, K, k_base, skn, skd, offs_n, offs_d, N, D, sqrt_dk, PREC)
    p = tl.where(col_ok[None, :], tl.exp(s - m1[:, None]) / l1[:, None], 0.0)
    tile = offs_m[:, None] * sdn + offs_n[None, :] * sdm
    tmask = row_ok[:, None] & col_ok[None, :]
    dprior = 1.0 / (1.0 + tl.load(DIST + tile, mask=tmask, other=0.0))
    aprior = 1.0 / (1.0 + tl.load(ANG + tile, mask=tmask, other=0.0))
    z = alpha * p + beta * dprior + gamma * aprior
    p2 = tl.where(col_ok[None, :], tl.exp(z - c) / l2[:, None], 0.0)
    vt = tl.load(V + v_base + offs_n[None, :] * svn + offs_d[:, None] * svd,
                 mask=col_ok[None, :] & (offs_d[:, None] < D), other=0.0)
    dp2 = tl.dot(do, vt, input_precision=PREC)
    dz = tl.where(tmask, p2 * (dp2 - delta2[:, None]), 0.0)
    return p, p2, dz, dprior, aprior


@triton.jit
def _bwd_rows_kernel(Q, K, V, DIST, ANG, ABG, DO, M1, L1, L2, DELTA2, DELTA1, PART, DQ,
                     sqb, sqh, sqn, sqd, skb, skh, skn, skd, svb, svh, svn, svd, sdb, sdh, sdn_, sdd, sdn, sdm,
                     H, N, NS, D, sqrt_dk, MODE: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, PREC: tl.constexpr):
    """MODE 0 (B0, optional): delta2 = rowsum(P2 * dP2) from the recomputed tiles, as eager softmax-backward
    computes it (default uses the equal-in-exact-arithmetic rowsum(dO * O); measured: no accuracy difference).
    MODE 1 (B1): delta1 + coefficient partials.  MODE 2 (B2): dQ."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n0 = tl.arange(0, BLOCK_N)
    row_ok = offs_m < N
    dmask = row_ok[:, None] & (offs_d[None, :] < D)
    q = tl.load(Q + b * sqb + h * sqh + offs_m[:, None] * sqn + offs_d[None, :] * sqd, mask=dmask, other=0.0)
    do = tl.load(DO + b * sdb + h * sdh + offs_m[:, None] * sdn_ + offs_d[None, :] * sdd, mask=dmask, other=0.0)
    stat = pid_bh * NS + offs_m
    m1 = tl.load(M1 + stat)
    l1 = tl.load(L1 + stat)
    l2 = tl.load(L2 + stat)
    delta2 = tl.load(DELTA2 + stat)
    alpha = tl.load(ABG)
    beta = tl.load(ABG + 1)
    gamma = tl.load(ABG + 2)
    c = tl.abs(alpha) + tl.abs(beta) + tl.abs(gamma)
    k_base = b * skb + h * skh
    v_base = b * svb + h * svh
    if MODE == 0:
        d2 = tl.zeros([BLOCK_M], tl.float32)
        for start in range(0, N, BLOCK_N):
            offs_n = start + offs_n0
            p, p2, dz, dpr, apr = _recompute(q, do, K, V, k_base, v_base, skn, skd, svn, svd, DIST, ANG, sdn, sdm,
                                             offs_m, offs_n, offs_d, m1, l1, l2, delta2, alpha, beta, gamma, c,
                                             N, D, sqrt_dk, PREC)
            # dz with delta2 = 0 is P2 * dP2 (masked)
            d2 += tl.sum(dz, 1)
        tl.store(DELTA2 + stat, d2, mask=row_ok)
    elif MODE == 2:
        delta1 = tl.load(DELTA1 + stat)
        dq = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
        for start in range(0, N, BLOCK_N):
            offs_n = start + offs_n0
            p, p2, dz, dpr, apr = _recompute(q, do, K, V, k_base, v_base, skn, skd, svn, svd, DIST, ANG, sdn, sdm,
                                             offs_m, offs_n, offs_d, m1, l1, l2, delta2, alpha, beta, gamma, c,
                                             N, D, sqrt_dk, PREC)
            ds = p * (alpha * dz - delta1[:, None])
            k = tl.load(K + k_base + offs_n[:, None] * skn + offs_d[None, :] * skd,
                        mask=(offs_n[:, None] < N) & (offs_d[None, :] < D), other=0.0)
            dq += tl.dot(ds, k, input_precision=PREC)
        tl.store(DQ + b * sqb + h * sqh + offs_m[:, None] * sqn + offs_d[None, :] * sqd, dq / sqrt_dk, mask=dmask)
    else:
        d1 = tl.zeros([BLOCK_M], tl.float32)
        ga = tl.zeros([BLOCK_M], tl.float32)
        gb = tl.zeros([BLOCK_M], tl.float32)
        gc = tl.zeros([BLOCK_M], tl.float32)
        for start in range(0, N, BLOCK_N):
            offs_n = start + offs_n0
            p, p2, dz, dpr, apr = _recompute(q, do, K, V, k_base, v_base, skn, skd, svn, svd, DIST, ANG, sdn, sdm,
                                             offs_m, offs_n, offs_d, m1, l1, l2, delta2, alpha, beta, gamma, c,
                                             N, D, sqrt_dk, PREC)
            ga += tl.sum(dz * p, 1)
            gb += tl.sum(dz * dpr, 1)
            gc += tl.sum(dz * apr, 1)
        d1 = alpha * ga
        tl.store(DELTA1 + stat, d1, mask=row_ok)
        prog = pid_bh * tl.num_programs(0) + pid_m
        tl.store(PART + prog * 3 + 0, tl.sum(tl.where(row_ok, ga, 0.0), 0))
        tl.store(PART + prog * 3 + 1, tl.sum(tl.where(row_ok, gb, 0.0), 0))
        tl.store(PART + prog * 3 + 2, tl.sum(tl.where(row_ok, gc, 0.0), 0))


@triton.jit
def _bwd_cols_kernel(Q, K, V, DIST, ANG, ABG, DO, M1, L1, L2, DELTA2, DELTA1, DK, DV,
                     sqb, sqh, sqn, sqd, skb, skh, skn, skd, svb, svh, svn, svd, sdb, sdh, sdn_, sdd, sdn, sdm,
                     H, N, NS, D, sqrt_dk,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, PREC: tl.constexpr):
    """B3: one program per key block j; dK_j = sum_i dS^T Q / sqrt(d), dV_j = sum_i P2^T dO."""
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    offs_m0 = tl.arange(0, BLOCK_M)
    col_ok = offs_n < N
    alpha = tl.load(ABG)
    beta = tl.load(ABG + 1)
    gamma = tl.load(ABG + 2)
    c = tl.abs(alpha) + tl.abs(beta) + tl.abs(gamma)
    k_base = b * skb + h * skh
    v_base = b * svb + h * svh
    dk = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)
    for start in range(0, N, BLOCK_M):
        offs_m = start + offs_m0
        row_ok = offs_m < N
        dmask = row_ok[:, None] & (offs_d[None, :] < D)
        q = tl.load(Q + b * sqb + h * sqh + offs_m[:, None] * sqn + offs_d[None, :] * sqd, mask=dmask, other=0.0)
        do = tl.load(DO + b * sdb + h * sdh + offs_m[:, None] * sdn_ + offs_d[None, :] * sdd, mask=dmask, other=0.0)
        stat = pid_bh * NS + offs_m
        m1 = tl.load(M1 + stat)
        l1 = tl.load(L1 + stat)
        l2 = tl.load(L2 + stat)
        delta2 = tl.load(DELTA2 + stat)
        delta1 = tl.load(DELTA1 + stat)
        p, p2, dz, dpr, apr = _recompute(q, do, K, V, k_base, v_base, skn, skd, svn, svd, DIST, ANG, sdn, sdm,
                                         offs_m, offs_n, offs_d, m1, l1, l2, delta2, alpha, beta, gamma, c,
                                         N, D, sqrt_dk, PREC)
        p2 = tl.where(row_ok[:, None], p2, 0.0)
        ds = tl.where(row_ok[:, None], p * (alpha * dz - delta1[:, None]), 0.0)
        dv += tl.dot(tl.trans(p2), do, input_precision=PREC)
        dk += tl.dot(tl.trans(ds), q, input_precision=PREC)
    nmask = col_ok[:, None] & (offs_d[None, :] < D)
    tl.store(DK + b * skb + h * skh + offs_n[:, None] * skn + offs_d[None, :] * skd, dk / sqrt_dk, mask=nmask)
    tl.store(DV + b * svb + h * svh + offs_n[:, None] * svn + offs_d[None, :] * svd, dv, mask=nmask)


CFG = dict(BLOCK_M=16, BLOCK_N=32, num_warps=4, num_stages=2,  # best of the sweep in FUSION_FEASIBILITY_REPORT.md
           delta2="do_o")  # "do_o": delta2 = rowsum(dO*O) (FlashAttention identity); "recompute": extra B0 pass
                           # (tested: no accuracy gain on real layers, see FUSION_FEASIBILITY_REPORT.md)


def _stages(block_d):
    # D=192 pads to a 256-wide tile; two pipeline stages then exceed the A100's 163 KB shared memory
    return 1 if block_d >= 256 else CFG["num_stages"]


def _padded_rows(N):
    blk = max(CFG["BLOCK_M"], CFG["BLOCK_N"])
    return triton.cdiv(N, blk) * blk


def _prec():
    return "tf32" if torch.backends.cuda.matmul.allow_tf32 else "ieee"


class FusedGeoAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, dist, ang, alpha, beta, gamma):
        B, H, N, D = q.shape
        dist = dist.contiguous().float()
        ang = ang.contiguous().float()
        abg = torch.cat([alpha.reshape(1), beta.reshape(1), gamma.reshape(1)]).float().contiguous()
        o = torch.empty((B, H, N, D), device=q.device, dtype=torch.float32)
        # per-row statistics, padded to whole blocks with safe values so kernels load them unmasked
        NS = _padded_rows(N)
        m1 = torch.zeros((B * H, NS), device=q.device, dtype=torch.float32)
        l1, l2 = torch.ones_like(m1), torch.ones_like(m1)
        BD = triton.next_power_of_2(D)
        grid = (triton.cdiv(N, CFG["BLOCK_M"]), B * H)
        prec = _prec()
        _fwd_kernel[grid](q, k, v, dist, ang, abg, o, m1, l1, l2, *q.stride(), *k.stride(), *v.stride(), *o.stride(),
                          *dist.stride(), H, N, NS, D, math.sqrt(D), BLOCK_M=CFG["BLOCK_M"], BLOCK_N=CFG["BLOCK_N"],
                          BLOCK_D=BD, PREC=prec, num_warps=CFG["num_warps"], num_stages=_stages(BD))
        ctx.save_for_backward(q, k, v, dist, ang, abg, o, m1, l1, l2)
        ctx.prec = prec
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, dist, ang, abg, o, m1, l1, l2 = ctx.saved_tensors
        B, H, N, D = q.shape
        do = do.float()
        NS = m1.shape[1]
        delta2 = torch.zeros_like(m1)   # "recompute" mode fills it in B0 (MODE 0) from P2 * dP2
        if CFG["delta2"] == "do_o":     # default: FlashAttention identity rowsum(dO * O)
            delta2[:, :N] = (do * o).sum(-1).reshape(B * H, N)
        delta1 = torch.zeros_like(m1)
        BM, BN, BD = CFG["BLOCK_M"], CFG["BLOCK_N"], triton.next_power_of_2(D)
        nprog_m = triton.cdiv(N, BM)
        part = torch.zeros((B * H * nprog_m, 3), device=q.device, dtype=torch.float32)
        common = (H, N, NS, D, math.sqrt(D))
        kw = dict(BLOCK_M=BM, BLOCK_N=BN, BLOCK_D=BD, PREC=ctx.prec, num_warps=CFG["num_warps"],
                  num_stages=_stages(BD))
        args = (*q.stride(), *k.stride(), *v.stride(), *do.stride(), *dist.stride())
        # gradients are written with their input's strides (Q/K/V are transposed views in production)
        dq_q = torch.empty_strided(q.shape, q.stride(), device=q.device, dtype=torch.float32)
        modes = (1, 2) if CFG["delta2"] == "do_o" else (0, 1, 2)
        for mode in modes:
            _bwd_rows_kernel[(nprog_m, B * H)](q, k, v, dist, ang, abg, do, m1, l1, l2, delta2, delta1, part, dq_q,
                                               *args, *common, MODE=mode, **kw)
        dk_k = torch.empty_strided(k.shape, k.stride(), device=k.device, dtype=torch.float32)
        dv_v = torch.empty_strided(v.shape, v.stride(), device=v.device, dtype=torch.float32)
        _bwd_cols_kernel[(triton.cdiv(N, BN), B * H)](q, k, v, dist, ang, abg, do, m1, l1, l2, delta2, delta1,
                                                      dk_k, dv_v, *args, *common, **kw)
        g = part.sum(0)
        return dq_q, dk_k, dv_v, None, None, g[0].reshape(1), g[1].reshape(1), g[2].reshape(1)


def fused_geo_attention(q, k, v, dist, ang, alpha, beta, gamma):
    return FusedGeoAttention.apply(q, k, v, dist, ang, alpha, beta, gamma)


def eager_geo_attention(q, k, v, dist, ang, alpha, beta, gamma):
    """Verbatim production math (custom_encoder_model.MultiHeadAttention.scaled_dot_product_attention)."""
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    attn_scores = torch.softmax(attn_scores, dim=-1)
    distances = 1 / (1 + dist)
    angularities = 1 / (1 + ang)
    new_score = alpha * attn_scores + beta * distances + gamma * angularities
    modified_attn_weights = torch.softmax(new_score, dim=-1)
    return torch.matmul(modified_attn_weights, v)


def patch_model(model, impl="fused"):
    """Swap the attention core of every encoder layer; impl='eager' restores the production method."""
    for layer in model.encoder_layers:
        att = layer.self_attn
        if impl == "eager":
            att.__dict__.pop("scaled_dot_product_attention", None)
            continue

        def core(Q, K, V, distances, angularities, mask=None, _att=att):
            assert mask is None, "ForeSWE never passes a mask"
            return fused_geo_attention(Q, K, V, distances, angularities, _att.alpha, _att.beta, _att.gamma)
        att.scaled_dot_product_attention = core
    return model
