"""Fused geo-biased attention, v2: D split into 64-wide chunks instead of padding D=192 to 256.

Same operator, algorithm and autograd contract as foreswe_bench/geo_attention.py (kept unchanged as the
benchmark reference). Differences:
  * head dimension processed as NC <= 3 chunks of DC = 64 (no power-of-two padding: D=192 = 3 x 64);
    QK^T and dO V^T accumulate three partial tl.dot products, the D-wide accumulators (O, dQ, dK, dV)
    are three [*, 64] tiles
  * independent tile configs for the forward, the row-parallel backward and the column-parallel backward
  * optional DEBUG stores of the backward's per-tile intermediates (P1, dZ, dS) for numerical diagnosis
  * MODE/ACC switches for the saturated-softmax accuracy work (see SATURATED_SOFTMAX_BACKWARD_DEBUG.md)
"""
import math

import torch
import triton
import triton.language as tl

DC = 64  # head-dimension chunk


@triton.jit
def _ld_rows(P, base, offs_r, sr, offs_c, sc, R, D, c0: tl.constexpr):
    """[rows, DC] tile of columns c0 .. c0+DC."""
    cols = c0 + offs_c
    return tl.load(P + base + offs_r[:, None] * sr + cols[None, :] * sc,
                   mask=(offs_r[:, None] < R) & (cols[None, :] < D), other=0.0)


@triton.jit
def _ld_cols(P, base, offs_r, sr, offs_c, sc, R, D, c0: tl.constexpr):
    """[DC, rows] transposed tile of columns c0 .. c0+DC."""
    cols = c0 + offs_c
    return tl.load(P + base + offs_r[None, :] * sr + cols[:, None] * sc,
                   mask=(offs_r[None, :] < R) & (cols[:, None] < D), other=0.0)


@triton.jit
def _rn_tf32(x):
    """Round an fp32 tile to TF32 (10-bit mantissa) with round-to-nearest (ties away), like cuBLAS' operand
    conversion; the tensor core's own conversion of an already-TF32 value is then exact."""
    i = x.to(tl.int32, bitcast=True)
    i = (i + 0x1000) & -8192  # -8192 == 0xFFFFE000: clear the 13 low mantissa bits
    return i.to(tl.float32, bitcast=True)


@triton.jit
def _mm(a, b, PREC: tl.constexpr):
    """tl.dot with the kernel's precision mode; "tf32rn" = TF32 tensor cores on round-to-nearest operands."""
    if PREC == "tf32rn":
        return tl.dot(_rn_tf32(a), _rn_tf32(b), input_precision="tf32")
    else:
        return tl.dot(a, b, input_precision=PREC)


@triton.jit
def _dot3(a0, a1, a2, b0, b1, b2, NC: tl.constexpr, PREC: tl.constexpr):
    r = _mm(a0, b0, PREC)
    if NC > 1:
        r += _mm(a1, b1, PREC)
    if NC > 2:
        r += _mm(a2, b2, PREC)
    return r


@triton.jit
def _scores3(q0, q1, q2, K, k_base, skn, skd, offs_n, offs_c, N, D, sqrt_dk,
             DCc: tl.constexpr, NC: tl.constexpr, PREC: tl.constexpr, SROUND: tl.constexpr):
    k0 = _ld_cols(K, k_base, offs_n, skn, offs_c, skd, N, D, 0)
    k1 = _ld_cols(K, k_base, offs_n, skn, offs_c, skd, N, D, DCc)
    k2 = _ld_cols(K, k_base, offs_n, skn, offs_c, skd, N, D, 2 * DCc)
    acc = _dot3(q0, q1, q2, k0, k1, k2, NC, PREC)
    if SROUND:  # correctly rounded scaling: S is materialised identically in every kernel (no FMA contraction)
        return tl.math.div_rn(acc, sqrt_dk)
    return acc / sqrt_dk


@triton.jit
def _fwd_kernel(Q, K, V, DIST, ANG, ABG, O, M1, L1, L2,
                sqb, sqh, sqn, sqd, skb, skh, skn, skd, svb, svh, svn, svd, sob, soh, son, sod, sdn, sdm,
                H, N, NS, D, sqrt_dk,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, DCc: tl.constexpr, NC: tl.constexpr,
                PREC: tl.constexpr, SPREC: tl.constexpr, SROUND: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, DCc)
    offs_n0 = tl.arange(0, BLOCK_N)
    row_ok = offs_m < N
    qb = b * sqb + h * sqh
    q0 = _ld_rows(Q, qb, offs_m, sqn, offs_c, sqd, N, D, 0)
    q1 = _ld_rows(Q, qb, offs_m, sqn, offs_c, sqd, N, D, DCc)
    q2 = _ld_rows(Q, qb, offs_m, sqn, offs_c, sqd, N, D, 2 * DCc)
    alpha = tl.load(ABG)
    beta = tl.load(ABG + 1)
    gamma = tl.load(ABG + 2)
    c = tl.abs(alpha) + tl.abs(beta) + tl.abs(gamma)
    k_base = b * skb + h * skh
    v_base = b * svb + h * svh
    m1 = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l1 = tl.zeros([BLOCK_M], tl.float32)
    for start in range(0, N, BLOCK_N):
        offs_n = start + offs_n0
        s = _scores3(q0, q1, q2, K, k_base, skn, skd, offs_n, offs_c, N, D, sqrt_dk, DCc, NC, SPREC, SROUND)
        s = tl.where(offs_n[None, :] < N, s, float("-inf"))
        m_new = tl.maximum(m1, tl.max(s, 1))
        l1 = l1 * tl.exp(m1 - m_new) + tl.sum(tl.exp(s - m_new[:, None]), 1)
        m1 = m_new
    acc0 = tl.zeros([BLOCK_M, DCc], tl.float32)
    acc1 = tl.zeros([BLOCK_M, DCc], tl.float32)
    acc2 = tl.zeros([BLOCK_M, DCc], tl.float32)
    l2 = tl.zeros([BLOCK_M], tl.float32)
    for start in range(0, N, BLOCK_N):
        offs_n = start + offs_n0
        col_ok = offs_n < N
        s = _scores3(q0, q1, q2, K, k_base, skn, skd, offs_n, offs_c, N, D, sqrt_dk, DCc, NC, SPREC, SROUND)
        p = tl.exp(s - m1[:, None]) / l1[:, None]
        tile = offs_m[:, None] * sdn + offs_n[None, :] * sdm
        tmask = row_ok[:, None] & col_ok[None, :]
        z = alpha * p + beta * (1.0 / (1.0 + tl.load(DIST + tile, mask=tmask, other=0.0))) \
            + gamma * (1.0 / (1.0 + tl.load(ANG + tile, mask=tmask, other=0.0)))
        e = tl.where(col_ok[None, :], tl.exp(z - c), 0.0)
        l2 += tl.sum(e, 1)
        acc0 += _mm(e, _ld_rows(V, v_base, offs_n, svn, offs_c, svd, N, D, 0), PREC)
        if NC > 1:
            acc1 += _mm(e, _ld_rows(V, v_base, offs_n, svn, offs_c, svd, N, D, DCc), PREC)
        if NC > 2:
            acc2 += _mm(e, _ld_rows(V, v_base, offs_n, svn, offs_c, svd, N, D, 2 * DCc), PREC)
    ob = b * sob + h * soh
    for_c = offs_c
    tl.store(O + ob + offs_m[:, None] * son + for_c[None, :] * sod, acc0 / l2[:, None],
             mask=row_ok[:, None] & (for_c[None, :] < D))
    if NC > 1:
        tl.store(O + ob + offs_m[:, None] * son + (DCc + for_c)[None, :] * sod, acc1 / l2[:, None],
                 mask=row_ok[:, None] & ((DCc + for_c)[None, :] < D))
    if NC > 2:
        tl.store(O + ob + offs_m[:, None] * son + (2 * DCc + for_c)[None, :] * sod, acc2 / l2[:, None],
                 mask=row_ok[:, None] & ((2 * DCc + for_c)[None, :] < D))
    stat = pid_bh * NS + offs_m
    tl.store(M1 + stat, m1, mask=row_ok)
    tl.store(L1 + stat, l1, mask=row_ok)
    tl.store(L2 + stat, l2, mask=row_ok)


@triton.jit
def _recompute3(q0, q1, q2, do0, do1, do2, K, V, k_base, v_base, skn, skd, svn, svd, DIST, ANG, sdn, sdm,
                offs_m, offs_n, offs_c, m1, l1, l2, delta2, alpha, beta, gamma, c, N, D, sqrt_dk,
                DCc: tl.constexpr, NC: tl.constexpr, PREC: tl.constexpr, SPREC: tl.constexpr, SROUND: tl.constexpr):
    col_ok = offs_n < N
    row_ok = offs_m < N
    s = _scores3(q0, q1, q2, K, k_base, skn, skd, offs_n, offs_c, N, D, sqrt_dk, DCc, NC, SPREC, SROUND)
    p = tl.where(col_ok[None, :], tl.exp(s - m1[:, None]) / l1[:, None], 0.0)
    tile = offs_m[:, None] * sdn + offs_n[None, :] * sdm
    tmask = row_ok[:, None] & col_ok[None, :]
    dpr = 1.0 / (1.0 + tl.load(DIST + tile, mask=tmask, other=0.0))
    apr = 1.0 / (1.0 + tl.load(ANG + tile, mask=tmask, other=0.0))
    z = alpha * p + beta * dpr + gamma * apr
    p2 = tl.where(col_ok[None, :], tl.exp(z - c) / l2[:, None], 0.0)
    v0 = _ld_cols(V, v_base, offs_n, svn, offs_c, svd, N, D, 0)
    v1 = _ld_cols(V, v_base, offs_n, svn, offs_c, svd, N, D, DCc)
    v2 = _ld_cols(V, v_base, offs_n, svn, offs_c, svd, N, D, 2 * DCc)
    dp2 = _dot3(do0, do1, do2, v0, v1, v2, NC, PREC)
    dz = tl.where(tmask, p2 * (dp2 - delta2[:, None]), 0.0)
    return s, p, p2, dz, dpr, apr


@triton.jit
def _bwd_rows_kernel(Q, K, V, DIST, ANG, ABG, DO, M1, L1, L2, DELTA2, DELTA1, RS, PART, DQ, DBG_P, DBG_DZ, DBG_DS,
                     sqb, sqh, sqn, sqd, skb, skh, skn, skd, svb, svh, svn, svd, sdb, sdh, sdn_, sdd, sdn, sdm,
                     H, N, NS, D, sqrt_dk, MODE: tl.constexpr, DEBUG: tl.constexpr, ACC: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, DCc: tl.constexpr, NC: tl.constexpr,
                     PREC: tl.constexpr, SPREC: tl.constexpr, SROUND: tl.constexpr):
    """MODE 1: delta1 + d(alpha,beta,gamma) partials.  MODE 2: dQ (+ optional DEBUG intermediates).
    ACC selects the dS formulation (0 = production p*(alpha*dz - delta1); see _ds)."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, DCc)
    offs_n0 = tl.arange(0, BLOCK_N)
    row_ok = offs_m < N
    qb = b * sqb + h * sqh
    db = b * sdb + h * sdh
    q0 = _ld_rows(Q, qb, offs_m, sqn, offs_c, sqd, N, D, 0)
    q1 = _ld_rows(Q, qb, offs_m, sqn, offs_c, sqd, N, D, DCc)
    q2 = _ld_rows(Q, qb, offs_m, sqn, offs_c, sqd, N, D, 2 * DCc)
    do0 = _ld_rows(DO, db, offs_m, sdn_, offs_c, sdd, N, D, 0)
    do1 = _ld_rows(DO, db, offs_m, sdn_, offs_c, sdd, N, D, DCc)
    do2 = _ld_rows(DO, db, offs_m, sdn_, offs_c, sdd, N, D, 2 * DCc)
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
        # optional B0: delta2 = rowsum(P2 * dP2) from the recomputed tiles (DELTA2 enters as 0, so dz = P2*dP2),
        # making rowsum(dZ) cancel exactly as in eager softmax-backward
        d2 = tl.zeros([BLOCK_M], tl.float32)
        for start in range(0, N, BLOCK_N):
            offs_n = start + offs_n0
            s, p, p2, dz, dpr, apr = _recompute3(q0, q1, q2, do0, do1, do2, K, V, k_base, v_base, skn, skd, svn,
                                                 svd, DIST, ANG, sdn, sdm, offs_m, offs_n, offs_c, m1, l1, l2,
                                                 delta2, alpha, beta, gamma, c, N, D, sqrt_dk, DCc, NC, PREC, SPREC, SROUND)
            d2 += tl.sum(dz, 1)
        tl.store(DELTA2 + stat, d2, mask=row_ok)
    elif MODE == 2:
        delta1 = tl.load(DELTA1 + stat)
        rs = tl.load(RS + stat)
        dq0 = tl.zeros([BLOCK_M, DCc], tl.float32)
        dq1 = tl.zeros([BLOCK_M, DCc], tl.float32)
        dq2 = tl.zeros([BLOCK_M, DCc], tl.float32)
        for start in range(0, N, BLOCK_N):
            offs_n = start + offs_n0
            s, p, p2, dz, dpr, apr = _recompute3(q0, q1, q2, do0, do1, do2, K, V, k_base, v_base, skn, skd, svn,
                                                 svd, DIST, ANG, sdn, sdm, offs_m, offs_n, offs_c, m1, l1, l2,
                                                 delta2, alpha, beta, gamma, c, N, D, sqrt_dk, DCc, NC, PREC, SPREC, SROUND)
            if ACC == 1:
                p = p / rs[:, None]  # renormalise the recomputed P1 with its own row sum (from B1)
            ds = _ds(p, dz, alpha, delta1, ACC)
            if DEBUG:
                dbg = pid_bh * NS * NS + offs_m[:, None] * NS + offs_n[None, :]
                dmask = row_ok[:, None] & (offs_n[None, :] < N)
                tl.store(DBG_P + dbg, p, mask=dmask)
                tl.store(DBG_DZ + dbg, dz, mask=dmask)
                tl.store(DBG_DS + dbg, ds, mask=dmask)
            dq0 += _mm(ds, _ld_rows(K, k_base, offs_n, skn, offs_c, skd, N, D, 0), PREC)
            if NC > 1:
                dq1 += _mm(ds, _ld_rows(K, k_base, offs_n, skn, offs_c, skd, N, D, DCc), PREC)
            if NC > 2:
                dq2 += _mm(ds, _ld_rows(K, k_base, offs_n, skn, offs_c, skd, N, D, 2 * DCc), PREC)
        cols = offs_c
        tl.store(DQ + qb + offs_m[:, None] * sqn + cols[None, :] * sqd, dq0 / sqrt_dk,
                 mask=row_ok[:, None] & (cols[None, :] < D))
        if NC > 1:
            tl.store(DQ + qb + offs_m[:, None] * sqn + (DCc + cols)[None, :] * sqd, dq1 / sqrt_dk,
                     mask=row_ok[:, None] & ((DCc + cols)[None, :] < D))
        if NC > 2:
            tl.store(DQ + qb + offs_m[:, None] * sqn + (2 * DCc + cols)[None, :] * sqd, dq2 / sqrt_dk,
                     mask=row_ok[:, None] & ((2 * DCc + cols)[None, :] < D))
    else:
        ga = tl.zeros([BLOCK_M], tl.float32)
        # d(alpha, beta, gamma) are global sums with heavy cancellation (rowsum(dZ) = 0): accumulate in FP64
        ca = tl.zeros([BLOCK_M], tl.float64)
        cb = tl.zeros([BLOCK_M], tl.float64)
        cc = tl.zeros([BLOCK_M], tl.float64)
        sp = tl.zeros([BLOCK_M], tl.float32)
        for start in range(0, N, BLOCK_N):
            offs_n = start + offs_n0
            s, p, p2, dz, dpr, apr = _recompute3(q0, q1, q2, do0, do1, do2, K, V, k_base, v_base, skn, skd, svn,
                                                 svd, DIST, ANG, sdn, sdm, offs_m, offs_n, offs_c, m1, l1, l2,
                                                 delta2, alpha, beta, gamma, c, N, D, sqrt_dk, DCc, NC, PREC, SPREC, SROUND)
            ga += tl.sum(dz * p, 1)
            dz64 = dz.to(tl.float64)
            ca += tl.sum(dz64 * p.to(tl.float64), 1)
            cb += tl.sum(dz64 * dpr.to(tl.float64), 1)
            cc += tl.sum(dz64 * apr.to(tl.float64), 1)
            sp += tl.sum(p, 1)
        tl.store(RS + stat, sp, mask=row_ok)
        if ACC == 1:
            ga = ga / sp  # rowsum(q * dZ) with q = P1 / rowsum(P1): consistent with the renormalised P1
            ca = ca / sp.to(tl.float64)
        tl.store(DELTA1 + stat, alpha * ga, mask=row_ok)
        prog = pid_bh * tl.num_programs(0) + pid_m
        tl.store(PART + prog * 3 + 0, tl.sum(tl.where(row_ok, ca, 0.0), 0))
        tl.store(PART + prog * 3 + 1, tl.sum(tl.where(row_ok, cb, 0.0), 0))
        tl.store(PART + prog * 3 + 2, tl.sum(tl.where(row_ok, cc, 0.0), 0))


@triton.jit
def _ds(p, dz, alpha, delta1, ACC: tl.constexpr):
    """dS = P1 * (alpha*dZ - delta1). Callers pass the renormalised P1 when ACC == 1."""
    return p * (alpha * dz - delta1[:, None])


@triton.jit
def _bwd_cols_kernel(Q, K, V, DIST, ANG, ABG, DO, M1, L1, L2, DELTA2, DELTA1, RS, DK, DV,
                     sqb, sqh, sqn, sqd, skb, skh, skn, skd, svb, svh, svn, svd, sdb, sdh, sdn_, sdd, sdn, sdm,
                     H, N, NS, D, sqrt_dk, ACC: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, DCc: tl.constexpr, NC: tl.constexpr,
                     PREC: tl.constexpr, SPREC: tl.constexpr, SROUND: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_c = tl.arange(0, DCc)
    offs_m0 = tl.arange(0, BLOCK_M)
    col_ok = offs_n < N
    alpha = tl.load(ABG)
    beta = tl.load(ABG + 1)
    gamma = tl.load(ABG + 2)
    c = tl.abs(alpha) + tl.abs(beta) + tl.abs(gamma)
    qb = b * sqb + h * sqh
    db = b * sdb + h * sdh
    k_base = b * skb + h * skh
    v_base = b * svb + h * svh
    dk0 = tl.zeros([BLOCK_N, DCc], tl.float32)
    dk1 = tl.zeros([BLOCK_N, DCc], tl.float32)
    dk2 = tl.zeros([BLOCK_N, DCc], tl.float32)
    dv0 = tl.zeros([BLOCK_N, DCc], tl.float32)
    dv1 = tl.zeros([BLOCK_N, DCc], tl.float32)
    dv2 = tl.zeros([BLOCK_N, DCc], tl.float32)
    for start in range(0, N, BLOCK_M):
        offs_m = start + offs_m0
        row_ok = offs_m < N
        q0 = _ld_rows(Q, qb, offs_m, sqn, offs_c, sqd, N, D, 0)
        q1 = _ld_rows(Q, qb, offs_m, sqn, offs_c, sqd, N, D, DCc)
        q2 = _ld_rows(Q, qb, offs_m, sqn, offs_c, sqd, N, D, 2 * DCc)
        do0 = _ld_rows(DO, db, offs_m, sdn_, offs_c, sdd, N, D, 0)
        do1 = _ld_rows(DO, db, offs_m, sdn_, offs_c, sdd, N, D, DCc)
        do2 = _ld_rows(DO, db, offs_m, sdn_, offs_c, sdd, N, D, 2 * DCc)
        stat = pid_bh * NS + offs_m
        m1 = tl.load(M1 + stat)
        l1 = tl.load(L1 + stat)
        l2 = tl.load(L2 + stat)
        delta2 = tl.load(DELTA2 + stat)
        delta1 = tl.load(DELTA1 + stat)
        rs = tl.load(RS + stat)
        s, p, p2, dz, dpr, apr = _recompute3(q0, q1, q2, do0, do1, do2, K, V, k_base, v_base, skn, skd, svn, svd,
                                             DIST, ANG, sdn, sdm, offs_m, offs_n, offs_c, m1, l1, l2, delta2,
                                             alpha, beta, gamma, c, N, D, sqrt_dk, DCc, NC, PREC, SPREC, SROUND)
        p2 = tl.where(row_ok[:, None], p2, 0.0)
        if ACC == 1:
            p = p / rs[:, None]
        ds = tl.where(row_ok[:, None], _ds(p, dz, alpha, delta1, ACC), 0.0)
        p2t = tl.trans(p2)
        dst = tl.trans(ds)
        dv0 += _mm(p2t, do0, PREC)
        dk0 += _mm(dst, q0, PREC)
        if NC > 1:
            dv1 += _mm(p2t, do1, PREC)
            dk1 += _mm(dst, q1, PREC)
        if NC > 2:
            dv2 += _mm(p2t, do2, PREC)
            dk2 += _mm(dst, q2, PREC)
    cols = offs_c
    nm = col_ok[:, None]
    tl.store(DK + k_base + offs_n[:, None] * skn + cols[None, :] * skd, dk0 / sqrt_dk, mask=nm & (cols[None, :] < D))
    tl.store(DV + v_base + offs_n[:, None] * svn + cols[None, :] * svd, dv0, mask=nm & (cols[None, :] < D))
    if NC > 1:
        c1 = DCc + cols
        tl.store(DK + k_base + offs_n[:, None] * skn + c1[None, :] * skd, dk1 / sqrt_dk, mask=nm & (c1[None, :] < D))
        tl.store(DV + v_base + offs_n[:, None] * svn + c1[None, :] * svd, dv1, mask=nm & (c1[None, :] < D))
    if NC > 2:
        c2 = 2 * DCc + cols
        tl.store(DK + k_base + offs_n[:, None] * skn + c2[None, :] * skd, dk2 / sqrt_dk, mask=nm & (c2[None, :] < D))
        tl.store(DV + v_base + offs_n[:, None] * svn + c2[None, :] * svd, dv2, mask=nm & (c2[None, :] < D))


# Per-kernel launch configs (BLOCK_M, BLOCK_N, num_warps, num_stages); tuned in KERNEL_TILING_OPTIMIZATION.md
TUNED_TILES = dict(fwd=(32, 64, 4, 2), rows=(32, 32, 4, 1), cols=(32, 32, 8, 2))  # best of the bounded sweep
# Numerical presets (SATURATED_SOFTMAX_BACKWARD_DEBUG.md):
#   sround=True       correctly rounded S = QK^T / sqrt(d) (tl.math.div_rn) in every kernel, so the backward
#                     recomputes exactly the forward's rounded scores; without it FMA contraction leaves the
#                     recomputed P1 peak up to half an ulp of S above m1 (rowsum(P1) - 1 = 2.4e-4 on layer 0)
#   delta2="recompute" extra B0 pass: delta2 = rowsum(P2 * dP2) from the same tiles, so rowsum(dZ) cancels as in
#                     eager (needed for d(alpha, beta, gamma) accuracy)
#   tf32_rn=True      TF32 builds only: operands rounded to nearest TF32 before tl.dot (_rn_tf32); Triton 3.2's own
#                     conversion truncates, which biases O and d(gamma) (15.8x and 122x eager TF32 on layer 0)
#   score_tf32x3=True optional: 3xTF32 for the QK^T score dots only (dQ/dK error 0.2x eager, ~+0.8 ms fwd+bwd)
# Every preset sets every key, so switching presets never inherits a flag from the previous one.
PRESETS = {"accurate_experimental": dict(acc=0, sround=True, delta2="recompute", tf32_rn=True, score_tf32x3=False),
           "accurate_x3_experimental": dict(acc=0, sround=True, delta2="recompute", tf32_rn=True,
                                            score_tf32x3=True),
           "fast_experimental": dict(acc=0, sround=False, delta2="do_o", tf32_rn=False, score_tf32x3=False)}
CFG = dict(TUNED_TILES, **PRESETS["accurate_experimental"])  # never default to the inaccurate variant


def use_preset(name):
    CFG.update(PRESETS[name])
DEBUG = {}  # when DEBUG["on"] is True, the backward stores P1/dZ/dS and the stats here


def _prec():
    return "tf32" if torch.backends.cuda.matmul.allow_tf32 else "ieee"


LAUNCH_INFO = {}  # kernel name -> registers / spills / shared memory of the last compiled launch


def _launch(kernel, grid, cfg, *args, **kw):
    bm, bn, w, st = cfg
    ck = kernel[grid](*args, BLOCK_M=bm, BLOCK_N=bn, DCc=DC, num_warps=w, num_stages=st, **kw)
    name = kernel.fn.__name__ + ("_mode{}".format(kw["MODE"]) if "MODE" in kw else "")
    try:
        LAUNCH_INFO[name] = dict(n_regs=ck.n_regs, n_spills=ck.n_spills, shared_bytes=ck.metadata.shared,
                                 num_warps=w, num_stages=st, block_m=bm, block_n=bn, grid=list(grid))
    except AttributeError:
        pass
    return ck


class FusedGeoAttentionV2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, dist, ang, alpha, beta, gamma):
        B, H, N, D = q.shape
        NC = triton.cdiv(D, DC)
        assert NC <= 3, "v2 supports head dims up to 192; use geo_attention (padded) otherwise"
        dist = dist.contiguous().float()
        ang = ang.contiguous().float()
        abg = torch.cat([alpha.reshape(1), beta.reshape(1), gamma.reshape(1)]).float().contiguous()
        o = torch.empty((B, H, N, D), device=q.device, dtype=torch.float32)
        blk = max(CFG["fwd"][0], CFG["rows"][0], CFG["rows"][1], CFG["cols"][0], CFG["cols"][1])
        NS = triton.cdiv(N, blk) * blk
        m1 = torch.zeros((B * H, NS), device=q.device, dtype=torch.float32)
        l1, l2 = torch.ones_like(m1), torch.ones_like(m1)
        prec = _prec()
        if prec == "tf32" and CFG.get("tf32_rn"):
            prec = "tf32rn"  # round-to-nearest TF32 operands (see _rn_tf32)
        # score GEMM QK^T only: optional 3xTF32 split when running in TF32 (|S| ~ 5e3 on real layer 0)
        sprec = "tf32x3" if (prec in ("tf32", "tf32rn") and CFG.get("score_tf32x3")) else prec
        _launch(_fwd_kernel, (triton.cdiv(N, CFG["fwd"][0]), B * H), CFG["fwd"],
                q, k, v, dist, ang, abg, o, m1, l1, l2, *q.stride(), *k.stride(), *v.stride(), *o.stride(),
                *dist.stride(), H, N, NS, D, math.sqrt(D), NC=NC, PREC=prec, SPREC=sprec, SROUND=CFG["sround"])
        ctx.save_for_backward(q, k, v, dist, ang, abg, o, m1, l1, l2)
        ctx.prec, ctx.sprec, ctx.NC, ctx.sround = prec, sprec, NC, CFG["sround"]  # backward recomputes S exactly as forward
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, dist, ang, abg, o, m1, l1, l2 = ctx.saved_tensors
        B, H, N, D = q.shape
        do = do.float()
        NS = m1.shape[1]
        delta2 = torch.zeros_like(m1)
        if CFG["delta2"] == "do_o":  # FlashAttention identity; "recompute" fills it in B0 instead
            delta2[:, :N] = (do * o).sum(-1).reshape(B * H, N)
        delta1 = torch.zeros_like(m1)
        rs = torch.ones_like(m1)  # row sums of the recomputed P1 (written by B1)
        nprog_m = triton.cdiv(N, CFG["rows"][0])
        part = torch.zeros((B * H * nprog_m, 3), device=q.device, dtype=torch.float64)  # FP64 partial sums of d(alpha, beta, gamma)
        common = (H, N, NS, D, math.sqrt(D))
        args = (*q.stride(), *k.stride(), *v.stride(), *do.stride(), *dist.stride())
        kw = dict(NC=ctx.NC, PREC=ctx.prec, SPREC=ctx.sprec, ACC=CFG["acc"], SROUND=ctx.sround)
        dq = torch.empty_strided(q.shape, q.stride(), device=q.device, dtype=torch.float32)
        debug = bool(DEBUG.get("on"))
        if debug:
            dbg = [torch.zeros((B * H, NS, NS), device=q.device, dtype=torch.float32) for _ in range(3)]
        else:
            dbg = [dq, dq, dq]  # unused placeholders (DEBUG=False compiles the stores away)
        for mode in ((0, 1, 2) if CFG["delta2"] == "recompute" else (1, 2)):
            _launch(_bwd_rows_kernel, (nprog_m, B * H), CFG["rows"], q, k, v, dist, ang, abg, do, m1, l1, l2, delta2,
                    delta1, rs, part, dq, *dbg, *args, *common, MODE=mode, DEBUG=debug and mode == 2, **kw)
        dk = torch.empty_strided(k.shape, k.stride(), device=k.device, dtype=torch.float32)
        dv = torch.empty_strided(v.shape, v.stride(), device=v.device, dtype=torch.float32)
        _launch(_bwd_cols_kernel, (triton.cdiv(N, CFG["cols"][1]), B * H), CFG["cols"], q, k, v, dist, ang, abg, do,
                m1, l1, l2, delta2, delta1, rs, dk, dv, *args, *common, **kw)
        g = part.sum(0).float()
        if debug:
            DEBUG.update(m1=m1[:, :N].clone(), l1=l1[:, :N].clone(), l2=l2[:, :N].clone(),
                         delta2=delta2[:, :N].clone(), delta1=delta1[:, :N].clone(), rs=rs[:, :N].clone(),
                         P1=dbg[0][:, :N, :N].clone(), dZ=dbg[1][:, :N, :N].clone(), dS=dbg[2][:, :N, :N].clone())
        return dq, dk, dv, None, None, g[0].reshape(1), g[1].reshape(1), g[2].reshape(1)


def fused_geo_attention_v2(q, k, v, dist, ang, alpha, beta, gamma):
    return FusedGeoAttentionV2.apply(q, k, v, dist, ang, alpha, beta, gamma)


def patch_model(model, preset="accurate_experimental"):
    """Swap every encoder layer's attention core for the v2 kernel with the given numerical preset."""
    use_preset(preset)
    for layer in model.encoder_layers:
        att = layer.self_attn

        def core(Q, K, V, distances, angularities, mask=None, _att=att):
            assert mask is None, "ForeSWE never passes a mask"
            return fused_geo_attention_v2(Q, K, V, distances, angularities, _att.alpha, _att.beta, _att.gamma)
        att.scaled_dot_product_attention = core
    return model
