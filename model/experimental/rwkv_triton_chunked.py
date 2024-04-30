# -*- coding: utf-8 -*-

# Copyright (c) 2023-2024, Yu Zhang, Songlin Yang, Er

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.utils import chunk_reversed_cumsum_fwd
from fla.utils import contiguous


@triton.autotune(
    configs=[
        triton.Config({'BS': 16}, num_warps=2),
        triton.Config({'BS': 16}, num_warps=4),
        triton.Config({'BS': 16}, num_warps=8),
        triton.Config({'BS': 32}, num_warps=2),
        triton.Config({'BS': 32}, num_warps=4),
        triton.Config({'BS': 32}, num_warps=8),
        triton.Config({'BS': 64}, num_warps=2),
        triton.Config({'BS': 64}, num_warps=4),
        triton.Config({'BS': 64}, num_warps=8),
    ],
    key=['S']
)
@triton.jit
def chunk_rwkv6_fwd_kernel_cum(
    s,
    o,
    o_minus_s,
    s_s_h,
    s_s_t,
    s_s_d,
    T: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.)

    p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    p_o_minus_s = tl.make_block_ptr(o_minus_s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    # [BT, BS]
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    b_o = tl.dot(m_s, b_s, allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_o_minus_s, (b_o - b_s).to(p_o_minus_s.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def post_process_grad(
    q,
    k,
    v,
    u,
    do,
    dk,
    dq,
    du,
    scale,
    s_k_h,
    s_k_t,
    s_k_d,
    s_v_h,
    s_v_t,
    s_v_d,
    H,
    T: tl.constexpr,
    BT: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_h = i_bh % H

    # Note that BK = tl.next_power_of_2(K), BV = tl.next_power_of_2(V)
    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, 0), (BT, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, 0), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, 0), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, 0), (BT, BK), (1, 0))
    p_du = tl.make_block_ptr(du + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, 0), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, 0), (BT, BV), (1, 0))
    p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, 0), (BT, BV), (1, 0))
    p_u = tl.make_block_ptr(u + i_h * K, (K,), (1,), (0,), (BK,), (0,))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_u = tl.load(p_u, boundary_check=(0,))

    b_vdo = tl.sum(b_v * b_do, axis=1)
    b_du = b_vdo[:, None] * b_k * b_q * scale
    b_dq = b_vdo[:, None] * b_k * b_u[None, :] * scale
    b_dk = b_vdo[:, None] * b_q * b_u[None, :] * scale

    b_dq += tl.load(p_dq, boundary_check=(0, 1))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))

    b_dk += tl.load(p_dk, boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

    tl.store(p_du, b_du.to(p_du.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def post_process_grad_umat(
    q,
    k,
    v,
    u,
    do,
    dk,
    dq,
    scale,
    s_k_h,
    s_k_t,
    s_k_d,
    s_v_h,
    s_v_t,
    s_v_d,
    H,
    T: tl.constexpr,
    BT: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_h = i_bh % H

    # Note that BK = tl.next_power_of_2(K), BV = tl.next_power_of_2(V)
    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, 0), (BT, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, 0), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, 0), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, 0), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, 0), (BT, BV), (1, 0))

    p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, 0), (BT, BV), (1, 0))
    # u vector
    # p_u = tl.make_block_ptr(u + i_h * K, (K,), (1,), (0,), (BK,), (0,))
    # Parameters:
    # base – The base pointer to the parent tensor
    # shape – The shape of the parent tensor
    # strides – The strides of the parent tensor
    # offsets – The offsets to the block
    # block_shape – The shape of the block
    # order – The order of the original data format
    p_u = tl.make_block_ptr(u + i_h * K * V, (K, V), (K, 1), (0, 0), (BK, BV), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_u = tl.load(p_u, boundary_check=(0,))

    # b_vdo = tl.sum(b_v * b_do, axis=1)
    # b_du = b_vdo[:, None] * b_k * b_q * scale
    b_vdo = (scale * b_v * b_do)[None, :] * b_u
    b_dq = tl.sum(b_vdo * b_k, axis=0)
    b_dk = tl.sum(b_vdo * b_q, axis=0)

    b_dq += tl.load(p_dq, boundary_check=(0, 1))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))

    b_dk += tl.load(p_dk, boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

    # tl.store(p_du, b_du.to(p_du.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_rwkv6_fwd_kernel_h(
    k,
    v,
    g,
    h,
    h0,
    ht,
    s_k_h,
    s_k_t,
    s_k_d,
    s_v_h,
    s_v_t,
    s_v_d,
    s_h_h,
    s_h_t,
    s_h_d,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(h0 + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h += tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)
    for i_t in range(NT):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, s_h_d), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_g = tl.make_block_ptr(g + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_gn = tl.make_block_ptr(g + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + BT - 1) * K + i_k * BK,), (BK,), (0,))

        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BK, BT]
        b_g = tl.load(p_g, boundary_check=(0, 1))
        if i_t < NT - 1:
            # [BK,]
            b_gn = tl.load(p_gn, boundary_check=(0,))
        else:
            b_gn = tl.min(b_g, axis=1)
        b_h *= tl.exp(b_gn)[:, None]
        b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(b_k.dtype)
        b_h += tl.dot(b_k, b_v, allow_tf32=False)

    if STORE_FINAL_STATE:
        p_h = tl.make_block_ptr(ht + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_rwkv6_fwd_kernel_intra(
    q,
    k,
    g,
    gs,
    u,
    A,
    s_k_h,
    s_k_t,
    s_k_d,
    scale,
    H,
    T: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    DK: tl.constexpr,
):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i, i_j = i_c // (NC * NC), (i_c % (NC * NC)) // NC, (i_c % (NC * NC)) % NC
    i_h = i_bh % H
    n_bh = tl.num_programs(2)

    if i_i > i_j:
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        # p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_gs = tl.make_block_ptr(gs + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gk = tl.make_block_ptr(g + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gn = tl.make_block_ptr(g + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + i_i * BC) * K + i_k * BK,), (BK,), (0,))
        p_A = tl.make_block_ptr(A + (i_k*n_bh+i_bh)*T*BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
        # [BK,]
        b_gn = tl.load(p_gn, boundary_check=(0,))
        # [BC, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_gs = tl.load(p_gs, boundary_check=(0, 1))
        b_qg = (b_q * tl.exp(b_gs - b_gn[None, :]) * scale).to(b_q.dtype)
        # [BK, BC]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kg = (b_k * tl.exp(b_gn[:, None] - b_gk)).to(b_k.dtype)
        # [BC, BC]
        b_A = tl.dot(b_qg, b_kg, allow_tf32=False)
        tl.store(p_A, b_A.to(A.dtype.element_ty), boundary_check=(0, 1))
    elif i_i == i_j:
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_gs = tl.make_block_ptr(gs + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_gk = tl.make_block_ptr(g + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + i_j * BC) * K + i_k * BK,), (BK,), (0,))
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + i_j * BC) * K + i_k * BK,), (BK,), (0,))
        p_q_self = tl.make_block_ptr(q + i_bh * s_k_h, (T * K,), (s_k_d,),
                                     ((i_t * BT + i_j * BC) * K + i_k * BK,), (BK,), (0,))
        # [BC, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_gs = tl.load(p_gs, boundary_check=(0, 1))
        o_i = tl.arange(0, BC)
        o_A = (i_bh + i_k * n_bh) * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_j * BC
        m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T
        p_u = tl.make_block_ptr(u + i_h * DK, (DK,), (1,), (i_k * BK), (BK,), (0,))
        b_u = tl.load(p_u, boundary_check=(0,))
        # FIXME: if running with real U: add DV to make a square indexing matrix.
        # p_u = tl.make_block_ptr(u + i_h * DK * DV, (DK, DV), (DV,1), (i_k * BK * DV), (BK,BV), (0,))
        # b_u = tl.load(p_u, boundary_check=(0,1))

        for j in range(0, BC):
            # inter
            # [BK,]
            b_k = tl.load(p_k, boundary_check=(0,)).to(tl.float32)
            b_gk = tl.load(p_gk, boundary_check=(0,)).to(tl.float32)
            # [BC,]
            b_A = tl.sum(b_q * b_k[None, :] * tl.exp(b_gs - b_gk[None, :]) * scale, 1)
            b_A = tl.where(o_i > j, b_A, 0.)
            # self
            b_q_self = tl.load(p_q_self, boundary_check=(0,)).to(tl.float32)
            A_self = tl.sum(b_q_self * b_k * b_u * scale, axis=0)
            m_self = tl.arange(0, BC) == j
            b_A = tl.where(m_self, A_self[None], b_A)
            tl.store(A + o_A + j, b_A.to(A.dtype.element_ty), mask=m_A)
            p_k = tl.advance(p_k, (K,))
            p_q_self = tl.advance(p_q_self, (K,))
            p_gk = tl.advance(p_gk, (K,))


@triton.jit
def chunk_rwkv6_fwd_kernel_inter(
    q,
    v,
    gs,
    h,
    o,
    A,
    s_k_h,
    s_k_t,
    s_k_d,
    s_v_h,
    s_v_t,
    s_v_d,
    s_h_h,
    s_h_t,
    s_h_d,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_gs = tl.make_block_ptr(gs + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, s_h_d), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BT, BK]
        b_gs = tl.load(p_gs, boundary_check=(0, 1))
        # [BT, BK]
        b_qg = (b_q * tl.exp(b_gs)).to(b_q.dtype)
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # works but dkw, owing to divine benevolence
        # [BT, BV]
        if i_k >= 0:
            b_o += tl.dot(b_qg, b_h, allow_tf32=False)
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    # [BT, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    # [BT, BT]
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_o += tl.dot(b_A, b_v, allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_rwkv6_bwd_kernel_dh(
    q,
    g,
    gs,
    do,
    dh,
    s_k_h,
    s_k_t,
    s_k_d,
    s_v_h,
    s_v_t,
    s_v_d,
    s_h_h,
    s_h_t,
    s_h_d,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    for i_t in range(NT - 1, -1, -1):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K*V, (K, V), (s_h_t, s_h_d), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_gs = tl.make_block_ptr(gs + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_gn = tl.make_block_ptr(g + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + BT - 1) * K + i_k * BK,), (BK,), (0,))

        # [BK, BT]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BT, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))

        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))

        # [BK,]
        b_gn = tl.load(p_gn, boundary_check=(0,))
        # [BK, BV]
        b_dh *= tl.exp(b_gn)[:, None]
        # [BK, BT]
        b_gs = tl.load(p_gs, boundary_check=(0, 1))
        b_q = (b_q * tl.exp(b_gs)).to(b_q.dtype)

        # [BK, BV]
        b_dh += tl.dot(b_q, b_do, allow_tf32=False)


@triton.jit
def chunk_rwkv6_bwd_kernel_inter(
    k,
    v,
    h,
    g,
    gs,
    A,
    do,
    dh,
    dq,
    dk,
    dv,
    dA,
    s_k_h,
    s_k_t,
    s_k_d,
    s_v_h,
    s_v_t,
    s_v_d,
    s_h_h,
    s_h_t,
    s_h_d,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)

    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_gk = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_gq = tl.make_block_ptr(gs + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_gn = tl.make_block_ptr(g + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + BT - 1) * K + i_k * BK,), (BK,), (0,))
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (BT, T), (1, BT), (0, i_t * BT), (BT, BT), (0, 1))

    # [BT, BK]
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_gk = tl.load(p_gk, boundary_check=(0, 1))
    b_gq = tl.load(p_gq, boundary_check=(0, 1))
    b_gn = tl.exp(tl.load(p_gn, boundary_check=(0,))[None, :] - b_gk)
    b_k = (b_k * b_gn).to(b_k.dtype)
    # [BT, BT]
    b_A = tl.load(p_A, boundary_check=(0, 1))

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * V * K, (V, K), (s_h_d, s_h_t), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K*V, (K, V), (s_h_t, s_h_d), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_k*n_bh+i_bh) * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # [BT, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BK, BV]
        b_dh = tl.load(p_dh, boundary_check=(0, 1))

        # [BT, BV]
        b_dv = tl.dot(b_k, b_dh, allow_tf32=False)
        if i_k == 0:
            b_dv += tl.dot(b_A, b_do, allow_tf32=False)
        b_do = (b_do * scale).to(b_do.dtype)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
        # [BT, BT]
        b_dA += tl.dot(b_do, tl.trans(b_v), allow_tf32=False)
        # [BT, BK]
        b_dq += tl.dot(b_do, b_h, allow_tf32=False)
        # [BT, BK]
        b_dk += tl.dot(b_v, tl.trans(b_dh), allow_tf32=False)

    b_dq = b_dq * tl.exp(b_gq)
    b_dk = b_dk * b_gn

    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT, ), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] > o_i[None, :]
    # [BT, BT]
    b_dA = tl.where(m_s, b_dA, 0.).to(b_k.dtype)
    if i_k == 0:
        tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_rwkv6_bwd_kernel_intra(
    q,
    k,
    g,
    gs,
    dA,
    dq,
    dk,
    s_k_h,
    s_k_t,
    s_k_d,
    T: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr
):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i = i_c // NC, i_c % NC

    p_gs = tl.make_block_ptr(gs + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_gn = tl.make_block_ptr(g + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + i_i * BC) * K + i_k * BK,), (BK,), (0,))
    # [BK,]
    b_gn = tl.load(p_gn, boundary_check=(0,))
    # [BC, BK]
    b_gs = tl.load(p_gs, boundary_check=(0, 1))
    b_dq = tl.zeros([BC, BK], dtype=tl.float32)
    for i_j in range(0, i_i):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
        p_gk = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
        p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
        # [BC, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kg = (b_k * tl.exp(b_gn[None, :] - b_gk)).to(b_k.dtype)
        # [BC, BC]
        b_dA = tl.load(p_dA, boundary_check=(0, 1))
        # [BC, BK]
        b_dq += tl.dot(b_dA, b_kg, allow_tf32=False)
    b_dq *= tl.exp(b_gs - b_gn[None, :])
    o_i = tl.arange(0, BC)
    o_dA = i_bh * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_i * BC
    m_dA = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T

    for j in range(0, BC):
        p_kj = tl.make_block_ptr(k + i_bh * s_k_h, (T * K,), (1,), ((i_t * BT + i_i*BC+j) * K + i_k * BK,), (BK,), (0,))
        p_gkj = tl.make_block_ptr(g + i_bh * s_k_h, (T * K,), (1,), ((i_t * BT + i_i*BC+j) * K + i_k * BK,), (BK,), (0,))

        # [BC,]
        b_dA = tl.load(dA + o_dA + j, mask=m_dA, other=0)
        # [BK,]
        b_kj = tl.load(p_kj, boundary_check=(0,)).to(tl.float32)
        b_gkj = tl.load(p_gkj, boundary_check=(0,)).to(tl.float32)
        # [BC, BK]
        m_i = o_i[:, None] > j
        # [BC, BK]
        b_dq += tl.where(m_i, b_dA[:, None] * b_kj[None, :] * tl.exp(b_gs - b_gkj[None, :]), 0.)

    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))

    b_dq = b_dq + tl.load(p_dq, boundary_check=(0, 1))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))

    tl.debug_barrier()
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_gk = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_gn = tl.make_block_ptr(g + i_bh * s_k_h, (T*K,), (s_k_d,), ((i_t * BT + i_i * BC + BC - 1) * K + i_k * BK,), (BK,), (0,))
    # [BK,]
    b_gn = tl.load(p_gn, boundary_check=(0,))
    # [BC, BK]
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_gk = tl.load(p_gk, boundary_check=(0, 1))
    b_dk = tl.zeros([BC, BK], dtype=tl.float32)
    for i_j in range(i_i + 1, NC):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
        p_gs = tl.make_block_ptr(gs + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
        p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_j * BC, i_i * BC), (BC, BC), (1, 0))
        # [BC, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_gs = tl.load(p_gs, boundary_check=(0, 1))
        b_qg = (b_q * tl.exp(b_gs - b_gn[None, :])).to(b_q.dtype)
        # [BC, BC]
        b_dA = tl.load(p_dA, boundary_check=(0, 1))
        # [BC, BK]
        b_dk += tl.dot(tl.trans(b_dA), b_qg, allow_tf32=False)
    b_dk *= tl.exp(b_gn[None, :] - b_gk)

    o_dA = i_bh * T * BT + (i_t * BT + i_i * BC) * BT + i_i * BC + tl.arange(0, BC)
    for j in range(0, BC):
        p_qj = tl.make_block_ptr(q + i_bh * s_k_h, (T * K,), (1,), ((i_t * BT + i_i * BC + j) * K + i_k * BK,), (BK,), (0,))
        p_gqj = tl.make_block_ptr(gs + i_bh * s_k_h, (T * K,), (1,), ((i_t * BT + i_i * BC + j) * K + i_k * BK,), (BK,), (0,))
        # [BC,]
        b_dA = tl.load(dA + o_dA + j * BT, mask=(i_t * BT + i_i * BC + j < T), other=0)
        # [BK,]
        b_qj = tl.load(p_qj, boundary_check=(0,)).to(tl.float32)
        b_gqj = tl.load(p_gqj, boundary_check=(0,)).to(tl.float32)
        # [BC, BK]
        m_i = o_i[:, None] < j
        b_dk += tl.where(m_i, b_dA[:, None] * b_qj[None, :] * tl.exp(b_gqj[None, :] - b_gk), 0.)

    p_dk = tl.make_block_ptr(dk + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    b_dk = b_dk + tl.load(p_dk, boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


class ChunkRWKV6Function(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, r, k, v, g, u, scale, initial_state, output_final_state, checkpoint_level):
        q = r  # alias
        B, H, T, K, V = *q.shape, v.shape[-1]
        BT, BC = 64, 16
        BK = min(64, triton.next_power_of_2(K))
        BV = min(64, triton.next_power_of_2(V))
        NT, NC = triton.cdiv(T, BT), triton.cdiv(BT, BC)
        NK = triton.cdiv(K, BK)
        NV = triton.cdiv(V, BV)
        num_warps = 4 if BK == 64 else 2
        num_stages = 1

        def fwd_inner(q, k, v, g, B, H, T, K, V, BT, BK, BV, NT, h0=None, ht=None):
            NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
            h = q.new_empty(B, H, NT * K, V)
            grid = (NV, NK, B * H)
            chunk_rwkv6_fwd_kernel_h[grid](
                k, v, g, h, h0, ht,
                k.stride(1), k.stride(2), k.stride(3),
                v.stride(1), v.stride(2), v.stride(3),
                h.stride(1), h.stride(2), h.stride(3),
                T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT,
                USE_INITIAL_STATE=h0 is not None,
                STORE_FINAL_STATE=ht is not None,
                num_warps=num_warps,
                num_stages=num_stages
            )
            return h

        final_state = None
        if output_final_state:
            final_state = q.new_empty(B, H, K, V, dtype=torch.float)

        g_org, g, gs = g, torch.empty_like(g, dtype=torch.float), torch.empty_like(g, dtype=torch.float)
        def grid(meta): return ((triton.cdiv(meta['S'], meta['BS']), NT, B * H))
        # keep cummulative normalizer in fp32
        # this kernel is equivalent to
        # g_org = g_org.view(B, H, NT, BT, -1)
        # g = g_org.cumsum(-2).view(B, H, T, -1)
        # gs = g - g_org
        chunk_rwkv6_fwd_kernel_cum[grid](
            g_org, g, gs,
            g.stride(1), g.stride(2), g.stride(3),
            T=T, S=K, BT=BT
        )
        h = fwd_inner(
            q=q, k=k, v=v, g=g,
            B=B, H=H, T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT,
            h0=initial_state if initial_state is not None else None,
            ht=final_state if final_state is not None else None
        )
        A = q.new_zeros(NK, B, H, T, BT)
        u_ = u.new_zeros(H, K, V)
        grid = (NK, NT * NC * NC, B * H)
        chunk_rwkv6_fwd_kernel_intra[grid](
            q, k, g, gs, u_, A,
            k.stride(1), k.stride(2), k.stride(3),
            scale,
            H=H, T=T, K=K, BT=BT, BC=BC, BK=BK, NC=NC, DK=K,
            num_warps=num_warps,
            num_stages=num_stages
        )
        A = A.sum(0, dtype=A.dtype)
        o = torch.empty_like(v)

        grid = (NV, NT, B * H)
        chunk_rwkv6_fwd_kernel_inter[grid](
            q, v, gs, h, o, A,
            k.stride(1), k.stride(2), k.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            h.stride(1), h.stride(2), h.stride(3),
            scale,
            T=T, K=K, V=V, BT=BT, BK=BK, BV=BV,
            num_warps=num_warps,
            num_stages=num_stages
        )
        # FIXME: ignore U for now and just build extra output (price paid)
        o = o + torch.einsum('bhic,hcd,bhic,bhid->bhid', q, u, k, v)

        # if checkpoint_level >= 1:
        #     del g
        #     g = g_org
        if checkpoint_level > 1:
            del h
            h, initial_state = None, None
        del g, gs

        ctx.save_for_backward(q, k, v, g_org, u, h, initial_state, A)
        ctx.BT = BT
        ctx.scale = scale
        ctx.checkpoint_level = checkpoint_level

        return o, final_state

    @staticmethod
    @contiguous
    def backward(ctx, do, dht=None):
        q, k, v, g, u, h, initial_state, A = ctx.saved_tensors
        B, H, T, K, V = *q.shape, v.shape[-1]
        BT, BC = ctx.BT, 16
        BK = min(64, triton.next_power_of_2(K))
        BV = min(64, triton.next_power_of_2(V))
        NT, NC = triton.cdiv(T, BT), triton.cdiv(BT, BC)
        NK = triton.cdiv(K, BK)
        num_warps = 4 if BK == 64 else 2
        num_stages = 1

        def fwd_inner(q, k, v, g, B, H, T, K, V, BT, BK, BV, NT, h0=None, ht=None):
            NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
            h = q.new_empty(B, H, NT * K, V)
            grid = (NV, NK, B * H)
            chunk_rwkv6_fwd_kernel_h[grid](
                k, v, g, h, h0, ht,
                k.stride(1), k.stride(2), k.stride(3),
                v.stride(1), v.stride(2), v.stride(3),
                h.stride(1), h.stride(2), h.stride(3),
                T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT,
                USE_INITIAL_STATE=h0 is not None,
                STORE_FINAL_STATE=ht is not None,
                num_warps=num_warps,
                num_stages=num_stages
            )
            return h

        def bwd_inner(q, g, gs, do, B, H, T, K, V, BT, BK, BV, NT, scale):
            NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
            dh = q.new_empty(B, H, NT * K, V)
            grid = (NK, NV, B * H)
            chunk_rwkv6_bwd_kernel_dh[grid](
                q, g, gs, do, dh,
                q.stride(1), q.stride(2), q.stride(3),
                do.stride(1), do.stride(2), do.stride(3),
                dh.stride(1), dh.stride(2), dh.stride(3),
                scale,
                T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT,
                num_warps=num_warps,
                num_stages=num_stages
            )
            return dh

        # recompute cumulative log decays.
        g_org, g, gs = g, torch.empty_like(g, dtype=torch.float), torch.empty_like(g, dtype=torch.float)
        def grid(meta): return ((triton.cdiv(meta['S'], meta['BS']), NT, B * H))
        # keep cummulative normalizer in fp32
        # this kernel is equivalent to
        # g = g.view(B, H, NT, BT, -1).cumsum(-2).view(B, H, T, -1)
        chunk_rwkv6_fwd_kernel_cum[grid](
            g_org, g, gs,
            g.stride(1), g.stride(2), g.stride(3),
            T=T, S=K, BT=BT
        )

        # rerun the forward pass to get h if checkpoint_level >= 1
        if ctx.checkpoint_level == 1:
            h = fwd_inner(
                q=q, k=k, v=v, g=g,
                B=B, H=H, T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT,
                h0=initial_state if initial_state is not None else None,
                ht=None
            )

        scale = ctx.scale
        dh = bwd_inner(
            q, g, gs, do,
            B=B, H=H, T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT,
            scale=scale
        )
        dq = torch.empty_like(q, dtype=torch.float)
        dk = torch.empty_like(k, dtype=torch.float)
        dv = v.new_empty(NK, *v.shape)
        dA = q.new_zeros(B, H, T, BT)
        grid = (NK, NT, B * H)
        chunk_rwkv6_bwd_kernel_inter[grid](
            k, v, h, g, gs, A, do, dh, dq, dk, dv, dA,
            k.stride(1), k.stride(2), k.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            h.stride(1), h.stride(2), h.stride(3),
            scale,
            T=T, K=K, V=V, BT=BT, BK=BK, BV=BV,
            num_warps=num_warps,
            num_stages=num_stages
        )
        dv = dv.sum(0, dtype=dv.dtype)
        grid = (NK, NT * NC, B * H)
        chunk_rwkv6_bwd_kernel_intra[grid](
            q, k, g, gs, dA, dq, dk,
            k.stride(1), k.stride(2), k.stride(3),
            T=T, K=K, BT=BT, BC=BC, BK=BK, NC=NC,
            num_warps=num_warps,
            num_stages=num_stages
        )

        # TODO: fuse?
        dg = (dq * q)[:, :, 1:] - (dk * k)[:, :, 0:-1]
        dg = torch.nn.functional.pad(dg, (0, 0, 0, 1, 0, 0, 0, 0), value=0)
        dg = chunk_reversed_cumsum_fwd(dg).to(g)
        # # equivalent to the following pytorch code.
        # # du = ((do * v).sum(-1)[..., None] * k * q * scale).sum(-2).to(u)
        # # dq += ((do * v).sum(-1)[..., None] * k * scale * u[:, :, None, :])
        # # dk += ((do * v).sum(-1)[..., None] * q * scale * u[:, :, None, :])
        # BT = 64
        # grid = (triton.cdiv(T, BT), B * H)
        # du = torch.empty_like(g, dtype=torch.float)
        # post_process_grad[grid](
        #     q, k, v, u, do, dk, dq, du, scale,
        #     q.stride(1), q.stride(2), q.stride(3),
        #     v.stride(1), v.stride(2), v.stride(3), H=H,
        #     T=T, BT=BT, K=K, V=V, BK=triton.next_power_of_2(K), BV=triton.next_power_of_2(V),
        #     num_warps=4
        # )
        # du = du.sum([0, 2])

        # only multiply if differen
        qscale, kscale = q, k
        if abs(1 - scale) > 1e-5:
            qscale = q * scale
            kscale = k * scale

        du = torch.einsum('bhnv,bhnv,bhnk,bhnk->hkv', do, v, qscale, k)
        dq += torch.einsum('bhnv,bhnv,bhnk,hkv->bhnk', do, v, kscale, u)
        dk += torch.einsum('bhnv,bhnv,bhnk,hkv->bhnk', do, v, qscale, u)

        return dq.to(q), dk.to(k), dv.to(v), dg.to(g), du.to(u), None, None, None, None


def chunk_rwkv6(
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    u: torch.Tensor,
    scale: Optional[int] = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    checkpoint_level: Optional[int] = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        r (torch.Tensor):
            reception of shape `(B, H, T, K)`. Alias: q, query in linear attention.
        k (torch.Tensor):
            keys of shape `(B, H, T, K)`
        v (torch.Tensor):
            values of shape `(B, H, T, V)`
        w (torch.Tensor):
            data-dependent decays of shape `(B, H, T, K)` in log space! Alias: g.
        u (torch.Tensor):
            bonus of shape `(H, K)`
        scale (Optional[int]):
            Scale factor for the RWKV6 attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `(B, H, K, V)`. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `(B, H, K, V)`. Default: `False`.
        checkpoint_level (Optional[int]):
            Checkpointing level; higher values will save more memories and do more recomputations during backward.
            Default: `0`:
            - Level `0`: store forward hidden states for backprop.
            - Level `1`: recompute the forward hidden states during backward.
    """
    assert checkpoint_level in [0, 1]
    if scale is None:
        scale = r.shape[-1] ** -0.5
    if initial_state is not None:
        initial_state = initial_state.detach()
    o, final_state = ChunkRWKV6Function.apply(r, k, v, g, u, scale, initial_state, output_final_state, checkpoint_level)
    return o, final_state
