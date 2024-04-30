# -*- coding: utf-8 -*-

# Copyright (c) 2024, Songlin Yang, Eric Alcaide

from typing import Tuple

from math import ceil
import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_bwd, custom_fwd

from fla.ops.utils import chunk_reversed_cumsum_fwd
from fla.utils import contiguous
from typing import Optional


# on-the-fly computation without materializing hidden statets into HBMs

# FIXME: THIS IS A HYBRID BETWEEN `rwkv_triton_v6plus (fw)` and `rwkv_triton_chunked (bw)`
from .rwkv_triton_v6hypno import fused_recurrent_rwkv6_bwd_kernel_dq, fused_recurrent_rwkv6_bwd_kernel_dkv
from .rwkv_triton_chunked import (
    chunk_rwkv6_fwd_kernel_h,
    chunk_rwkv6_fwd_kernel_cum,
    chunk_rwkv6_fwd_kernel_intra,
    chunk_rwkv6_fwd_kernel_inter,
    chunk_rwkv6_bwd_kernel_dh,
    chunk_rwkv6_bwd_kernel_inter,
    chunk_rwkv6_bwd_kernel_intra,
    post_process_grad,
    post_process_grad_umat,
)


class ChunkedFwRecurrentBwRWKV6PlusFunction(torch.autograd.Function):

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

        def grid(meta):
            return ((triton.cdiv(meta['S'], meta['BS']), NT, B * H))

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
        # ctx.save_for_backward(q, k, v, g_org, u, initial_state)
        ctx.BT = BT
        ctx.scale = scale
        ctx.checkpoint_level = checkpoint_level
        ctx.reverse = False

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

        def grid(meta):
            return ((triton.cdiv(meta['S'], meta['BS']), NT, B * H))

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
        BT = 1 # 64
        grid = (triton.cdiv(T, BT), B * H)
        # du = torch.empty_like(g, dtype=torch.float)
        # post_process_grad[grid](
        #     q, k, v, u, do, dk, dq, du, scale,
        #     q.stride(1), q.stride(2), q.stride(3),
        #     v.stride(1), v.stride(2), v.stride(3), H=H,
        #     T=T, BT=BT, K=K, V=V, BK=triton.next_power_of_2(K), BV=triton.next_power_of_2(V),
        #     num_warps=4
        # )
        du = q.new_zeros(B, H, ceil(T/BT), K, V)
        post_process_grad_umat[grid](
            q, k, v, u, do, dk, dq, du, scale,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3), u.stride(1),  H=H,
            T=T, BT=BT, K=K, V=V, BK=triton.next_power_of_2(K), BV=triton.next_power_of_2(V),
            num_warps=4
        )
        du = du.sum((0, 2), dtype=u.dtype)


        # # # doing it in torch looses speed advantage compared to sequential kernel (lol)
        # # # # only multiply if differen
        # qscale = q
        # # # q, kscale = q, k
        # if abs(1 - scale) > 1e-5:
        #     qscale = q * scale
        #     # kscale = k * scale
        # # #
        # # # dov = do*v
        # du = torch.einsum('bhnv,bhnv,bhnk,bhnk->hkv', do, v, qscale, k)
        # # # dq += torch.einsum('bhnv,bhnk,hkv->bhnk', dov, kscale, u)
        # # # dk += torch.einsum('bhnv,bhnk,hkv->bhnk', dov, qscale, u)

        return dq.to(q), dk.to(k), dv.to(v), dg.to(g), du.to(u), None, None, None, None

    # # @custom_bwd # inner most decorator
    # @staticmethod
    # @contiguous
    # def backward(ctx, do, d_final_state=None):
    #     """ Recurrent backward pass as everything dependent on U. """
    #     q, k, v, w, u, initial_state = ctx.saved_tensors
    #     batch_size, n_heads, seq_len, d_head_qk = q.shape
    #     d_head_v = v.shape[-1]
    #     scale = ctx.scale
    #
    #     BK, BV = min(triton.next_power_of_2(d_head_qk), 16), min(triton.next_power_of_2(d_head_v), 64)
    #     NK, NV = triton.cdiv(d_head_qk, BK), triton.cdiv(d_head_v, BV)
    #     num_stages = 1
    #     num_warps = 1
    #     dq = q.new_empty(NV, batch_size, n_heads, seq_len,
    #                      d_head_qk, dtype=torch.float32)
    #     dq_aux = torch.empty_like(dq)
    #     grid = (NV, NK, batch_size * n_heads)
    #
    #     fused_recurrent_rwkv6_bwd_kernel_dq[grid](
    #         k, v, w, u, do, dq, dq_aux, initial_state,
    #         q.stride(1), q.stride(2), q.stride(3),
    #         v.stride(1), v.stride(2), v.stride(3),
    #         batch_size, n_heads, seq_len, scale,
    #         DK=d_head_qk, DV=d_head_v, BK=BK, BV=BV,
    #         num_warps=num_warps,
    #         num_stages=num_stages,
    #         USE_INITIAL_STATE=initial_state is not None,
    #         REVERSE=ctx.reverse,
    #     )
    #     dq = dq.sum(0).to(q)
    #     dq_aux_red = dq_aux.sum(0).to(w)
    #     del dq_aux
    #
    #     BK, BV = min(triton.next_power_of_2(d_head_qk), 32), min(triton.next_power_of_2(d_head_v), 32)
    #     NK, NV = triton.cdiv(d_head_qk, BK), triton.cdiv(d_head_v, BV)
    #     num_stages = 1
    #     num_warps = 1
    #     dk = q.new_empty(NV, batch_size, n_heads, seq_len,
    #                      d_head_qk, dtype=torch.float32)
    #     dk_aux = q.new_empty(NV, batch_size, n_heads, seq_len,
    #                          d_head_qk, dtype=torch.float32)
    #     dv = q.new_empty(NK, batch_size, n_heads, seq_len,
    #                      d_head_v, dtype=torch.float32)
    #     grid = (NV, NK, batch_size * n_heads)
    #
    #     fused_recurrent_rwkv6_bwd_kernel_dkv[grid](
    #         q, k, v, w, u, do, dk, dk_aux, dv,
    #         q.stride(1), q.stride(2), q.stride(3),
    #         v.stride(1), v.stride(2), v.stride(3),
    #         batch_size, n_heads, seq_len, scale,
    #         DK=d_head_qk, DV=d_head_v, BK=BK, BV=BV,
    #         num_warps=num_warps,
    #         num_stages=num_stages,
    #         USE_INITIAL_STATE=initial_state is not None,
    #         REVERSE=ctx.reverse,
    #     )
    #     dk = dk.sum(0).to(k)
    #     dv = dv.sum(0).to(v)
    #     dk_aux_red = dk_aux.sum(0).to(w)
    #     del dk_aux
    #
    #     dw = dq_aux_red[:, :, 1:] - dk_aux_red[:, :, 0:-1]
    #     dw = torch.nn.functional.pad(dw, (0, 0, 0, 1, 0, 0, 0, 0), value=0)
    #     dw = chunk_reversed_cumsum_fwd(dw).to(w)
    #     if initial_state is None:
    #         dw[:, :, 0] = 0.
    #
    #     qscale = q
    #     if abs(1 - scale) > 1e-5:
    #         qscale = q * scale
    #
    #     du = torch.einsum('bhnv,bhnv,bhnk,bhnk->hkv', do, v, qscale, k)
    #     # du = ((do*dv)[:, :, :, None] * (k * q * scale)[..., None]).sum((0, 2)).to(u)
    #     return dq, dk, dv, dw, du, None, None, None, None


# if scale is None, use d_head_qk ** -0.5 by default. Otherwise specify the scale yourself. e.g. scale = 1.0
def chunk_rwkv6hypno(
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
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
    assert checkpoint_level in {0, 1}
    if scale is None:
        scale = r.shape[-1] ** -0.5
    if initial_state is not None:
        initial_state = initial_state.detach()
    o, final_state = ChunkedFwRecurrentBwRWKV6PlusFunction.apply(r, k, v, w, u, scale, initial_state, output_final_state, checkpoint_level)
    return o, final_state
