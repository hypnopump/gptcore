# run with: TRITON_INTERPRET=0 python dev_rwkv6_triton.py

import torch as th
import torch.nn.functional as F

from fla.ops.rwkv6.recurrent_fuse import fused_recurrent_rwkv6
from fla.ops.rwkv6.recurrent_naive import naive_recurrent_rwkv6

import sys
sys.path.append("../")
from model.experimental.rwkv_triton_v7 import fused_recurrent_rwkv7hypno
from model.experimental.rwkv_triton_chunked import chunk_rwkv6


def naive_recurrent_rwkv6(
        q: th.Tensor,
        k: th.Tensor,
        v: th.Tensor,
        w: th.Tensor,
        u: th.Tensor,
        initial_state=None,
        output_final_state=False
):
    """ Inputs:
    * q, k: (B, H, L, K)
    * v: (B, H, L, V)
    * w: (B, H, L, K)
    * u: (H, K, V)
    Outputs: (B, H, L, K)
    """
    orig_dtype = q.dtype
    q, k, v, w, u = map(lambda x: x.float(), (q, k, v, w, u))
    batch_size, n_heads, seq_len, d_head_k = q.shape
    _, _, _, d_head_v = v.shape
    h = th.zeros(batch_size, n_heads, d_head_k, d_head_v, dtype=th.float32, device=q.device)
    o = th.zeros_like(v)

    if initial_state is not None:
        h += initial_state

    for i in range(seq_len):
        q_i = q[:, :, i, :]
        k_i = k[:, :, i]
        v_i = v[:, :, i, :]
        w_i = w[:, :, i].exp()
        kv_i = k_i[..., None] * v_i[..., None, :]
        o_i = (h + u[None, ..., None] * kv_i) * q_i[..., None]
        o[:, :, i] = o_i.sum(-2)
        h = h * w_i[..., None] + kv_i
    return o.to(orig_dtype)


def naive_recurrent_rwkv6_umat(
        q: th.Tensor,
        k: th.Tensor,
        v: th.Tensor,
        w: th.Tensor,
        u: th.Tensor,
        initial_state=None,
        output_final_state=False
):
    """ Inputs:
    * q, k: (B, H, L, K)
    * v: (B, H, L, V)
    * w: (B, H, L, K)
    * u: (H, K, V)
    Outputs: (B, H, L, K)
    """
    orig_dtype = q.dtype
    q, k, v, w, u = map(lambda x: x.float(), (q, k, v, w, u))
    batch_size, n_heads, seq_len, d_head_k = q.shape
    _, _, _, d_head_v = v.shape
    h = th.zeros(batch_size, n_heads, d_head_k, d_head_v, dtype=th.float32, device=q.device)
    o = th.zeros_like(v)

    if initial_state is not None:
        h += initial_state

    for i in range(seq_len):
        q_i = q[:, :, i, :]
        k_i = k[:, :, i]
        v_i = v[:, :, i, :]
        w_i = w[:, :, i].exp()
        kv_i = k_i[..., None] * v_i[..., None, :]
        o_i = (h + u[None, ...] * kv_i) * q_i[..., None]
        o[:, :, i] = o_i.sum(-2)
        h = h * w_i[..., None] + kv_i
    return o.to(orig_dtype)


def naive_recurrent_rwkv6_uwmat(
        q: th.Tensor,
        k: th.Tensor,
        v: th.Tensor,
        w: th.Tensor,
        u: th.Tensor,
        initial_state=None,
        output_final_state=False
):
    """ Inputs:
    * q, k: (B, H, L, K)
    * v: (B, H, L, V)
    * w: (B, H, L, K)
    * u: (H, K, V)
    Outputs: (B, H, L, K)
    """
    orig_dtype = q.dtype
    q, k, v, w, u = map(lambda x: x.float(), (q, k, v, w, u))
    batch_size, n_heads, seq_len, d_head_k = q.shape
    _, _, _, d_head_v = v.shape
    h = th.zeros(batch_size, n_heads, d_head_k, d_head_v, dtype=th.float32, device=q.device)
    o = th.zeros_like(v)

    if initial_state is not None:
        h += initial_state

    for i in range(seq_len):
        q_i = q[:, :, i, :]
        k_i = k[:, :, i]
        v_i = v[:, :, i, :]
        w_i = w[:, :, i].exp()
        kv_i = k_i[..., None] * v_i[..., None, :]
        o_i = (h + u[None, ...] * kv_i) * q_i[..., None]
        o[:, :, i] = o_i.sum(-2)
        h = h * w_i + kv_i
    return o.to(orig_dtype)


def test_v7(B, H, L, K, V):
    # B, H, L, K, V = 1, 1, 128, 256, 256
    def gen_inputs():
        th.manual_seed(17)
        device = "cuda"

        # triton inputs
        rt = th.randn(B, H, L, K, device=device, requires_grad=True)
        kt = th.randn(B, H, L, K, device=device, requires_grad=True)
        vt = th.randn(B, H, L, V, device=device, requires_grad=True)
        wt = th.randn(B, H, L, K, V, device=device, requires_grad=True)
        ut = th.randn(H, K, V, device=device, requires_grad=True)
        return rt, kt, vt, wt, ut


    rt, kt, vt, wt, ut = gen_inputs()
    w_ = -th.exp(wt)
    o = naive_recurrent_rwkv6_uwmat(rt, kt, vt, w_, ut)
    # print("naive kernel", o)
    o.mean().backward()
    grads = {
        "ut": ut.grad.clone(),
        "wt": wt.grad.clone(),
        "rt": rt.grad.clone(),
        "kt": kt.grad.clone(),
        "vt": vt.grad.clone()
    }

    # print("grad of ut", ut.grad)

    rt, kt, vt, wt, ut = gen_inputs()
    w_ = -th.exp(wt)
    ot, fstate = fused_recurrent_rwkv7hypno(rt, kt, vt, w_, ut, scale=1)
    # print("modified kernel", ot)
    print("native - modified kernel", o - ot)

    ot.mean().backward()
    # print("fused", ut.grad, rt.grad, kt.grad)
    grad2 = {
        "ut": ut.grad.clone(),
        "wt": wt.grad.clone(),
        "rt": rt.grad.clone(),
        "kt": kt.grad.clone(),
        "vt": vt.grad.clone()
    }
    for k, v in grads.items():
        print(f"k:{k} {grads[k] - grad2[k]}")

    # rt, kt, vt, wt, ut = gen_inputs()
    # u_ = ut[..., None].repeat(1, 1, V)
    # ot, fstate = fused_recurrent_rwkv6(rt, kt, vt, wt, ut, scale=1)
    # print("and default rwkv", ot)
    # print(ut.grad)


def test_chunked_v6plus(B, H, L, K, V):
    # B, H, L, K, V = 1, 1, 128, 256, 256
    def gen_inputs():
        th.manual_seed(17)
        device = "cuda"

        # triton inputs
        rt = th.randn(B, H, L, K, device=device, requires_grad=True)
        kt = th.randn(B, H, L, K, device=device, requires_grad=True)
        vt = th.randn(B, H, L, V, device=device, requires_grad=True)
        wt = th.randn(B, H, L, K, device=device, requires_grad=True)
        ut = th.randn(H, K, device=device, requires_grad=True)
        return rt, kt, vt, wt, ut


    rt, kt, vt, wt, ut = gen_inputs()
    w_ = -th.exp(wt)
    o = naive_recurrent_rwkv6(rt, kt, vt, w_, ut)
    # print("naive kernel", o)
    o.mean().backward()
    grads = {
        "ut": ut.grad.clone(),
        "wt": wt.grad.clone(),
        "rt": rt.grad.clone(),
        "kt": kt.grad.clone(),
        "vt": vt.grad.clone()
    }

    # print("grad of ut", ut.grad)

    rt, kt, vt, wt, ut = gen_inputs()
    w_ = -th.exp(wt)
    ot, fstate = chunk_rwkv6(rt, kt, vt, w_, ut, scale=1)
    # print("modified kernel", ot)
    print("native - modified kernel", o - ot)

    # ot.mean().backward()
    # # print("fused", ut.grad, rt.grad, kt.grad)
    # grad2 = {
    #     "ut": ut.grad.clone(),
    #     "wt": wt.grad.clone(),
    #     "rt": rt.grad.clone(),
    #     "kt": kt.grad.clone(),
    #     "vt": vt.grad.clone()
    # }
    # for k, v in grads.items():
    #     print(f"k:{k} {grads[k] - grad2[k]}")


#######################################

if __name__ == "__main__":
    B, H, L, K, V = 1, 1, 32, 16, 16
    # test_v7(B, H, L, K, V)
    test_chunked_v6plus(B, H, L, K, V)

