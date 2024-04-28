# This file is modified from https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v4neo/src/model.py and is separately licensed according to the following license:
"""
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
"""

########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

from typing import Callable, Any, Optional, Tuple, List, Iterable, Callable

import math

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from model.core import TransformerLayerPart, TimeLerp, ReluSquared
from model.interface import IFeedForwardSubLayer

from .rwkv_inner import rwkv_inner, rwkv_inner_umat
# supports Umat, not Wmat
from .rwkv_triton_v6hypno import fused_recurrent_rwkv6hypno
# supports both Umat and Wmat
from .rwkv_triton_v7 import fused_recurrent_rwkv7hypno
# supports Umat ^ Zmat
from .rwkv_triton_v6hypno2 import fused_recurrent_rwkv6hypno2

# base RWKV6. chunked is significantly faster (80ktok/s vs 50ktok/s on uncompiled L8D512H8; seq_len=1024; A100)
from fla.ops.rwkv6.recurrent_fuse import fused_recurrent_rwkv6
from fla.ops.rwkv6.chunk import chunk_rwkv6


# Compiled torch functions
rwkv_inner_compiled = torch.compile(rwkv_inner)
rwkv_inner_umat_compiled = torch.compile(rwkv_inner_umat)

# Compile-friendly fused kernels
chunk_rwkv6_compiled = torch._dynamo.disable(torch.jit.ignore(chunk_rwkv6))
fused_recurrent_rwkv6_compiled = torch._dynamo.disable(torch.jit.ignore(fused_recurrent_rwkv6))
fused_recurrent_rwkv6hypno_compiled = torch._dynamo.disable(torch.jit.ignore(fused_recurrent_rwkv6hypno))
fused_recurrent_rwkv6hypno2_compiled = torch._dynamo.disable(torch.jit.ignore(fused_recurrent_rwkv6hypno2))
fused_recurrent_rwkv7hypno_compiled = torch._dynamo.disable(torch.jit.ignore(fused_recurrent_rwkv7hypno))


TAU = 9

@th.compile
def safe_wclamp(x: th.Tensor) -> th.Tensor:
    return -th.exp(- F.elu(-x + TAU) + TAU)


# version without u 'bonus' term
def rwkv6_0_simple_recurrent(r_in, k_in, v_in, w_in, kv_state):
    B,H,L,K = r_in.shape
    V = v_in.size(-1)
    out = []
    for t in range(L):
        r, k, v, w = r_in[...,t:t+1,:], k_in[...,t:t+1,:], v_in[...,t:t+1,:], w_in[...,t:t+1,:]
        kv_state = (w.mT * kv_state) + k.mT @ v # KV
        out.append( r @ kv_state ) # 1K @ KV -> 1V
    out = torch.cat(out, dim=-2)
    return out, kv_state

# version including u 'bonus' term
def rwkv6_0_recurrent(r_in, k_in, v_in, w_in, u, kv_state):
    B,H,L,K = r_in.shape
    V = v_in.size(-1)
    L = r_in.size(-2)
    out = []
    for t in range(L):
        r, k, v, w = r_in[...,t:t+1,:], k_in[...,t:t+1,:], v_in[...,t:t+1,:], w_in[...,t:t+1,:]
        kv = k.mT @ v # KV
        out.append( r @ (kv_state + u.mT * kv) ) # 1K @ KV -> 1V
        kv_state = (w.mT * kv_state) + kv # KV
    out = torch.cat(out, dim=-2)
    return out, kv_state

def rwkv7_recurrent(r_in, k_in, v_in, w_in, u, kv_state = 0.):
    """
    * r_in: (B,H,L,K)
    * k_in: (B,H,L,K)
    * v_in: (B,H,L,V)
    * w_in: (B,H,L,K,V)
    * u: (H,K,V)
    """
    dtype = r_in.dtype
    B,H,L,K = r_in.shape
    V = v_in.size(-1)
    L = r_in.size(-2)
    out = []
    kmt = k_in.mT.float()
    w_ = w_in.float().unbind(dim=-3)
    r_ = r_in.float().unsqueeze(-3).unbind(dim=-2)
    kv_state = kv_state.float()
    kvs = torch.einsum('bhik,bhid->bhikd', k_in, v_in).unbind(dim=-3)
    for t in range(L):
        r, k, v = r_in[...,t:t+1,:], kmt[...,t:t+1], v_in[...,t:t+1,:]
        # w = w_in[...,t,:,:]
        kv = kvs[t]
        # out.append( r @ (kv_state + u * kv) ) # 1K @ KV -> 1V
        # kv_state = (w * kv_state) + kv  # KV
        out.append(r_[t] @ torch.addcmul(kv_state, u, kv))
        kv_state = torch.addcmul(kv, w_[t], kv_state)
    out = torch.cat(out, dim=-2).to(dtype)
    return out, None

def sanity_check():
    torch.manual_seed(1337)
    
    T = 9
    B = 1
    H = 1
    K,V = 3,5
    r = torch.rand(B,H,T,K)
    k = torch.rand(B,H,T,K)
    v = torch.rand(B,H,T,V)
    w = torch.rand(1,H,T,K).expand(B,H,T,K)
    u = torch.rand(1,H,1,K)
    kv_state = torch.zeros(B,H,K,V,device=v.device,dtype=v.dtype)

    #precision_dtype, precision_min_val = torch.float64, 1e-10 # good for fp64 (1.7e-308 ^ (1/16.0) == 5.8e-20)
    precision_dtype, precision_min_val = torch.float32, 0.005 # good for fp32 (1.175e-38 ^ (1/16.0) < 0.00426)
    w = w.clamp(precision_min_val)

    # recurrent
    out, _ = rwkv6_0_recurrent(r,k,v,w,u,kv_state)
    print(out)

    # parallel
    out, _ = rwkv_inner(r,k,v,w,u,kv_state,chunk_len=3)
    print(out)

if __name__ == "__main__":
    sanity_check()
    exit()

from util.config import Factory

import posemb.interface

import model.interface
import model.core
from model.hparams import HParams


from model.rwkv import RWKVConfig

import norm

class RWKV6_0_AttentionSubLayer(model.core.TransformerLayerPart, model.interface.IAttentionSubLayer):
    def __init__(self):
        super().__init__()

        hparams, layer_id = self.hparams, self.layer_id

        args = RWKVConfig(hparams)
        self.umat = True   # True
        self.zmat = False  # True
        self.wmat = True   # True
        self.k_one_minus_w = False

        self.args = args
        self.layer_id = layer_id
        self.ctx_len = args.ctx_len
        self.n_embd = args.n_embd

        self.n_head = args.n_head
        self.n_kv_head = args.n_kv_head
        self.r_head_size = args.dim_rk // args.n_head
        self.k_head_size = args.dim_rk // args.n_head
        self.v_head_size = args.dim_v // args.n_head
        assert args.dim_rk % self.n_head == 0
        assert args.dim_rk % self.n_kv_head == 0
        assert args.dim_v % self.n_kv_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / max(args.n_layer - 1, 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            self.x_maa = nn.Parameter(1 - torch.pow(ddd, ratio_1_to_almost0))
            self.r_maa = nn.Parameter(1 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.w_maa = nn.Parameter(1 - torch.pow(ddd, ratio_1_to_almost0))
            self.k_maa = nn.Parameter(1 - torch.pow(ddd, ratio_1_to_almost0))
            self.v_maa = nn.Parameter(1 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.g_maa = nn.Parameter(1 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            TIME_MIX_EXTRA_DIM = 32
            self.tm_w1 = nn.Parameter(torch.empty(args.n_embd, TIME_MIX_EXTRA_DIM * 5).uniform_(-0.01, 0.01))
            self.tm_w2 = nn.Parameter(torch.zeros(5, TIME_MIX_EXTRA_DIM, args.n_embd))
            W_MIX_EXTRA_DIM = 64*2
            self.td_w1 = nn.Parameter(torch.empty(args.n_embd, W_MIX_EXTRA_DIM).uniform_(-0.01, 0.01))
            self.td_w2 = nn.Parameter(torch.zeros(W_MIX_EXTRA_DIM, args.n_embd*2))

            # fancy time_decay
            k_dim_att = args.n_kv_head * self.k_head_size
            decay_speed = torch.ones(k_dim_att, self.v_head_size)
            for n in range(k_dim_att):
                decay_speed[n] = -6 + 5 * (n / max(k_dim_att - 1, 1)) ** (0.7 + 1.3 * ratio_0_to_1)

            time_decay = decay_speed.reshape(self.n_kv_head, self.k_head_size, self.v_head_size) # (KVH, K, V)
            if self.wmat:
                self.time_decay = nn.Parameter(time_decay)  # (KVH, K, V)
            else:
                self.time_decay = nn.Parameter(time_decay[..., 0])  # (KVH, K)
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            tmp = torch.zeros(k_dim_att, self.v_head_size)
            for n in range(k_dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / max(k_dim_att - 1, 1))) + zigzag
                tmp[n, :] += torch.randn_like(tmp[n, :]) * 1e-4

            time_first = tmp.reshape(self.n_kv_head, self.k_head_size, self.v_head_size)
            if self.umat:
                self.time_first = nn.Parameter(time_first) # (KVH, K, K)
            else: 
                self.time_first = nn.Parameter(time_first[..., 0]) # (KVH, K)

            self.time_filter = nn.Parameter(1 + 1e-5 * torch.randn(self.n_kv_head, self.k_head_size, self.v_head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        # headwise r:
        # with torch.no_grad():
        #     mats = []
        #     receptance = nn.Linear(args.n_embd, self.n_head * self.r_head_size, bias=False)
        #     for i in range(self.n_head):
        #         mats.append(receptance.weight.T[i*self.r_head_size:(i+1)*self.r_head_size, i*self.r_head_size:(i+1)*self.r_head_size].clone())
        # self.receptance = nn.Parameter(torch.stack(mats, dim=0)) # (H, K, K)

        # single head, headwise rkw:
        # receptance, key, decay = [], [], []
        # with torch.no_grad():
        #     # receptance.append(nn.Linear(self.r_head_size, self.r_head_size, bias=False).weight.T)
        #     # key.append(nn.Linear(self.k_head_size, self.k_head_size, bias=False).weight.T)
        #     decay.append(nn.Linear(self.k_head_size, self.k_head_size, bias=False).weight.T)

        # self.receptance = nn.Parameter(torch.stack(receptance, dim=0))  # (H, K, K)
        # self.key = nn.Parameter(torch.stack(key, dim=0))  # (H, K, K)
        # self.decay = nn.Parameter(torch.stack(decay, dim=0))  # (H, K, K)


        # classic
        self.receptance = nn.Linear(args.n_embd, self.n_head * self.r_head_size, bias=False)
        self.key = nn.Linear(args.n_embd, self.n_kv_head * self.k_head_size, bias=False)
        self.value = nn.Linear(args.n_embd, self.n_kv_head * self.v_head_size, bias=False)
        self.output = nn.Linear(args.dim_v, args.n_embd, bias=False)
        self.gate = nn.Linear(args.n_embd, args.dim_v, bias=False)

        self.rotary_positional_embedding = hparams.rotary_positional_embedding_factory(hparams.max_sequence_length, int(hparams.d_qk_ratio * hparams.d_model / hparams.n_head))

        # self.ln_x = nn.GroupNorm(self.n_kv_head, args.dim_v)
        self.ln_x = nn.LayerNorm(args.dim_v, elementwise_affine=False)

    def post_init_fn(self, myself):
        zero = [self.output]
        for m in zero:
            nn.init.zeros_(m.weight)
        # FIXME - init ln_x with something like layer_scale * 0.7
        ortho = [self.value, self.gate]
        for m in ortho:
            if m.weight.shape[0] > m.weight.shape[1]:
                gain = math.sqrt(m.weight.shape[0] / m.weight.  shape[1])
            else:
                gain = 1.0
            nn.init.orthogonal_(m.weight, gain=gain)

    def forward(self, xq : Tensor, xk : Tensor, xv : Tensor, recurrent_memory : Optional[Tensor] = None):
        x = xq # FIXME - support encoder-decoder models

        # 24 is optimal chunk length (longer will use too much memory and cause precision problems or even numerical instability, shorter is inefficient)
        chunk_len = 24

        # padding to support fast path for non-exact chunk size multiple sequence lengths        
        n_padding = (chunk_len - x.size(-2) % chunk_len) % chunk_len
        if n_padding != 0:
            x = F.pad(x, [0, 0, 0, n_padding, 0, 0])

        H = self.n_head
        KVH = self.n_kv_head
        R = self.r_head_size
        K = self.k_head_size
        V = self.v_head_size

        B, T, C = x.size()

        xx = x
        sx = self.time_shift(x) - xx
        #sx = torch.cat((sx.unsqueeze(0), xx[:-1,:])) - xx

        xxx = xx + sx * self.x_maa
        xxx = torch.tanh(xxx @ self.tm_w1).view(B*T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.tm_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        wx = xx + sx * (self.w_maa + mw)
        kx = xx + sx * (self.k_maa + mk)
        vx = xx + sx * (self.v_maa + mv)
        rx = xx + sx * (self.r_maa + mr)
        gx = xx + sx * (self.g_maa + mg)

        r = self.receptance(rx).view(B, T, H, K).transpose(1, 2)  # BHTK
        k = self.key(kx).view(B, T, KVH, K).transpose(1, 2)      # BHTK
        v = self.value(vx).view(B, T, KVH, V).transpose(1, 2)  # BHTV
        g = F.silu(self.gate(gx))

        # headwise, multi-head R
        # r = th.einsum('bthk,hkd->bhtd', rx.view(B, T, H, K), self.receptance)  # BHTK
        # k = th.einsum('bthk,hkd->bhtd', kx.view(B, T, H, K), self.key)  # BHTK
        # wk = th.einsum('bthk,hkd->bhtd', wx.view(B, T, H, K), self.decay)  # BHTK

        # headwise, single-head (weights-wise)
        # r = self.receptance(rx.view(B, T, H, K).transpose(1, 2))  # BHTK
        # k = self.key(kx.view(B, T, H, K).transpose(1, 2))      # BHTK
        # wk = self.decay(wx.view(B, T, H, K).transpose(1, 2))  # BHTKV



        r, k = self.rotary_positional_embedding((r, k))

        # support for grouped-query attention
        # if there are fewer k/v heads than total heads, repeat them until the number matches
        time_decay = self.time_decay.float() # (KVH,K)
        time_first = self.time_first.float() # (KVH,K)
        if KVH < H:
            reps = H // KVH
            k = k[:,:,None,:,:].expand(B, KVH, reps, T, K).contiguous().view(B, H, T, K)
            v = v[:,:,None,:,:].expand(B, KVH, reps, T, V).contiguous().view(B, H, T, V)
            time_decay = time_decay.expand(reps, KVH, K).contiguous().view(H, K)
            time_first = time_first.expand(reps, KVH, K).contiguous().view(H, K)

        kv_state = recurrent_memory

        # if kv_state is None:
        #     kv_state = torch.zeros(B, H, K, V, device=r.device, dtype=r.dtype)
        #
        # if r.dtype == torch.bfloat16 and kv_state.dtype != torch.bfloat16:
        #     kv_state = kv_state.contiguous().to(torch.bfloat16)

        eps = 1e-5
        tau = th.e
        if self.wmat:
            w = time_decay.view(1, H, 1, K, V)
            loraw = (torch.tanh(wx @ self.td_w1) @ self.td_w2).view(B, T, H, 2*K).transpose(1, 2) # BHTK()
            lorak, lorav = loraw.chunk(2, dim=-1)
            w = w + lorak[..., None] * lorav[..., None, :]
            # w = -torch.exp(w) # log(exp(-exp))
            # w = (-w.exp()).exp()
            w = safe_wclamp(w)  # w = -th.exp(- F.elu(-w+tau)+tau)
            # w = w[..., [0]].repeat(1, 1, 1, 1, V)
            # w = (eps + (1-eps) * w).log()  # (B, H, T, K, V)
        else:
            w = time_decay.view(1, H, 1, K)
            if self.k_one_minus_w:
                w = w+k
                w = safe_wclamp(w)  # w = -th.exp(- F.elu(-w+tau)+tau)
                # w = (-w.exp()).exp()
                k = (1-w.exp()).to(r)
            else:
                w = w + (torch.tanh(wx @ self.td_w1) @ self.td_w2).view(B, T, H, K).transpose(1, 2)  # BHTK
                # w = w + k
                w = safe_wclamp(w)  # w = -th.exp(- F.elu(-w + tau) + tau)
                # w = (-w.exp()).exp()

            # w = -torch.exp(w) # log(exp(-exp))
            # w = (eps + (1 - eps) * w).log()  # (B, H, T, K)

        if self.umat:
            u = time_first.view(H,K,V)
            if self.wmat:
                out, s = fused_recurrent_rwkv7hypno_compiled(r, k, v, w, u, initial_state=kv_state, scale=1.0)
                # kv_state = torch.zeros(B, H, K, V, device=r.device, dtype=r.dtype)
                # out, s = rwkv7_recurrent(r, k, v, w, u, kv_state)
            else:
                if self.zmat:
                    time_filter = self.time_filter.float() # (KVH,K,V)
                    z = time_filter.view(H,K,V)
                    z = z * (2 - u)
                    out, s = fused_recurrent_rwkv6hypno2_compiled(r, k, v, w, u, z, initial_state=kv_state, scale=1.0)
                else:
                    # torch + compile = 115 ktok/s || torch+fcompile = 73 ktok/s || torch = 45 ktok/s || recurrent = 40 ktok/s || chunked = ???
                    out, s = fused_recurrent_rwkv6hypno_compiled(r, k, v, w, u, initial_state=kv_state, scale=1.0)
                    # kv_state = torch.zeros(B, H, K, V, device=r.device, dtype=r.dtype)
                    # out, s = rwkv_inner_umat(r, k, v, w, u, kv_state, chunk_len)
        else:
            # torch + th.compile = 200 ktok/s || torch+fcompile = 80 ktok/s || torch = 70 ktok/s || recurrent = 50 ktok/s || chunked = 100 ktok/s
            # kv_state = torch.zeros(B, H, K, V, device=r.device, dtype=r.dtype)
            # w = th.exp(w)
            # u = time_first.view(1,H,1,K)
            # out, s = rwkv_inner(r, k, v, w, u, kv_state, chunk_len)

            u = time_first.view(H, K)
            # out, s = fused_recurrent_rwkv6_compiled(r, k, v, w, u, initial_state=kv_state, scale=1.0)
            out, s = chunk_rwkv6_compiled(r, k, v, w, u, initial_state=kv_state, scale=1.0, checkpoint_level=0)


        out = out.transpose(1,2).reshape(B*T, H*V)
        out = self.ln_x(out / self.args.head_size_divisor).view(B, T, H*V) #  - self.ln_x.bias
        # out = self.ln_x(out).view(B, T, H * V) - self.ln_x.bias
        # g=1.
        # out = out.transpose(1, 2).reshape(B*T, H*V)
        # out = self.ln_x(out).view(B, T, H*V)

        out = self.output(out * g)

        if n_padding != 0:
            out = out[..., :-n_padding, :]  # BTC

        return out


class DDLorExp(TransformerLayerPart):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, bias: bool = False, init_w: float = 1.):
        super().__init__()
        """ Data-dependent low-rank exponential """
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.bias = bias

        self.w1 = th.nn.Parameter(th.empty(in_dim, hidden_dim).uniform_(-0.01, 0.01))
        self.w2 = th.nn.Parameter(th.zeros(hidden_dim, out_dim))
        self.gmult = th.nn.Parameter(init_w * th.ones(out_dim))
        if self.bias:
            self.gbias = th.nn.Parameter(th.zeros(out_dim))


    def forward(self, x: th.Tensor) -> th.Tensor:
        glora = (th.tanh(x @ self.w1) @ self.w2)
        gate = self.gmult.to(glora) * glora.exp()
        # gate = self.gmult.to(glora) * glora.softmax(dim=-1)

        if self.bias:
            gate = gate + self.gbias.to(gate)
        return gate
    

class DDLorElu2(TransformerLayerPart):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, bias: bool = False, init_w: float = 1.):
        super().__init__()
        """ Data-dependent low-rank exponential """
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.bias = bias

        self.w1 = th.nn.Parameter(th.empty(in_dim, hidden_dim).uniform_(-0.01, 0.01))
        self.w2 = th.nn.Parameter(th.zeros(hidden_dim, out_dim))
        self.gmult = th.nn.Parameter(init_w * th.ones(out_dim))
        if self.bias:
            self.gbias = th.nn.Parameter(th.zeros(out_dim))


    def forward(self, x: th.Tensor) -> th.Tensor:
        glora = (th.tanh(x @ self.w1) @ self.w2)
        elug = F.elu(glora)
        # TODO: try this again. hypnoFFN and this lol
        elu2 = th.where(glora < 0., elug+1, (glora+0.5).square()+0.75)
        gate = self.gmult.to(glora) * elu2
        if self.bias:
            gate = gate + self.gbias.to(gate)
        return gate


class RWKVFeedForwardSubLayer(TransformerLayerPart, IFeedForwardSubLayer):
    def __init__(self, 
                 hidden_activation_factory : Callable = Factory(ReluSquared), 
                 gate_activation_factory : Callable = Factory(th.nn.Sigmoid),
                 time_mixer_factory : Callable = Factory(TimeLerp)):
        super().__init__()
        D = self.hparams.d_model
        F = int(self.hparams.feedforward_d_model_ratio * self.hparams.d_model)
        self.time_mixer_hidden = time_mixer_factory()
        self.time_mixer_gate = time_mixer_factory()
        self.w_hidden = th.nn.Linear(D, F, bias=False)
        self.hidden_activation = hidden_activation_factory()
        self.w_out = th.nn.Linear(F, D, bias=False)
        self.w_gate = th.nn.Linear(D, D, bias=False)
        self.gate_activation = gate_activation_factory()

    def forward(self, x : th.Tensor):
        x_hidden = self.time_mixer_hidden(x)
        x_gate = self.time_mixer_gate(x)
        hidden = self.w_hidden(x_hidden)
        hidden = self.hidden_activation(hidden)

        # FIXME - try this, there was a paper that claimed it was better!
        # hidden = norm.RMSNorm.F(hidden)

        gate = self.w_gate(x_gate)
        gate = self.gate_activation(gate)
        return gate * self.w_out(hidden)


class HypnoFeedForwardSubLayer(TransformerLayerPart, IFeedForwardSubLayer):
    def __init__(self, 
                 hidden_activation_factory : Callable = Factory(ReluSquared), 
                 gate_activation_factory : Callable = Factory(th.nn.Sigmoid),
                 time_mixer_factory : Callable = Factory(TimeLerp)):
        super().__init__()
        """ Warning! Should be used with wider (3 -> 3.5 hidden dim than baseline) """
        D = self.hparams.d_model
        F = int(self.hparams.feedforward_d_model_ratio * self.hparams.d_model)
        self.time_mixer_hidden = time_mixer_factory()

        self.w_hidden = th.nn.Linear(D, F, bias=False)
        self.hidden_activation = hidden_activation_factory()
        self.w_out = th.nn.Linear(F, D, bias=False)
        GATE_EXP_DIM = 64 ## D//8
        self.w_gate = DDLorExp(in_dim=D, hidden_dim=GATE_EXP_DIM, out_dim=D)

    def forward(self, x : th.Tensor):
        x_hidden = self.time_mixer_hidden(x)

        hidden = self.w_hidden(x_hidden)
        hidden = self.hidden_activation(hidden)

        gate = self.w_gate(x_hidden)
        return gate * self.w_out(hidden)