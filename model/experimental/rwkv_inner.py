import torch
import torch.nn as nn
import torch.nn.functional as F

import torch as th
from torch import Tensor

# 24 is optimal chunk length (longer will use too much memory and cause precision problems or even numerical instability, shorter is inefficient)
def rwkv_inner(r,k,v,w,u,kv_state,chunk_len:int=24,precision_dtype:torch.dtype=torch.float32):
    """
    expects
    r : (B,H,L,K)
    k : (B,H,L,K)
    v : (B,H,L,V)
    w : (B,H,L,K) or (1,H,L,K)
    u : (1,H,1,K)
    kv_state : (B,H,K,V)
    """
    B,H,L,K = k.size()
    V = v.size(-1)
    T = chunk_len

    if L == 1:
        kv = k.mT @ v
        out = r @ (kv_state + u.mT * kv)
        kv_state = w.mT * kv_state + kv
        return out, kv_state
    else:
        # FIXME - support fast path for non-exact multiples
        # ensure it's an exact multiple
        if L % T != 0:
            T = 1

        N = L // T

        # this has to be done to avoid numerical instability (inf/NaN) when w is used as a divisor up to chunk_length//2 places away (so precision_min_val^(T//2) has to be in fp range)
        # NOTE - this does not account for the impact of the size of R, K so we currently use the chunk_len=32 numbers for chunk_len=24
        assert(precision_dtype == torch.float32 or precision_dtype == torch.float64)
        if precision_dtype == torch.float32:
            precision_min_val = 0.005 # good for fp32 (1.175e-38 ^ (1/16.0) < 0.00426)
        else: #elif precision_dtype == torch.float64:
            precision_min_val = 1e-10 # good for fp64 (1.7e-308 ^ (1/16.0) < 5.8e-20)
        w = w.clamp(precision_min_val)

        # calculate cumulative decay in log space where it won't overflow
        w_log = w.float().log() # (1,H,L,K) or (B,H,L,K)

        # chunked view of w_log
        wc_log = w_log.view(w.size(0),H,N,T,K)
        wc_log_cum = wc_log.cumsum(dim=-2)

        # chunked view of shifted_w_log
        shifted_wc_log_cum = F.pad(wc_log_cum, (0, 0, 1, -1))


        # NOTE - we have to apply the decay weight from TWO ahead.. ONE ahead gets no decay (log==0)
        # pre-applied weights
        # left side is prior chunk (w_inter), right side is current chunk (w_intra)
        # without u...
        # w0   w1   w2   w3   | w4   w5   w6   w7          
        # w1:4 w2:4 w3:4 w4:4 | w4:5 w4:6 w4:7 w4:8
        # with u...
        # w0   w1   w2   w3   | w4   w5   w6   w7          
        # w1:4 w2:4 w3:4 w4:4 | w4:4 w4:5 w4:6 w4:7

        # ws decays the entire current state (representing t-1) to the prior block (t-2)
        ws = wc_log.sum(dim=-2, keepdim=True) # 1HN1K or BHN1K
        # w_inter is the decay to the end of the current block, since it will be applied at the next iteration when current (t) becomes prior (t-1)
        # this formula because e.g. w1:4 = w0:4 - w0:1
        w_inter = ws - wc_log_cum # 1HNTK or BHNTK (w^(T-1) ... w^0)
        # w_intra is the decay from the beginning of the current block (t), since it will be applied to current queries (t) against prior state (representing keys+values up to but not including block t)
        # this formula because e.g. w1:3 = w0:3 - w0
        w_intra = wc_log_cum - wc_log # 1HNTK or BHNTK (w^0 ... w^(T-2))

        ws = list(ws.mT.exp().to(r.dtype).unbind(dim=-3)) # N x 1HK1 or BHK1 !!NOTE THE .mT HERE!!
        w_inter = w_inter.exp().to(r.dtype) # 1HNTK or BHNTK
        w_intra = w_intra.exp().to(r.dtype) # 1HNTK or BHNTK

        # chunked view of r, k, v
        r = r.view(B,H,N,T,K) 
        k = k.view(B,H,N,T,K) 
        v = v.view(B,H,N,T,V)
        u = u.unsqueeze(2).to(r.dtype) # (1,H,1,1,K)

        # parallel calculation of all intra-chunk attention contributions
        wc_log_offset = shifted_wc_log_cum[...,T//2:T//2+1,:] # B,H,N,1,K
        r_decay = (shifted_wc_log_cum - wc_log_offset).to(precision_dtype).exp() # B,H,N,T,K
        k_inv_decay = (wc_log_offset - wc_log_cum).to(precision_dtype).exp() # B,H,N,T,K
        a = ((r*r_decay) @ (k*k_inv_decay).mT).to(r.dtype).tril(-1) # B,H,N,T,T
        # add u term to attention (NOTE - the tril(-1) above zeroed the diagonal)
        a = a + torch.einsum('bhntk,bhntk->bhnt', r, u * k).diag_embed()
        out = a @ v # BHNTV
        # alternate way of adding in u
        # out = out + torch.einsum('bhntk,bhntk,bhntv->bhntv', r, u * k, v) 

        # parallel precalculation of chunked (k*wk).mT@v for use in recurrent state calc below
        wkv = (k * w_inter).mT @ v # BHNKV
        wkv = list(wkv.unbind(dim=-3)) # N x BHKV

        # recurrent calculation of all states
        states = []
        for i in range(N):
            states.append(kv_state)
            kv_state = kv_state * ws[i] + wkv[i] # BHKV
            # equivalent non-precalced version
            #wkv = (k[...,i,:,:] * wk[...,i,:,:]).mT @ v[...,i,:,:]
            #kv_state = kv_state * ws[i] + wkv
        states = torch.stack(states, dim=2) # BHNKV       

        # parallel application of all r to states
        out = out + (r * w_intra) @ states # BHNTV
        out = out.view(B,H,L,V)
        return out, kv_state


def delta_rule_chunkwise2(q: th.Tensor, k: th.Tensor, v: th.Tensor, beta: th.Tensor, chunk_size: int = 24) -> th.Tensor:
    """ https://arxiv.org/abs/2404.07143
    Wrap with functorch for batched inputs.
    Notation:
    - t: chunks
    - c: chunk_size
    - d: dimension
    Inputs:
    * q: (L, D)
    * k: (L, D)
    * v: (L, D)
    * beta: (L) -> scalar for gating. stands for sigmoid(beta) in paper.
    * chunk_size: int. Lower for less mem.
    Outputs: (L, D)
    """
    l, d_k = q.shape
    d_v = v.shape[-1]
    T, C = l // chunk_size, chunk_size
    if l % C != 0:
        C = 1
    T = l // C

    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    # (l, d) -> (t, c, d)
    q, k, v, k_beta = map(lambda x: x.reshape(T, C, -1), [q, k, v, k_beta])
    # (t, ci, cj)
    attn = th.tril(k_beta @ k.mT, -1)

    # (t, c, dv+dk)
    k_cumsum = th.zeros_like(v)
    k_cumdecay = th.zeros_like(k_beta)
    k_cumsum[:, 0] = v[:, 0]
    k_cumdecay[:, 0] = k_beta[:, 0]

    for i in range(1, C):
        # (t, 1, d) -> (t, 1, d) - [ [ (t, 1, :i) -> (t, 1, :i) ] @ (t, :i, d) ] -> (t, 1, d)
        k_cumsum[:, i] = v[:, i] - (attn[:, i, :i, None].mT @ k_cumsum[:, :i]).squeeze(1)
        # (t, 1, d) -> (t, 1, d) - [ [ (t, 1, :i) -> (t, 1, :i) ] @ (t, :i, d) ] -> (t, 1, d)
        k_cumdecay[:, i] = k_beta[:, i] - (attn[:, i, :i, None].mT @ k_cumdecay[:, :i]).squeeze(1)

    v = k_cumsum
    s = k.new_zeros(d_k, d_v)
    S = []
    V = []

    kt = k.mT
    for i in range(0, T):
        S.append(s)
        v_new = v[i] - k_cumdecay[i] @ s
        s = s + kt[i] @ v_new
        V.append(v_new)
    S = th.stack(S, 0)
    V = th.stack(V, 0)

    q = q * (d_k ** -0.5)
    qk = th.tril(q @ kt)
    out = q @ S + qk @ V
    # (t, c, d) -> (l, d)
    return out.reshape(l, -1)


vvmap_delta_rule = th.vmap(th.vmap(delta_rule_chunkwise2))