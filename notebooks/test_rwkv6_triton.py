import triton
import torch as th
import torch.nn.functional as F

from fla.ops.rwkv6.recurrent_fuse import fused_recurrent_rwkv6
from fla.ops.rwkv6.chunk import chunk_rwkv6
# from fla.ops.rwkv_6.recurrent_fuse import fused_recurrent_rwkv6

# from .model.experimental.rwkv_inner import rwkv_inner

import sys
sys.path.append("../")
from model.experimental.rwkv_triton_v6hypno import fused_recurrent_rwkv6hypno
from model.experimental.rwkv_triton_chunked import chunk_rwkv6
from model.experimental.rwkv_triton_v6hypno_chunked import chunked_fw_recurrent_bw_rwkv6hypno



# 24 is optimal chunk length (longer will use too much memory and cause precision problems or even numerical instability, shorter is inefficient)
def rwkv_inner(r,k,v,w,u,kv_state,chunk_len:int=24,precision_dtype:th.dtype=th.float32):
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
        assert(precision_dtype == th.float32 or precision_dtype == th.float64)
        if precision_dtype == th.float32:
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
        a = a + th.einsum('bhntk,bhntk->bhnt', r, u * k).diag_embed()
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
        states = th.stack(states, dim=2) # BHNKV       

        # parallel application of all r to states
        out = out + (r * w_intra) @ states # BHNTV
        out = out.view(B,H,L,V)
        return out, kv_state
    

def test_rwkv(): 
    B, H, L, K, V = 1, 1, 1024, 128, 128

    th.manual_seed(17)
    
    # triton inputs
    rt, kt, wt = th.randn(3, B, H, L, K, device="cuda").to(th.bfloat16)
    vt = th.randn(B, H, L, V, device="cuda").to(th.bfloat16)
    ut = th.randn(H, K, device="cuda").to(th.bfloat16)
    
    # cuda inputs
    rc, kc, vc, wc = rt.clone(), kt.clone(), vt.clone(), wt.clone()
    uc = ut.clone()

    # triton
    rt.requires_grad = True
    kt.requires_grad = True
    vt.requires_grad = True
    wt.requires_grad = True
    ut.requires_grad = True
    
    # cuda 
    rc.requires_grad = True
    kc.requires_grad = True
    vc.requires_grad = True
    wc.requires_grad = True
    uc.requires_grad = True

    # triton
    wt_ = th.log(th.sigmoid(wt)) # F.logsigmoid(wt)
    ot, state = fused_recurrent_rwkv6(rt, kt, vt, wt_, ut, scale=1.0)
    ot.mean().backward()

    # cuda
    wc_ = wc.sigmoid()
    uc_ = uc.view(1, H, 1, K)
    state = th.zeros(B, H, K, V, device=rc.device, dtype=rc.dtype)
    oc, state = rwkv_inner(rc, kc, vc, wc_, uc_, state)
    oc.mean().backward()

    # # outputs failed
    # assert th.allclose(oc, ot, atol=1.), breakpoint()
    # # gradients failed
    # assert th.allclose(uc.grad, ut.grad, atol=1e-4), breakpoint()
    # assert th.allclose(wc.grad, wt.grad, atol=1e-5), breakpoint()
    # assert th.allclose(kc.grad, kt.grad, atol=1e-5), breakpoint()
    # assert th.allclose(vc.grad, vt.grad, atol=1e-5), breakpoint()
    # assert th.allclose(rc.grad, rt.grad, atol=1e-5), breakpoint()

    assert th.allclose(oc, ot, atol=1.)
    # gradients failed
    assert th.allclose(uc.grad, ut.grad, atol=1e-4)
    assert th.allclose(wc.grad, wt.grad, atol=1e-5)
    assert th.allclose(kc.grad, kt.grad, atol=1e-5)
    assert th.allclose(vc.grad, vt.grad, atol=1e-5)
    assert th.allclose(rc.grad, rt.grad, atol=1e-5)


SCALES = [
    [1024, 512, 8],
    [1024, 512, 4],
    [1024, 512, 2],
    [1024, 512, 1],
    [2 * 1024, 512, 8],
    [2 * 1024, 512, 4],
    [2 * 1024, 512, 2],
    [2 * 1024, 512, 1],
    [4 * 1024, 512, 8],
    [4 * 1024, 512, 4],
    [4 * 1024, 512, 2],
    [4 * 1024, 512, 1],
    [4 * 1024, 4096, 16],
    [4 * 1024, 2048, 8],
    [4 * 1024, 1024, 8],
    [4 * 1024, 1024, 4],
    [8 * 1024, 512, 8],
    [8 * 1024, 512, 4],
    [8 * 1024, 512, 2],
    [8 * 1024, 512, 1],
    [1 * 1024, 2 * 512, 2 * 8],
    [1 * 1024, 2 * 512, 2 * 4],
    [1 * 1024, 2 * 512, 2 * 2],
    [1 * 1024, 2 * 512, 2 * 1],
    [1 * 1024, 4 * 512, 4 * 8],
    [1 * 1024, 4 * 512, 4 * 4],
    [1 * 1024, 4 * 512, 4 * 2],
    [1 * 1024, 4 * 512, 4 * 1],
]

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["L", "KH", "H"],
        x_vals=SCALES,
        line_arg="method",
        line_vals=["torch", "triton_recurrent", "triton_chunked"],
        line_names=["Torch", "Triton Recurrent", "Triton Chunked"],
        styles=[("red", "-"), ("green", "-"), ("blue", "-")],
        ylabel="time, ms",
        plot_name="RWKV6 kernel Torch chunked vs Triton sequential vs Triton Chunked",
        args={},
    ),
)
def benchmark(L, KH, H, method):
    quantiles = [0.5, 0.2, 0.8]
    with th.device("cuda"):
        B = 1
        K = V = KH // H
        rt, kt, wt = th.randn(3, B, H, L, K, device="cuda", requires_grad=True).to(th.bfloat16)
        vt = th.randn(B, H, L, V, device="cuda", requires_grad=True).to(th.bfloat16)
        ut = th.randn(H, K, device="cuda", requires_grad=True).to(th.bfloat16)

    def step():
        match method:
            case "triton_recurrent":
                with th.enable_grad():
                    wt_ = th.log(th.sigmoid(wt)) # F.logsigmoid(wt)
                    ot, state = fused_recurrent_rwkv6(rt, kt, vt, wt_, ut, scale=1.0)
                    ot = ot.mean()
                ot.backward()
            case "triton_chunked":
                with th.enable_grad():
                    wt_ = th.log(th.sigmoid(wt))  # F.logsigmoid(wt)
                    ot, state = chunk_rwkv6(rt, kt, vt, wt_, ut, scale=1.0)
                    ot = ot.mean()
                ot.backward()
            case "torch":
                with th.enable_grad():
                    wt_ = th.sigmoid(wt)
                    ut_ = ut.view(1, H, 1, K)
                    state = th.zeros(B, H, K, V, device=rt.device, dtype=rt.dtype)
                    ot, state = rwkv_inner(rt, kt, vt, wt_, ut_, kv_state=state)
                    ot = ot.mean()
                ot.backward()

    return triton.testing.do_bench(step, quantiles=quantiles)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["L", "KH", "H"],
        x_vals=SCALES,
        line_arg="method",
        line_vals=["triton_recurrent", "triton_chunked_fw_recurrent_bw"],
        line_names=["Triton Recurrent", "Triton Chunked Fw Recurrent Bw"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="time, ms",
        plot_name="RWKV6Hypno Triton sequential vs Triton Chunked Fw + Recurrent Bw",
        args={},
    ),
)
def benchmark_v6hypno(L, KH, H, method):
    quantiles = [0.5, 0.2, 0.8]
    with th.device("cuda"):
        B = 1
        K = V = KH // H
        rt, kt, wt = th.randn(3, B, H, L, K, device="cuda", requires_grad=True).to(th.bfloat16)
        vt = th.randn(B, H, L, V, device="cuda", requires_grad=True).to(th.bfloat16)
        ut = th.randn(H, K, V, device="cuda", requires_grad=True).to(th.bfloat16)

    def step():
        match method:
            case "triton_recurrent":
                with th.enable_grad():
                    wt_ = th.log(th.sigmoid(wt)) # F.logsigmoid(wt)
                    ot, state = fused_recurrent_rwkv6hypno(rt, kt, vt, wt_, ut, scale=1.0)
                    ot = ot.mean()
                ot.backward()
            case "triton_chunked_fw_recurrent_bw":
                with th.enable_grad():
                    wt_ = th.log(th.sigmoid(wt))  # F.logsigmoid(wt)
                    ot, state = chunked_fw_recurrent_bw_rwkv6hypno(rt, kt, vt, wt_, ut, scale=1.0)
                    ot = ot.mean()
                ot.backward()

    return triton.testing.do_bench(step, quantiles=quantiles)


if __name__ == "__main__":
    # benchmark.run(show_plots=True, print_data=True)
    benchmark_v6hypno.run(show_plots=True, print_data=True)