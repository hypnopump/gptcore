import abc

from util.config import Factory

from typing import Any, Optional, Tuple, List, Iterable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class IBiasMask():
    @abc.abstractmethod
    def forward(self, q:Tensor):
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, q:Tensor):
        raise NotImplementedError

class IMulMask():
    @abc.abstractmethod
    def forward(self, q:Tensor):
        raise NotImplementedError
    
    @abc.abstractmethod
    def __call__(self, q:Tensor):
        raise NotImplementedError

def causal_mul_mask_inf(T):
    mask = torch.ones(T, T)
    mask = mask.masked_fill(mask.tril() == 0, float('-inf')) # (T, T)
    return mask

def causal_mul_mask_zeros(T):
    mask = torch.ones(T, T).tril() # (T, T)
    return mask

def causal_bias_mask(T):
    return torch.full((T, T), float('-inf')).triu(1)

class NoBiasMask(nn.Module, IBiasMask):
    def __init__(self, block_size : int, n_heads : int, layer_id : int):
        super().__init__()

    def forward(self, q:Tensor):
        return 0.0

class NoMulMask(nn.Module, IMulMask):
    def __init__(self, block_size : int, n_heads : int, layer_id : int):
        super().__init__()

    def forward(self, q:Tensor):
        return 1.0

class CausalMulMaskInf(nn.Module, IMulMask):
    # using first instance as a cache so we don't waste memory by duplicating our registered buffers per layer
    cache = None
    def __init__(self, block_size : int, n_heads : int, layer_id : int):
        super().__init__()
        if CausalMulMaskInf.cache is None:
            CausalMulMaskInf.cache = self
            T = block_size
            self.register_buffer('mask', causal_mul_mask_inf(T))

    def forward(self, q:Tensor):
        return CausalMulMaskInf.cache.mask

class CausalMulMaskZeros(nn.Module, IMulMask):
    # using first instance as a cache so we don't waste memory by duplicating our registered buffers per layer
    cache = None
    def __init__(self, block_size : int, n_heads : int, layer_id : int):
        super().__init__()
        if CausalMulMaskZeros.cache is None:
            CausalMulMaskZeros.cache = self
            T = block_size
            self.register_buffer('mask', causal_mul_mask_zeros(T))

    def forward(self, q:Tensor):
        return CausalMulMaskZeros.cache.mask

class CausalBiasMask(nn.Module, IBiasMask):
    # using first instance as a cache so we don't waste memory by duplicating our registered buffers per layer
    cache = None
    def __init__(self, block_size : int, n_heads : int, layer_id : int):
        super().__init__()
        if CausalBiasMask.cache is None:
            CausalBiasMask.cache = self
            T = block_size
            self.register_buffer('mask', causal_bias_mask(T))

    def forward(self, q:Tensor):
        return CausalBiasMask.cache.mask

def alibi_mask(T: int, H: int, idxs: Tensor | None = None) -> Tensor:
    if idxs is None:
        idxs = torch.arange(T)[None]
    bias = (idxs[:, None, :] - idxs[:, :, None]).float()  # (B, T, T)
    bias = bias + causal_bias_mask(T)  # (B, T, T)
    bias = bias.expand(H, -1, -1)  # (B, H, T, T)
    head_bias_slopes = (2 ** torch.linspace(-8.0/H, -8.0, H))[None, :, None, None]  # (B, H, 1, 1)
    bias = bias * head_bias_slopes  # (B, H, T, T)
    return bias

class AlibiMask(nn.Module, IBiasMask):
    # using first instance as a cache so we don't waste memory by duplicating our registered buffers per layer
    cache = None
    def __init__(self, block_size : int, n_heads : int, layer_id : int, use_cache: bool = True):
        super().__init__()
        self.use_cache = use_cache
        self.h = n_heads
        if AlibiMask.cache is None:
            AlibiMask.cache = self
            T = block_size
            H = n_heads
            self.register_buffer('mask', alibi_mask(T, H))

    def forward(self, q: Tensor, idxs: Tensor | None = None):
        if self.use_cache and idxs is None:
            return AlibiMask.cache.mask[:, :, :q.size(-2), :q.size(-2)]
        elif idxs is not None:
            if self.use_cache is False:
                NotImplementedError("idxs is not None and use_cache is False. Gen mask using idxs? then select mask")
            else:
                b, l = q.size(0), q.size(-2)
                _, h, t1, t2 = AlibiMask.cache.mask.shape
                mask = AlibiMask.cache.mask

            # select only positions of mask that are in idxs
            bmask = mask.repeat(b, 1, 1, 1)
            idx_mask = mask.new_zeros(b, t1, dtype=torch.bool)
            idx_mask.scatter_(1, idxs, True)

            i_idx_mask = idx_mask[:, None, :].repeat(1, h, 1)
            j_idx_mask = idx_mask[:, None, None, :].repeat(1, h, l, 1)

            # (B, H, I, J) -> (B, H, T1, J) -> (B, H, T1, T2)
            return bmask[i_idx_mask].reshape(b, h, l, -1)[j_idx_mask].reshape(b, h, l, l)
