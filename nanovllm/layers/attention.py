import torch
from torch import nn
import triton
import triton.language as tl
from typing import Optional

# Lazy import for flash attention to reduce startup time
_flash_attn_varlen_func = None
_flash_attn_with_kvcache = None

def _get_flash_attn():
    global _flash_attn_varlen_func, _flash_attn_with_kvcache
    if _flash_attn_varlen_func is None:
        from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
        _flash_attn_varlen_func = flash_attn_varlen_func
        _flash_attn_with_kvcache = flash_attn_with_kvcache
    return _flash_attn_varlen_func, _flash_attn_with_kvcache

from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # Pre-allocate tensor views to avoid repeated memory allocations
        batch_size = q.size(0)
        q_view = q.view(batch_size, self.num_heads, self.head_dim)
        k_view = k.view(batch_size, self.num_kv_heads, self.head_dim)
        v_view = v.view(batch_size, self.num_kv_heads, self.head_dim)
        
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        
        # Only store to cache if cache is allocated
        if k_cache.numel() > 0 and v_cache.numel() > 0:
            store_kvcache(k_view, v_view, k_cache, v_cache, context.slot_mapping)
        
        # Get flash attention functions lazily
        flash_attn_varlen_func, flash_attn_with_kvcache = _get_flash_attn()
        
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k_view, v_view = k_cache, v_cache
            o = flash_attn_varlen_func(q_view, k_view, v_view,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q_view.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        
        # Use contiguous view for better memory layout
        return o.view(batch_size, self.num_heads * self.head_dim).contiguous()
