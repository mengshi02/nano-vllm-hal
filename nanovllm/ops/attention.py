
from __future__ import annotations

import torch
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from nanovllm.utils.context import Context

def attention_forward_flash(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
    context: Context,
) -> torch.Tensor:
    if context.is_prefill:
        if context.block_tables is not None:
            k, v = k_cache, v_cache
        o = flash_attn_varlen_func(
            q,
            k,
            v,
            max_seqlen_q=context.max_seqlen_q,
            cu_seqlens_q=context.cu_seqlens_q,
            max_seqlen_k=context.max_seqlen_k,
            cu_seqlens_k=context.cu_seqlens_k,
            softmax_scale=scale,
            causal=True,
            block_table=context.block_tables,
        )
    else:
        o = flash_attn_with_kvcache(
            q.unsqueeze(1),
            k_cache,
            v_cache,
            cache_seqlens=context.context_lens,
            block_table=context.block_tables,
            softmax_scale=scale,
            causal=True,
        )
    return o

attention_forward = attention_forward_flash
