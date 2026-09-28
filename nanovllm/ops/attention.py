"""Attention op family -- default flash-attn implementation (CUDA).

The signature is the seam every vendor plugs into: each backend registers a
function that takes the projected q/k/v plus the paged k/v cache and the current
``Context`` and returns the attention output.  Hyper-parameters that vendors
need (the number of heads, head dim, scale) are carried on the ``Attention``
module and passed in, so a vendor kernel never needs to reach back into the
engine.
"""

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
        # Prefix cache: when block_tables is set, q's full k/v lives in the
        # cache, so read from the paged cache instead of the freshly-project
        # (and chunked) k/v.
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