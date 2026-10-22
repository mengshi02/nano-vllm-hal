from __future__ import annotations

import torch

from nanovllm.ops.attention_gather import attention_forward_gather
from nanovllm.utils.context import Context


def attention_forward_musa(q, k, v, k_cache, v_cache, scale, context):
    try:
        from flash_attn_3.interface import flash_attn_varlen_func
    except ImportError:
        from flash_attn import flash_attn_varlen_func
    return attention_forward_gather(q, k, v, k_cache, v_cache, scale,
                                    context, flash_attn_varlen_func)


attention_forward = attention_forward_musa
