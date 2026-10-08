from __future__ import annotations

import torch

from nanovllm.utils.context import Context


def _expand_kv_mha(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    if x.shape[-2] == num_heads:
        return x
    ratio = num_heads // x.shape[-2]
    x = x.repeat_interleave(ratio, dim=-2)
    return x


def attention_forward_musa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
    context: Context,
) -> torch.Tensor:
    try:
        from flash_attn import flash_attn_varlen_func
    except ImportError:
        from flash_attn_3.interface import flash_attn_varlen_func

    num_heads = q.shape[-2]
    if context.is_prefill:
        k = _expand_kv_mha(k, num_heads)
        v = _expand_kv_mha(v, num_heads)
        o = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=context.cu_seqlens_q,
            cu_seqlens_k=context.cu_seqlens_k,
            max_seqlen_q=context.max_seqlen_q,
            max_seqlen_k=context.max_seqlen_k,
            softmax_scale=scale,
            causal=True,
        )
        return o
    bs = q.shape[0]
    if k_cache.dim() < 2:
        o = flash_attn_varlen_func(
            q, _expand_kv_mha(k, num_heads), _expand_kv_mha(v, num_heads),
            cu_seqlens_q=torch.arange(0, bs + 1, dtype=torch.int32, device=q.device),
            cu_seqlens_k=torch.arange(0, bs + 1, dtype=torch.int32, device=q.device),
            max_seqlen_q=1,
            max_seqlen_k=1,
            softmax_scale=scale,
            causal=True,
        )
        return o
    block_size = k_cache.shape[1]
    pages = context.block_tables.long()
    num_pages = pages.shape[1]
    gk = k_cache[pages]
    gv = v_cache[pages]
    dk = gk.reshape(bs, num_pages * block_size, *k_cache.shape[-2:])
    dv = gv.reshape(bs, num_pages * block_size, *v_cache.shape[-2:])
    lens = context.context_lens[:bs]
    dk_flat = torch.cat([dk[i, : int(lens[i])] for i in range(bs)], dim=0)
    dv_flat = torch.cat([dv[i, : int(lens[i])] for i in range(bs)], dim=0)
    dk_flat = _expand_kv_mha(dk_flat, num_heads).contiguous()
    dv_flat = _expand_kv_mha(dv_flat, num_heads).contiguous()
    cu_q = torch.arange(0, bs + 1, dtype=torch.int32, device=q.device)
    cu_k = torch.zeros(bs + 1, dtype=torch.int32, device=q.device)
    cu_k[1:] = torch.cumsum(lens, dim=0)
    o = flash_attn_varlen_func(
        q, dk_flat, dv_flat,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=1,
        max_seqlen_k=int(lens.max()),
        softmax_scale=scale,
        causal=True,
    )
    return o


attention_forward = attention_forward_musa
