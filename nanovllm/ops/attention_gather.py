from __future__ import annotations

import torch

from nanovllm.utils.context import Context


def _expand_kv_mha(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    if x.shape[-2] == num_heads:
        return x
    ratio = num_heads // x.shape[-2]
    return x.repeat_interleave(ratio, dim=-2)


def _gather_paged_kv(k_cache, v_cache, block_tables, lens, num_heads):
    batch = len(lens)
    block_size = k_cache.shape[1]
    pages = block_tables.long()
    num_pages = pages.shape[1]
    gathered_k = k_cache[pages].reshape(batch, num_pages * block_size, *k_cache.shape[-2:])
    gathered_v = v_cache[pages].reshape(batch, num_pages * block_size, *v_cache.shape[-2:])
    k = torch.cat([gathered_k[i, :int(lens[i])] for i in range(batch)])
    v = torch.cat([gathered_v[i, :int(lens[i])] for i in range(batch)])
    return _expand_kv_mha(k, num_heads).contiguous(), _expand_kv_mha(v, num_heads).contiguous()


def attention_forward_gather(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
    context: Context,
    flash_attn_varlen_func,
) -> torch.Tensor:
    """Dense varlen fallback for runtimes that lack a native paged kernel."""
    num_heads = q.shape[-2]
    if context.is_prefill:
        if context.block_tables is not None:
            # A resumed chunk or a shared prefix reads previously stored keys.
            lens = context.cu_seqlens_k[1:] - context.cu_seqlens_k[:-1]
            k, v = _gather_paged_kv(k_cache, v_cache, context.block_tables, lens, num_heads)
        else:
            k = _expand_kv_mha(k, num_heads)
            v = _expand_kv_mha(v, num_heads)
        return flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=context.cu_seqlens_q,
            cu_seqlens_k=context.cu_seqlens_k,
            max_seqlen_q=context.max_seqlen_q,
            max_seqlen_k=context.max_seqlen_k,
            softmax_scale=scale,
            causal=True,
        )

    bs = q.shape[0]
    if k_cache.dim() < 2:
        return flash_attn_varlen_func(
            q, _expand_kv_mha(k, num_heads), _expand_kv_mha(v, num_heads),
            cu_seqlens_q=torch.arange(0, bs + 1, dtype=torch.int32, device=q.device),
            cu_seqlens_k=torch.arange(0, bs + 1, dtype=torch.int32, device=q.device),
            max_seqlen_q=1,
            max_seqlen_k=1,
            softmax_scale=scale,
            causal=True,
        )

    lens = context.context_lens[:bs]
    dk_flat, dv_flat = _gather_paged_kv(k_cache, v_cache, context.block_tables, lens, num_heads)
    cu_q = torch.arange(0, bs + 1, dtype=torch.int32, device=q.device)
    cu_k = torch.zeros(bs + 1, dtype=torch.int32, device=q.device)
    cu_k[1:] = torch.cumsum(lens, dim=0)
    return flash_attn_varlen_func(
        q, dk_flat, dv_flat,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=1,
        max_seqlen_k=int(lens.max()),
        softmax_scale=scale,
        causal=True,
    )
