from __future__ import annotations

import torch

from nanovllm.utils.context import Context


def _causal_mask(cu_seqlens: list[int], total: int, device) -> torch.Tensor:
    m = torch.tril(torch.ones(total, total, dtype=torch.bool, device=device))
    m = torch.where(m, False, True)
    for i in range(len(cu_seqlens) - 1):
        s, e = cu_seqlens[i], cu_seqlens[i + 1]
        for j in range(i + 1, len(cu_seqlens) - 1):
            s2, e2 = cu_seqlens[j], cu_seqlens[j + 1]
            m[s:e, s2:e2] = True
    return m


def _gather_paged_kv_to_dense(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    block_size = k_cache.shape[1]
    seq_lens = context_lens.tolist()
    max_seq_len = max(seq_lens)
    num_blocks = (max_seq_len + block_size - 1) // block_size
    blocks = block_tables[: len(seq_lens), :num_blocks].long()
    gathered_k = k_cache[blocks]
    gathered_v = v_cache[blocks]
    lens = torch.tensor(seq_lens, dtype=torch.long, device=k_cache.device)
    mask = torch.arange(block_size, device=k_cache.device)[None, None, :] < lens[:, None, None]
    dense_k = gathered_k.permute(0, 1, 3, 2, 4).reshape(len(seq_lens), num_blocks * block_size, num_kv_heads, head_dim)
    dense_v = gathered_v.permute(0, 1, 3, 2, 4).reshape(len(seq_lens), num_blocks * block_size, num_kv_heads, head_dim)
    lens_per_token = mask.expand(-1, num_blocks, -1).reshape(len(seq_lens), num_blocks * block_size)
    dense_k = dense_k[lens_per_token].reshape(-1, num_kv_heads, head_dim)
    dense_v = dense_v[lens_per_token].reshape(-1, num_kv_heads, head_dim)
    actual_seq_lengths_kv = []
    cumsum = 0
    for length in seq_lens:
        cumsum += length
        actual_seq_lengths_kv.append(cumsum)
    return dense_k, dense_v, actual_seq_lengths_kv


def _paged_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
    context: Context,
) -> torch.Tensor:
    import torch_npu

    num_heads = q.shape[-2]
    head_dim = q.shape[-1]
    num_kv_heads = k_cache.shape[-2]
    context_lens = torch.tensor(
        context.cu_seqlens_k.tolist(), dtype=torch.int32, device=q.device
    ) if context.context_lens is None else context.context_lens
    k, v, actual_seq_lengths_kv = _gather_paged_kv_to_dense(
        k_cache, v_cache, context.block_tables, context_lens, num_kv_heads, head_dim
    )
    seq_lens = actual_seq_lengths_kv
    qlens = context.cu_seqlens_q.tolist()
    o = torch_npu.npu_fusion_attention(
        query=q,
        key=k,
        value=v,
        head_num=num_heads,
        input_layout="TND",
        scale=scale,
        pre_tockens=2147483647,
        next_tockens=0,
        actual_seq_qlen=qlens,
        actual_seq_kvlen=seq_lens,
        sparse_mode=0,
        atten_mask=_causal_mask(qlens, q.shape[0], q.device),
    )[0]
    return o


def attention_forward_ascend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
    context: Context,
) -> torch.Tensor:
    import torch_npu

    num_heads = q.shape[-2]
    head_dim = q.shape[-1]
    has_cache = k_cache.dim() >= 2
    if context.is_prefill:
        if has_cache and context.block_tables is not None:
            return _paged_prefill(q, k_cache, v_cache, scale, context)
        o = torch_npu.npu_fusion_attention(
            query=q,
            key=k,
            value=v,
            head_num=num_heads,
            input_layout="TND",
            scale=scale,
            pre_tockens=2147483647,
            next_tockens=0,
            actual_seq_qlen=context.cu_seqlens_q.tolist(),
            actual_seq_kvlen=context.cu_seqlens_k.tolist(),
            sparse_mode=0,
            atten_mask=_causal_mask(context.cu_seqlens_q.tolist(), q.shape[0], q.device),
        )[0]
        return o
    if not has_cache:
        o = torch_npu.npu_fusion_attention(
            query=q,
            key=k,
            value=v,
            head_num=num_heads,
            input_layout="TND",
            scale=scale,
            pre_tockens=2147483647,
            next_tockens=0,
            actual_seq_qlen=[1] * q.shape[0],
            actual_seq_kvlen=context.context_lens.tolist(),
            sparse_mode=0,
        )[0]
        return o
    num_kv_heads = k_cache.shape[-2]
    block_size = k_cache.shape[1]
    k_flat = k_cache.view(k_cache.shape[0], block_size, -1)
    v_flat = v_cache.view(v_cache.shape[0], block_size, -1)
    o = torch_npu.npu_fused_infer_attention_score(
        query=q.unsqueeze(1),
        key=k_flat,
        value=v_flat,
        input_layout="BSND",
        block_table=context.block_tables,
        block_size=block_size,
        actual_seq_lengths=[1] * q.shape[0],
        actual_seq_lengths_kv=context.context_lens.tolist(),
        num_key_value_heads=num_kv_heads,
        num_heads=num_heads,
        scale=scale,
        sparse_mode=0,
        pre_tokens=2147483647,
        next_tokens=0,
    )[0]
    return o


attention_forward = attention_forward_ascend
