
from __future__ import annotations

import torch


def store_kvcache_musa(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    num_heads, head_dim = key.shape[-2:]
    k_flat = k_cache.view(-1, num_heads * head_dim)
    v_flat = v_cache.view(-1, num_heads * head_dim)
    slots = slot_mapping.long()
    valid = slots >= 0
    k_flat[slots[valid]] = key[valid].reshape(-1, num_heads * head_dim)
    v_flat[slots[valid]] = value[valid].reshape(-1, num_heads * head_dim)
