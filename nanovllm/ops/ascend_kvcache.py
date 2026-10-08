from __future__ import annotations

import torch


def store_kvcache_ascend(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    import torch_npu

    torch_npu.npu_scatter_pa_kv_cache(
        key=key.contiguous(),
        value=value.contiguous(),
        key_cache=k_cache,
        value_cache=v_cache,
        slot_mapping=slot_mapping.contiguous(),
        cache_mode="Norm",
    )
