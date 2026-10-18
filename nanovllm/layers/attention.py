import torch
from torch import nn

from nanovllm.ops import get_attention_fn, get_kv_cache_fn
from nanovllm.platforms import current_platform
from nanovllm.utils.context import get_context


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
        self._store_kvcache = get_kv_cache_fn(current_platform.backend_name)
        self._attention = get_attention_fn(current_platform.backend_name)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            self._store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.prefill_context is not None:
            n = context.num_decode_tokens
            decoded = self._attention(q[:n], k[:n], v[:n], k_cache, v_cache,
                                      self.scale, context.decode_context)
            prefetched = self._attention(q[n:], k[n:], v[n:], k_cache, v_cache,
                                         self.scale, context.prefill_context)
            # Decode kernels may return [batch, 1, heads, dim], while prefill
            # kernels return [tokens, heads, dim]. Normalize before combining.
            o = torch.cat((decoded.reshape_as(q[:n]), prefetched.reshape_as(q[n:])))
        else:
            o = self._attention(q, k, v, k_cache, v_cache, self.scale, context)
        return o
