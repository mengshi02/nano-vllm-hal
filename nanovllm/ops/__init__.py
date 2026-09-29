
from __future__ import annotations

from typing import Any

from nanovllm.ops import rms_norm, rope, kvcache

_RMS_NORM: dict[str, Any] = {"default": rms_norm.rms_norm_forward}
_ROTARY: dict[str, Any] = {"default": rope.rotary_emb_forward}
_KV_CACHE: dict[str, Any] = {"default": kvcache.store_kvcache}
_ATTENTION: dict[str, Any] = {}

def _register_default_attention():
    try:
        from nanovllm.ops import attention

        _ATTENTION["default"] = attention.attention_forward
    except ImportError:
        _ATTENTION["default"] = None

_register_default_attention()

def _register_ascend_ops():
    try:
        from nanovllm.ops import ascend_attention, ascend_kvcache

        _ATTENTION["ascend"] = ascend_attention.attention_forward
        _KV_CACHE["ascend"] = ascend_kvcache.store_kvcache_ascend
    except ImportError:
        pass

_register_ascend_ops()

def register(family: dict[str, Any], backend_name: str):
    def deco(fn):
        family[backend_name] = fn
        return fn

    return deco

def _pick(family: dict[str, Any], backend_name: str):
    return family.get(backend_name, family["default"])

def get_rms_norm_fn(backend_name: str):
    return _pick(_RMS_NORM, backend_name)

def get_rotary_fn(backend_name: str):
    return _pick(_ROTARY, backend_name)

def get_kv_cache_fn(backend_name: str):
    return _pick(_KV_CACHE, backend_name)

def get_attention_fn(backend_name: str):
    fn = _ATTENTION.get(backend_name, _ATTENTION["default"])
    if fn is None:
        raise ValueError(f"no attention implementation for backend {backend_name}")
    return fn
