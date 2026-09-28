
from __future__ import annotations

from typing import Any

from nanovllm.ops import attention, rms_norm, rope, kvcache

_RMS_NORM: dict[str, Any] = {"default": rms_norm.rms_norm_forward}
_ROTARY: dict[str, Any] = {"default": rope.rotary_emb_forward}
_KV_CACHE: dict[str, Any] = {"default": kvcache.store_kvcache}
_ATTENTION: dict[str, Any] = {"default": attention.attention_forward}

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
    return _pick(_ATTENTION, backend_name)
