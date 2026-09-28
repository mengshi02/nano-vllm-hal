"""Operator families: the "what to run" half of the HAL.

Mirrors vLLM's ``_custom_ops`` family dispatch, but trimmed to what nano-vllm
actually needs.  Each family exposes a single forward function with one default
(torch-generic) implementation; vendors override a family entry under their
``backend_name`` and the runtime selects it via ``current_platform.backend_name``.

Families:

* ``forward_rms_norm``
* ``forward_rotary_emb``
* ``forward_kv_cache_store`` -- write K/V into the paged cache (triton kernel on
  CUDA, vendor fused op elsewhere)
* ``forward_attention``       -- prefill + decode attention

Phase 1 wires up the first three (pure, portable) families and keeps
``forward_attention`` on the existing flash-attn path selected by platform.  The
attention family is the biggest per-vendor job and is intentionally the seam
called out in the design doc for Ascend/Hygon/Moore.
"""

from __future__ import annotations

from typing import Any

from nanovllm.ops import attention, rms_norm, rope, kvcache

# Each registry maps backend_name -> callable.  Defaults are the torch-generic
# (or, for attention, flash-attn) implementations so that an unknown/impartial
# backend still runs.
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