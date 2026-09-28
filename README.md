# nano-vllm-hal

A lightweight vLLM implementation (~1,200 lines) with a **Hardware Abstraction Layer (HAL)**, rebuilt from [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).

One engine, four hardware targets:

| Vendor | Runtime | Comm backend | Attention / KV entry point |
|---|---|---|---|
| NVIDIA | `torch.cuda` | `nccl` | triton + flash-attn (default, reference) |
| Hygon DCU | ROCm | `nccl` (ROCm build) | ROCm fused ops |
| Ascend NPU | `torch_npu` | `hccl` | `torch_npu._p_attention` etc. |
| Moore Threads | `torch_musa` | `mccl` | MUSA fusion kernels |

## Design

Following vLLM's `Platform`/`Hardware` split: **what to compute** (engine, model) and **how to compute it** (device, kernels) are strictly separated.

- `nanovllm/platforms/` — one `Platform` class per hardware backend. `current_platform` is resolved once at import (`Ascend → MUSA → ROCm → CUDA → CPU`) and the engine only ever talks to it.
- `nanovllm/ops/` — operator families (attention / kv-cache store / rms-norm / rope). Each family has a torch-generic default; a vendor registers its kernel under its `backend_name` and the runtime picks it via `current_platform.backend_name`.
- `nanovllm/engine/`, `nanovllm/layers/`, `nanovllm/models/` — hardware-agnostic. Adding a new backend never touches these.

```python
from nanovllm.ops import register, _KV_CACHE

@register(_KV_CACHE, "musa")
def store_kvcache_musa(key, value, k_cache, v_cache, slot_mapping):
    ...  # vendor kernel
```

## Usage

```python
from nanovllm import LLM, SamplingParams

llm = LLM("Qwen/Qwen3-0.6B", enforce_eager=True)
prompts = ["introduce yourself", "list all prime numbers within 100"]
outputs = llm.generate(prompts, SamplingParams(temperature=0.6, max_tokens=256))
```

Or run the included examples:

```bash
python example.py   # end-to-end smoke
python bench.py      # throughput benchmark
```

Features: continuous batching, paged KV cache (prefix caching, block reuse), tensor parallelism, CUDA-graph decode, torch.compile.

## Adding a new backend

1. Add `nanovllm/platforms/<vendor>.py` implementing the `Platform` interface (`backend_name` / `device_name` / `dispatch_key` + device, memory, comm, graph-capture methods).
2. Insert the class into `_PLATFORMS` in `nanovllm/platforms/__init__.py` (before `CUDAPlatform` if the vendor runtime is CUDA-compatible).
3. Register vendor kernels under `nanovllm/ops/` with `@register(family, "<backend_name>")`.
4. Only if the backend needs model-level dispatch differences, branch on `current_platform.dispatch_key` inside `nanovllm/models/`.

The engine stays untouched.

## Status

- [x] Phase 1: HAL skeleton — platforms + op families, engine rewired to `current_platform`
- [x] CUDA smoke verified on H20 (Qwen3-0.6B, zero behavior change vs upstream)
- [ ] Ascend / ROCm / MUSA real kernels (stubs in place, pending vendor SDKs)
