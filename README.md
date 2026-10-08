<p align="center">
  <img src="assets/logo.optimized.png" alt="nano-vllm-hal" width="560">
</p>

**A complete miniature of vLLM v1, in ~2,000 lines — the codebase to read first if you want to understand vLLM.**

Built on top of [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) (~1,200 lines, a from-scratch lightweight vLLM), we added **~700 lines** that turn it into a faithful, scaled-down vLLM v1: the same `Platform`/op-family hardware abstraction, the same continuous batching / paged KV cache engine, the same What-vs-How separation — small enough to read in an afternoon, real enough to run Qwen3 end-to-end on production hardware.

| | lines | what it is |
|---|---|---|
| engine core (inherited) | ~1,200 | scheduler, block manager, paged KV cache, TP, CUDA graph — the "What" |
| **HAL layer (added)** | **~700** | `platforms/` + `ops/` — device, memory, comm, kernels — the "How" |
| **total** | **~2,000** | a readable, runnable miniature of vLLM v1 |

One engine, four hardware targets:

| Vendor | Runtime | Comm backend | Attention / KV entry point |
|---|---|---|---|
| NVIDIA | `torch.cuda` | `nccl` | triton + flash-attn (default, reference) |
| Hygon DCU | DTK (HIP) | `nccl` (HIP build) | triton + flash-attn (shares the default family) |
| Ascend NPU | `torch_npu` | `hccl` | `torch_npu` fused attention ops |
| Moore Threads | `torch_musa` | `mccl` | MUSA fusion kernels |

## Design

Mirrors vLLM v1's `Platform`/`Hardware` split: **what to compute** (engine, model) and **how to compute it** (device, kernels) are strictly separated.

- `nanovllm/platforms/` — one `Platform` class per backend. `current_platform` is resolved once at import (`Ascend → MUSA → Hygon → ROCm → CUDA → CPU`) and the engine only ever talks to it: device, memory, process group, graph capture.
- `nanovllm/ops/` — operator families (attention / kv-cache store / rms-norm / rope). Each family has a default implementation; a vendor registers its kernel under its `backend_name` and the runtime picks it via `current_platform.backend_name`.
- `nanovllm/engine/`, `nanovllm/layers/`, `nanovllm/models/` — hardware-agnostic, never change when a backend is added.

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
python bench.py     # throughput benchmark
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
- [x] CUDA production-ready on H20: bench.py 256 concurrent sequences, 134k tokens, 7,570 tok/s (exclusive GPU, NGC public image), 3,388 on a shared GPU; zero failures
- [x] Ascend 910B4 bring-up: real kernels via `torch_npu` fused attention ops; bench.py 256 concurrent sequences, 134k tokens, 612 tok/s, zero failures
- [x] Moore Threads S5000 bring-up: real kernels via MUSA `mate` attention; bench.py 256 concurrent sequences, 134k tokens, 236 tok/s, zero failures
- [x] Hygon DCU BW1102 bring-up: DTK/HIP runtime reuses the default triton + flash-attn family unchanged; bench.py 256 concurrent sequences, 134k tokens, 4,326 tok/s, zero failures
