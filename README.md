<p align="center">
  <img src="assets/logo.optimized.png" alt="nano-vllm-hal" width="560">
</p>

# nano-vllm-hal

A small, readable inference engine based on [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm), extended with hardware abstraction, tensor/pipeline parallelism, and educational prefill/decode (PD) separation. It runs Qwen3 on NVIDIA CUDA, Moore Threads MUSA, Ascend NPU, and Hygon DCU with the appropriate runtimes.

## Quick start

Install the dependencies for your hardware runtime and set the local Qwen3 model path in `example.py` and `bench.py` before running them:

```bash
python example.py
python bench.py
```

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/path/to/Qwen3-0.6B", enforce_eager=True)
outputs = llm.generate(["Introduce yourself"], SamplingParams(max_tokens=32))
```

The engine supports continuous batching, paged KV cache with prefix reuse, chunked prefill, TP, and synchronous PP. For PD separation, start a decode process and connect a separate prefill process:

```bash
# Decode host
python3 pd_remote.py decode /path/to/Qwen3-0.6B --bind 0.0.0.0
# Prefill host
python3 pd_remote.py prefill /path/to/Qwen3-0.6B --host D_HOST --cross-block --compare
```

P and D need matching model weights and TP/PP settings. This learning-oriented path uses CPU-staged KV transfer over TCP; see the guide for multi-card layouts, multi-P/D routing, verification, and limitations.