# Changelog

## 0.3.1 - 2026-10-22

### Removed

- Removed the legacy `TP=1, DP=EP` OLMoE expert layout. Expert parallelism is now supported only as a subdimension of tensor parallelism (`EP | TP`).

### Changed

- Replaced internal node identifiers in validation notes with generic two-node S5000 descriptions.
- OLMoE `EP` configuration now requires `EP` to divide `TP`; the previous `DP=EP` binding is rejected.

## 0.3.0 - 2026-10-22

### Added

- Unified parallel topology for dense Qwen3 and sparse OLMoE models: `DP x TP x PP`, with MoE expert parallelism as a TP subdimension (`EP | TP`).
- OLMoE-1B-7B inference support with expert-aware weight sharding and the legacy `TP=1, DP=EP` layout.
- Radix prefix reuse with 16-token KV pages, including the slower eager gather path used for fine-grained CUDA reuse.
- Cross-host prefill/decode separation with matching producer/decoder topology and prefix-aware KV transfer.
- Validation records for NVIDIA H20 and Moore Threads S5000, including cross-machine PD on two S5000 machines.

### Fixed

- Tensor-parallel step statistics are now collected before scheduler postprocessing clears per-sequence scheduled-token counters.
- OLMoE parity tooling now uses the active platform device instead of hardcoding CUDA, allowing the same checks to run on MUSA.

### Validation

- Dense Qwen3 and sparse OLMoE passed configuration, layout, scheduler, expert sharding, teacher-forced logits, chunked prefill, mixed prefill/decode, DP/TP/EP/PP parity, and PD smoke tests on H20.
- OLMoE passed the same core topology checks on MUSA S5000. `TP=2/EP=2` and `TP=2/EP=1` matched TP=1 top-1 choices over eight teacher-forced steps with mean logit RMS around 0.13-0.15, consistent with BF16 reduction noise.
- Cross-machine PD on two S5000 machines matched 12/12 generated tokens for `TP=1/PP=1` and `TP=2/PP=1`; the repeated request reused one prefix page.

### Known limitations

- Dense models require `EP=1`; OLMoE requires `EP` to divide both `TP` and the expert count.
- PD V1 requires producer and decoder to use identical `DP/TP/PP/EP` configurations. Asymmetric P/D layouts and KV resharding remain future work.
- The clean H20 validation used the SDPA fallback rather than FlashAttention. The CUDA 16-token KV-page path uses eager gather and is slower than the default 256-token path.
- BF16 tensor-parallel reductions can produce small logit differences; free-generation argmax may occasionally differ even when teacher-forced parity is valid.

### Packaging

- Test files are retained in the development repository but excluded from source distributions. Wheels contain only the `nanovllm` package.
