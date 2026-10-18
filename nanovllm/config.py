import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    tp_fp32_matmul: bool = False
    pipeline_parallel_size: int = 1
    device_offset: int = 0
    worker_timeout: float = 600.0
    init_method: str = ""
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        if self.device_offset < 0:
            raise ValueError("device_offset must be non-negative")
        if self.worker_timeout <= 0:
            raise ValueError("worker_timeout must be positive")
        self.hf_config = AutoConfig.from_pretrained(self.model)
        if not 1 <= self.pipeline_parallel_size <= self.hf_config.num_hidden_layers:
            raise ValueError("pipeline_parallel_size must be between 1 and num_hidden_layers")
        if (self.hf_config.num_attention_heads % self.tensor_parallel_size or
                self.hf_config.num_key_value_heads % self.tensor_parallel_size or
                self.hf_config.vocab_size % self.tensor_parallel_size):
            raise ValueError("attention heads, KV heads and vocabulary must be divisible by tensor_parallel_size")
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
