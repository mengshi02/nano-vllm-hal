from dataclasses import dataclass

import torch

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams


@dataclass
class KVHandoff:
    prompt_token_ids: list[int]
    first_token_id: int
    sampling_params: SamplingParams
    kv_blocks: torch.Tensor | list[torch.Tensor]
    model: str
    block_size: int


class KVLease:
    """P owns these physical KV blocks until D acknowledges the read."""

    def __init__(self, engine, sequence, first_token_id, sampling_params):
        self.engine = engine
        self.sequence = sequence
        self.first_token_id = first_token_id
        self.sampling_params = sampling_params
        self.released = False

    def read(self, start_blocks: int = 0):
        if self.released:
            raise RuntimeError("KV lease has been released")
        if not 0 <= start_blocks < len(self.sequence.block_table):
            raise ValueError("Invalid KV suffix")
        block_ids = self.sequence.block_table[start_blocks:]
        num_tokens = self.sequence.num_prompt_tokens - start_blocks * self.engine.config.kvcache_block_size
        if (getattr(self.engine.config, "tensor_parallel_size", 1) *
                getattr(self.engine.config, "pipeline_parallel_size", 1) > 1):
            return self.engine.model_runner.call_all(
                "export_kv", block_ids, num_tokens)
        return self.engine.model_runner.call(
            "export_kv", block_ids, num_tokens)

    def release(self):
        if not self.released:
            self.released = True
            self.engine.scheduler.block_manager.deallocate(self.sequence)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()


class DisaggregatedLLM:
    """Educational 1P/1D engines with matching TP/PP layouts; requests run in order."""

    def __init__(self, model: str, devices: tuple[int, int] = (0, 1), **kwargs):
        world_size = kwargs.get("tensor_parallel_size", 1) * kwargs.get("pipeline_parallel_size", 1)
        if (len(devices) != 2 or min(devices) < 0 or
                set(range(devices[0], devices[0] + world_size)) &
                set(range(devices[1], devices[1] + world_size))):
            raise ValueError("P and D need disjoint contiguous ranges of TP × PP devices")
        if "device_offset" in kwargs:
            raise ValueError("Use devices=(prefill_device, decode_device)")

        self.prefill = LLMEngine(model, device_offset=devices[0], **kwargs)
        try:
            self.decode = LLMEngine(model, device_offset=devices[1], **kwargs)
        except BaseException:
            self.prefill.exit()
            raise
        self.decode.scheduler.decode_only = True

    def exit(self):
        try:
            self.prefill.exit()
        finally:
            self.decode.exit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.exit()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
    ) -> list[dict]:
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        if len(prompts) != len(sampling_params):
            raise ValueError("Every prompt needs sampling parameters")

        outputs = []
        for prompt, params in zip(prompts, sampling_params):
            handoff = self.prefill.prefill_for_handoff(prompt, params)
            seq = self.decode.resume_from_handoff(handoff)
            while not seq.is_finished:
                self.decode.step()
            token_ids = seq.completion_token_ids
            outputs.append({"text": self.decode.tokenizer.decode(token_ids),
                            "token_ids": token_ids})
        return outputs
