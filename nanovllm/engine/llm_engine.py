import atexit
import socket
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import start_model_runner
from nanovllm.engine.worker_client import RunnerClient
from nanovllm.platforms import current_platform


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        world_size = config.tensor_parallel_size * config.pipeline_parallel_size
        if config.device_offset + world_size > current_platform.device_count():
            raise ValueError(f"TP × PP requires devices {config.device_offset}..{config.device_offset + world_size - 1}, "
                             f"found {current_platform.device_count()}")
        if not config.init_method:
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                config.init_method = f"tcp://127.0.0.1:{listener.getsockname()[1]}"
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.connections = []
        self._closed = False
        ctx = mp.get_context("spawn")
        try:
            for i in range(world_size):
                parent, child = ctx.Pipe()
                process = ctx.Process(target=start_model_runner, args=(config, i, child))
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    child.close()
                    raise
                child.close()
                self.ps.append(process)
                self.connections.append(parent)
            self.model_runner = RunnerClient(self.ps, self.connections, config.worker_timeout)
            config.num_kvcache_blocks = self.model_runner.receive("ready")
            self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
            config.eos = self.tokenizer.eos_token_id
            self.scheduler = Scheduler(config)
        except BaseException:
            self._close_workers(force=True)
            raise
        atexit.register(self.exit)

    def exit(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.model_runner.call("exit")
        finally:
            self._close_workers()

    def _close_workers(self, force=False):
        if force:
            for p in self.ps:
                if p.is_alive():
                    p.terminate()
        for p in self.ps:
            p.join(timeout=1 if force else 5)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
                    p.join()
        for connection in self.connections:
            connection.close()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if not prompt:
            raise ValueError("prompt must contain at least one token")
        if sampling_params.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if len(prompt) + sampling_params.max_tokens - 1 > self.config.max_model_len:
            raise ValueError("prompt and max_tokens exceed max_model_len")
        num_blocks = (len(prompt) + sampling_params.max_tokens - 1 +
                      self.config.kvcache_block_size - 1) // self.config.kvcache_block_size
        if num_blocks > self.config.num_kvcache_blocks:
            raise ValueError("request exceeds available KV cache blocks")
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs = self.scheduler.schedule()
        num_prefill = sum(seq.num_scheduled_tokens for seq in seqs if seq.is_prefill)
        num_decode = sum(seq.num_scheduled_tokens for seq in seqs if not seq.is_prefill)
        try:
            token_ids = self.model_runner.call("run", seqs)
        except BaseException:
            self._closed = True
            self._close_workers(force=True)
            raise
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_prefill, num_decode

    def is_finished(self):
        return self.scheduler.is_finished()

    def _prefill_for_handoff(self, prompt: str | list[int], sampling_params: SamplingParams):
        """Compute a prompt, leaving its KV blocks owned by the returned sequence."""
        if not self.scheduler.is_finished():
            raise RuntimeError("Prefill engine must be idle before a handoff")
        self.add_request(prompt, sampling_params)
        seq = self.scheduler.waiting[0]
        complete = False
        try:
            while True:
                seqs = self.scheduler.schedule()
                assert seqs == [seq] and seq.is_prefill
                token_ids = self.model_runner.call("run", seqs)
                self.scheduler.block_manager.hash_blocks(seq)
                seq.num_cached_tokens += seq.num_scheduled_tokens
                seq.num_scheduled_tokens = 0
                if seq.num_cached_tokens == seq.num_prompt_tokens:
                    complete = True
                    return seq, token_ids[0]
        except BaseException:
            self._closed = True
            self._close_workers(force=True)
            raise
        finally:
            if seq in self.scheduler.running:
                self.scheduler.running.remove(seq)
            if seq in self.scheduler.waiting:
                self.scheduler.waiting.remove(seq)
            if not complete:
                self.scheduler.block_manager.deallocate(seq)

    def prefill_for_handoff(self, prompt: str | list[int], sampling_params: SamplingParams):
        """Copy KV to CPU for the local example, then release its GPU blocks."""
        from nanovllm.engine.pd import KVHandoff

        seq, first_token = self._prefill_for_handoff(prompt, sampling_params)
        try:
            if (getattr(self.config, "tensor_parallel_size", 1) *
                    getattr(self.config, "pipeline_parallel_size", 1) == 1):
                kv = self.model_runner.call("export_kv", seq.block_table, seq.num_prompt_tokens)
            else:
                kv = self.model_runner.call_all("export_kv", seq.block_table, seq.num_prompt_tokens)
            return KVHandoff(seq.prompt_token_ids, first_token, sampling_params,
                             kv, self.config.model, self.config.kvcache_block_size)
        finally:
            self.scheduler.block_manager.deallocate(seq)

    def prefill_for_remote(self, prompt: str | list[int], sampling_params: SamplingParams):
        """Keep P's GPU blocks until D has received and installed their KV."""
        from nanovllm.engine.pd import KVLease

        seq, first_token = self._prefill_for_handoff(prompt, sampling_params)
        return KVLease(self, seq, first_token, sampling_params)

    def resume_from_handoff(self, handoff):
        """Allocate local blocks, import prompt KV, then admit decode."""
        if handoff.model != self.config.model or handoff.block_size != self.config.kvcache_block_size:
            raise ValueError("Prefill and decode model or KV block size differ")
        prompt = handoff.prompt_token_ids
        max_cached_tokens = len(prompt) + handoff.sampling_params.max_tokens - 1
        if (not prompt or handoff.sampling_params.max_tokens < 1 or
                max_cached_tokens > self.config.max_model_len):
            raise ValueError("Handoff exceeds the decode model length")
        required_blocks = (max_cached_tokens + self.config.kvcache_block_size - 1) // self.config.kvcache_block_size
        if required_blocks > self.config.num_kvcache_blocks:
            raise ValueError("Handoff exceeds the decode engine's KV cache capacity")
        seq = Sequence(prompt, handoff.sampling_params)
        manager = self.scheduler.block_manager
        if len(manager.free_block_ids) < seq.num_blocks:
            raise RuntimeError("Decode engine has insufficient KV blocks for the prompt")
        manager.allocate(seq, 0)
        try:
            if isinstance(handoff.kv_blocks, list):
                if len(handoff.kv_blocks) != (getattr(self.config, "tensor_parallel_size", 1) *
                                             getattr(self.config, "pipeline_parallel_size", 1)):
                    raise ValueError("PD KV shard count differs from TP × PP")
                self.model_runner.call_each("import_kv", [
                    (seq.block_table, len(prompt), blocks) for blocks in handoff.kv_blocks])
            elif (getattr(self.config, "tensor_parallel_size", 1) *
                  getattr(self.config, "pipeline_parallel_size", 1) == 1):
                self.model_runner.call("import_kv", seq.block_table, len(prompt), handoff.kv_blocks)
            else:
                raise ValueError("PD requires one KV shard per model worker")
            seq.num_scheduled_tokens = len(prompt)
            manager.hash_blocks(seq)
            seq.num_cached_tokens = len(prompt)
            seq.num_scheduled_tokens = 0
            seq.append_token(handoff.first_token_id)
            seq.is_prefill = False
            seq.status = SequenceStatus.RUNNING
            if ((not seq.ignore_eos and handoff.first_token_id == self.config.eos)
                    or seq.num_completion_tokens == seq.max_tokens):
                seq.status = SequenceStatus.FINISHED
                manager.deallocate(seq)
            else:
                self.scheduler.running.append(seq)
            return seq
        except BaseException:
            manager.deallocate(seq)
            raise

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_prefill, num_decode = self.step()
            elapsed = perf_counter() - t
            if num_prefill:
                prefill_throughput = num_prefill / elapsed
            if num_decode:
                decode_throughput = num_decode / elapsed
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
