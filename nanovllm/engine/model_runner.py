import traceback
import torch
import torch.distributed as dist
from multiprocessing.connection import Connection

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.platforms import current_platform
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.utils.parallel import init_tensor_parallel_group, layer_range


class ModelRunner:

    def __init__(self, config: Config, rank: int):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.tp_size = config.tensor_parallel_size
        self.pp_size = config.pipeline_parallel_size
        self.world_size = self.tp_size * self.pp_size
        self.rank = rank
        self.tp_rank = rank % self.tp_size
        self.pp_rank = rank // self.tp_size
        self.local_layers = layer_range(hf_config.num_hidden_layers, self.pp_size, self.pp_rank)
        self.platform = current_platform

        self.platform.set_device(config.device_offset + rank)
        self.platform.init_process_group(self.world_size, rank, config.init_method)
        init_tensor_parallel_group(self.tp_size, self.pp_size, rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(getattr(hf_config, "dtype", None) or getattr(hf_config, "torch_dtype"))
        self.platform.set_default_device()
        self.model = Qwen3ForCausalLM(hf_config, self.pp_rank, self.pp_size,
                                      tp_fp32_matmul=config.tp_fp32_matmul)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if self.pp_size == 1 and not self.enforce_eager and self.platform.is_capture_supported():
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

    def exit(self):
        if self.world_size > 1:
            dist.barrier()
        if self.pp_size == 1 and not self.enforce_eager and self.platform.is_capture_supported():
            del self.graphs, self.graph_pool
        self.platform.synchronize()
        self.platform.destroy_process_group()

    def warmup_model(self):
        self.platform.empty_cache()
        self.platform.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs)
        self.platform.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        model_dtype = getattr(hf_config, "dtype", None) or getattr(hf_config, "torch_dtype")
        free, total = self.platform.mem_get_info()
        used = total - free
        peak = self.platform.memory_stats()["allocated_bytes.all.peak"]
        current = self.platform.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.tp_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * len(self.local_layers) * self.block_size * num_kv_heads * head_dim * model_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        if self.world_size > 1:
            num_blocks = torch.tensor([config.num_kvcache_blocks], dtype=torch.int64, device=self.platform.device())
            dist.all_reduce(num_blocks, op=dist.ReduceOp.MIN)
            config.num_kvcache_blocks = int(num_blocks.item())
        if config.num_kvcache_blocks <= 0:
            raise ValueError("Insufficient memory for KV cache on at least one TP/PP rank")
        self.kv_cache = torch.empty(2, len(self.local_layers), config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
        assert layer_id == len(self.local_layers)

    def export_kv(self, block_ids: list[int], num_tokens: int) -> torch.Tensor:
        """Copy the prompt's physical blocks to CPU in logical block order."""
        if num_tokens <= 0 or len(block_ids) != (num_tokens + self.block_size - 1) // self.block_size:
            raise ValueError("Invalid prompt KV block table")
        blocks = self.kv_cache[:, :, block_ids].cpu()
        tail = num_tokens % self.block_size
        if tail:
            blocks[:, :, -1, tail:].zero_()
        return blocks

    def import_kv(self, block_ids: list[int], num_tokens: int, blocks: torch.Tensor) -> None:
        """Install CPU KV into independently allocated local physical blocks."""
        expected = (2, len(self.local_layers), len(block_ids), *self.kv_cache.shape[3:])
        if (num_tokens <= 0 or len(block_ids) != (num_tokens + self.block_size - 1) // self.block_size
                or blocks.device.type != "cpu" or blocks.shape != expected
                or blocks.dtype != self.kv_cache.dtype):
            raise ValueError("Incompatible prompt KV cache")
        self.kv_cache[:, :, block_ids] = blocks.to(self.platform.device())

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).to(self.platform.device(), non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            if self.pp_rank == 0:
                input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = (torch.tensor(input_ids, dtype=torch.int64, pin_memory=True)
                     .to(self.platform.device(), non_blocking=True) if self.pp_rank == 0 else None)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).to(self.platform.device(), non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).to(self.platform.device(), non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).to(self.platform.device(), non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).to(self.platform.device(), non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            if self.pp_rank == 0:
                input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = (torch.tensor(input_ids, dtype=torch.int64, pin_memory=True)
                     .to(self.platform.device(), non_blocking=True) if self.pp_rank == 0 else None)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).to(self.platform.device(), non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).to(self.platform.device(), non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).to(self.platform.device(), non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).to(self.platform.device(), non_blocking=True)
        return temperatures

    def prepare_mixed(self, seqs: list[Sequence]):
        decode_seqs = [seq for seq in seqs if not seq.is_prefill]
        prefill_seqs = [seq for seq in seqs if seq.is_prefill]
        assert seqs == decode_seqs + prefill_seqs
        decode_ids, decode_positions = self.prepare_decode(decode_seqs)
        decode_context = get_context()
        prefill_ids, prefill_positions = self.prepare_prefill(prefill_seqs)
        prefill_context = get_context()
        num_decode = len(decode_seqs)
        last_prefill = prefill_context.cu_seqlens_q[1:] - 1 + num_decode
        last_indices = torch.cat((torch.arange(num_decode, device=self.platform.device()), last_prefill))
        set_context(False,
                    slot_mapping=torch.cat((decode_context.slot_mapping, prefill_context.slot_mapping)),
                    decode_context=decode_context, prefill_context=prefill_context,
                    num_decode_tokens=num_decode, last_token_indices=last_indices)
        ids = (torch.cat((decode_ids, prefill_ids)) if self.pp_rank == 0 else None)
        return ids, torch.cat((decode_positions, prefill_positions))

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        assert self.pp_size == 1
        if (is_prefill or get_context().prefill_context is not None or self.enforce_eager or
                input_ids.size(0) > 512 or not self.platform.is_capture_supported()):
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence]) -> list[int]:
        is_prefill = all(seq.is_prefill for seq in seqs)
        if is_prefill:
            input_ids, positions = self.prepare_prefill(seqs)
        elif all(not seq.is_prefill for seq in seqs):
            input_ids, positions = self.prepare_decode(seqs)
        else:
            input_ids, positions = self.prepare_mixed(seqs)
        if self.pp_size == 1:
            temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
            logits = self.run_model(input_ids, positions, is_prefill)
            token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        else:
            token_ids = self.run_pipeline(input_ids, positions, seqs)
        reset_context()
        return token_ids

    @torch.inference_mode()
    def run_pipeline(self, input_ids: torch.Tensor | None, positions: torch.Tensor,
                     seqs: list[Sequence]) -> list[int] | None:
        hidden_states = residual = None
        if self.pp_rank:
            shape = (positions.numel(), self.config.hf_config.hidden_size)
            dtype = next(self.model.parameters()).dtype
            hidden_states = torch.empty(shape, dtype=dtype, device=self.platform.device())
            residual = torch.empty_like(hidden_states)
            source = self.rank - self.tp_size
            dist.recv(hidden_states, src=source)
            dist.recv(residual, src=source)
        output = self.model(input_ids if self.pp_rank == 0 else None, positions,
                            hidden_states, residual)
        if self.pp_rank < self.pp_size - 1:
            hidden_states, residual = output
            destination = self.rank + self.tp_size
            dist.send(hidden_states.contiguous(), dst=destination)
            dist.send(residual.contiguous(), dst=destination)
        else:
            logits = self.model.compute_logits(output)
            if self.tp_rank == 0:
                temperatures = self.prepare_sample(seqs)
                sampled = self.sampler(logits, temperatures)
                if self.rank != 0:
                    dist.send(sampled.contiguous(), dst=0)
                else:
                    return sampled.tolist()
        if self.rank == 0:
            sampled = torch.empty(len(seqs), dtype=torch.int64, device=self.platform.device())
            dist.recv(sampled, src=(self.pp_size - 1) * self.tp_size)
            return sampled.tolist()
        return None

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = self.platform.graph_class()()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with self.platform.graph_capture_context(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            self.platform.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )


def start_model_runner(config: Config, rank: int, connection: Connection):
    try:
        Sequence.block_size = config.kvcache_block_size
        runner = ModelRunner(config, rank)
        connection.send(("ready", config.num_kvcache_blocks))
        while True:
            method_name, args = connection.recv()
            if method_name == "exit":
                runner.exit()
                connection.send(("ok", None))
                break
            result = getattr(runner, method_name)(*args)
            connection.send(("ok", result if rank == 0 or method_name == "export_kv" else None))
            # Drop both exported results and imported arguments after use;
            # retaining either keeps a potentially large /dev/shm KV file alive.
            result = None
            args = None
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        connection.close()
