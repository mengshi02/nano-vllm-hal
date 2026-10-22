"""Compare local hash/radix caches and prefix-aware scheduling on a trace.

Each turn arrives after the preceding turn finishes. max_tokens=1 makes
completion time the first-token latency. Results exclude remote PD transport.
"""

import argparse
import json
from math import ceil
from statistics import mean
from time import perf_counter

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams


def conversation_trace(tokenizer, block_size: int) -> list[list[list[int]]]:
    shared = tokenizer.encode("You are a careful assistant. Explain each answer with examples. ")
    shared = (shared * (block_size * 2 // len(shared) + 1))[:block_size * 2]
    marker = tokenizer.encode(" More detail:", add_special_tokens=False)[0]
    histories = [list(shared) for _ in range(3)]
    trace = []
    for turn in range(4):
        batch = []
        for conversation in range(len(histories)):
            segment = tokenizer.encode(
                f"\nUser {conversation}: explain topic {turn} in more detail.\n"
                f"Assistant: this is turn {turn} for conversation {conversation}.",
                add_special_tokens=False)
            # Advance by a whole KV page on each turn, so a bounded cache
            # must choose which old conversation branches to retain.
            histories[conversation].extend(
                (segment * (block_size // len(segment) + 1))[:block_size])
            batch.append(histories[conversation] + [marker])
        trace.append(batch)
    return trace


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, ceil(len(ordered) * fraction) - 1)]


def run(model, policy, options, cache_blocks=None):
    index = "hash" if policy == "hash+aware" else policy
    engine = LLMEngine(model, prefix_cache=index, **options)
    try:
        engine.scheduler.cache_aware_scheduling = policy != "hash"
        manager = engine.scheduler.block_manager
        if cache_blocks is not None:
            if not 1 <= cache_blocks <= engine.config.num_kvcache_blocks:
                raise ValueError(f"cache-blocks must be between 1 and {engine.config.num_kvcache_blocks}")
            # Restrict the logical cache; workers retain the same physical KV
            # allocation in both runs, avoiding a GPU memory confounder.
            from nanovllm.engine.block_manager import BlockManager
            manager = BlockManager(cache_blocks, engine.config.kvcache_block_size, index)
            engine.scheduler.block_manager = manager
        trace = conversation_trace(engine.tokenizer, engine.config.kvcache_block_size)
        params = SamplingParams(temperature=1e-5, max_tokens=1)
        reused_blocks = {}
        eligible_blocks = {}
        original_allocate = manager.allocate

        def measure_allocate(seq, cached):
            if seq.seq_id not in reused_blocks:
                reused_blocks[seq.seq_id] = cached
                eligible_blocks[seq.seq_id] = min(
                    len(seq.prompt_token_ids) // manager.block_size, seq.num_blocks - 1)
            return original_allocate(seq, cached)

        manager.allocate = measure_allocate
        latency = []
        per_turn = []
        computed_tokens = 0
        for turn, prompts in enumerate(trace):
            started = {}
            for prompt in prompts:
                engine.add_request(prompt, params)
                started[engine.scheduler.waiting[-1].seq_id] = perf_counter()
            finished = {}
            while not engine.is_finished():
                outputs, num_prefill, _ = engine.step()
                computed_tokens += num_prefill
                now = perf_counter()
                for seq_id, _ in outputs:
                    finished[seq_id] = now - started[seq_id]
            turn_latency = [finished[seq_id] for seq_id in started]
            latency.extend(turn_latency)
            per_turn.append({"turn": turn + 1,
                             "hit_rate": round(sum(reused_blocks[seq_id] for seq_id in started) /
                                               sum(eligible_blocks[seq_id] for seq_id in started), 4),
                             "ttft_p50_seconds": round(percentile(turn_latency, .50), 4),
                             "ttft_p95_seconds": round(percentile(turn_latency, .95), 4)})
        hit_blocks = sum(reused_blocks.values())
        candidate_blocks = sum(eligible_blocks.values())
        return {
            "cache": policy, "requests": sum(map(len, trace)),
            "logical_cache_blocks": len(manager.blocks),
            "prompt_tokens": sum(len(prompt) for batch in trace for prompt in batch),
            "reused_blocks": hit_blocks, "eligible_blocks": candidate_blocks,
            "hit_rate": round(hit_blocks / candidate_blocks, 4),
            "prefill_computed_tokens": computed_tokens,
            "ttft_seconds": [round(value, 4) for value in latency],
            "ttft_p50_seconds": round(percentile(latency, .50), 4),
            "ttft_p95_seconds": round(percentile(latency, .95), 4),
            "ttft_mean_seconds": round(mean(latency), 4),
            "per_turn": per_turn,
        }
    finally:
        engine.exit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Local Qwen3 model directory")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.2)
    parser.add_argument("--token-budget", type=int, default=2048)
    parser.add_argument("--cache-blocks", type=int, default=None,
                        help="Identical logical KV block budget; try 12 to exercise eviction")
    args = parser.parse_args()
    options = dict(device_offset=args.device, enforce_eager=True,
                   gpu_memory_utilization=args.gpu_memory_utilization,
                   max_num_batched_tokens=args.token_budget)
    print(json.dumps([run(args.model, policy, options, args.cache_blocks)
                      for policy in ("hash", "hash+aware", "radix")],
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
