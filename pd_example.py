"""Run educational PD handoff on two disjoint local device groups.

Example: python3 pd_example.py /path/to/Qwen3-0.6B --devices 0 1
"""

import argparse

from nanovllm import LLM, SamplingParams
from nanovllm.engine.pd import DisaggregatedLLM


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Local Qwen3 model directory")
    parser.add_argument("--devices", nargs=2, type=int, default=(0, 1), metavar=("P", "D"))
    parser.add_argument("--tp", type=int, default=1, help="Tensor-parallel workers per stage on P and D")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline stages on P and D")
    parser.add_argument("--prompt", default="Introduce yourself in one sentence.")
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--token-budget", type=int, default=128,
                        help="Small budgets exercise chunked prefill")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.1,
                        help="Fraction of each device reserved for model and KV cache")
    parser.add_argument("--compare", action="store_true",
                        help="Compare every generated token with ordinary inference using the same TP/PP")
    parser.add_argument("--cross-block", action="store_true",
                        help="Extend the prompt past one 256-token KV block")
    args = parser.parse_args()

    params = SamplingParams(temperature=1e-5, max_tokens=args.max_tokens)
    engine_options = dict(max_num_batched_tokens=args.token_budget, enforce_eager=True,
                          gpu_memory_utilization=args.gpu_memory_utilization,
                          tensor_parallel_size=args.tp, pipeline_parallel_size=args.pp)
    with DisaggregatedLLM(args.model, devices=tuple(args.devices),
                           **engine_options) as llm:
        prompt = llm.prefill.tokenizer.encode(args.prompt)
        if args.cross_block:
            if not prompt:
                raise ValueError("--cross-block needs a prompt with at least one token")
            prompt = (prompt * (257 // len(prompt) + 1))[:257]
        pd_output = llm.generate([prompt], params)[0]
        print("PD:", pd_output)

    if args.compare:
        baseline = LLM(args.model, device_offset=args.devices[0], **engine_options)
        try:
            reference = baseline.generate([prompt], params, use_tqdm=False)[0]["token_ids"]
        finally:
            baseline.exit()
        actual = pd_output["token_ids"]
        for position, (expected, received) in enumerate(zip(reference, actual)):
            if expected != received:
                raise SystemExit(f"Token mismatch at index {position}: baseline={expected}, PD={received}")
        if len(reference) != len(actual):
            raise SystemExit(f"Length mismatch: baseline={len(reference)}, PD={len(actual)}")
        print(f"Matched all {len(actual)} generated tokens")
