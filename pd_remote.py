"""Run the cross-host learning example (single D or routed P to several Ds).

Start D first: python3 pd_remote.py decode /path/to/Qwen3 --bind 0.0.0.0
Then on P:  python3 pd_remote.py prefill /path/to/Qwen3 --host D_HOST --cross-block --compare
For multiple Ds: python3 pd_remote.py routed /path/to/Qwen3 --decode D1:29581 --decode D2:29582 --requests 4
"""

import argparse

from nanovllm import LLM, SamplingParams
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.pd_network import (
    generate_remote, generate_remote_routed, model_fingerprint, serve_decode,
)


def decode_endpoint(value: str) -> tuple[str, int]:
    host, separator, port = value.rpartition(":")
    try:
        number = int(port)
    except ValueError:
        number = 0
    if not separator or not host or not 1 <= number <= 65535:
        raise argparse.ArgumentTypeError("decode endpoint must be HOST:PORT")
    return host, number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("prefill", "decode", "routed"))
    parser.add_argument("model", help="Local Qwen3 model directory on this host")
    parser.add_argument("--host", help="Decode host/IP for the prefill client")
    parser.add_argument("--decode", action="append", type=decode_endpoint, default=[],
                        metavar="HOST:PORT", help="Decode endpoint for routed P; repeat for each D")
    parser.add_argument("--requests", type=int, default=4, help="Requests from one routed P engine")
    parser.add_argument("--max-inflight", type=int, default=2,
                        help="Maximum requests still decoding across the D endpoints")
    parser.add_argument("--bind", default="127.0.0.1", help="Decode listen address")
    parser.add_argument("--port", type=int, default=29581)
    parser.add_argument("--device", type=int, default=0, help="Local device ID on this host")
    parser.add_argument("--tp", type=int, default=1, help="Tensor-parallel workers per stage on each P/D")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline stages on each P/D")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.2)
    parser.add_argument("--token-budget", type=int, default=128)
    parser.add_argument("--prompt", default="Introduce yourself in one sentence.")
    parser.add_argument("--max-tokens", type=int, default=12)
    parser.add_argument("--cross-block", action="store_true", help="Use a 257-token prompt")
    parser.add_argument("--compare", action="store_true", help="Compare output with local ordinary generation")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.timeout <= 0 or args.tp < 1 or args.pp < 1:
        parser.error("port must be 1..65535; timeout, tp and pp must be positive")
    if args.role == "prefill" and not args.host:
        parser.error("prefill requires --host pointing to the decode server")
    if args.role == "routed" and (not args.decode or args.requests < 1 or args.max_inflight < 1):
        parser.error("routed requires --decode and positive --requests/--max-inflight")

    options = dict(device_offset=args.device, gpu_memory_utilization=args.gpu_memory_utilization,
                   max_num_batched_tokens=args.token_budget, enforce_eager=True,
                   tensor_parallel_size=args.tp, pipeline_parallel_size=args.pp)
    fingerprint = model_fingerprint(args.model)
    engine = LLMEngine(args.model, **options)
    try:
        if args.role == "decode":
            serve_decode(engine, args.bind, args.port, fingerprint, args.timeout)
            return
        prompt = engine.tokenizer.encode(args.prompt)
        if args.cross_block:
            if not prompt:
                raise ValueError("--cross-block needs a nonempty prompt")
            prompt = (prompt * (257 // len(prompt) + 1))[:257]
        params = SamplingParams(temperature=1e-5, max_tokens=args.max_tokens)
        if args.role == "routed":
            prompts = [prompt] * args.requests
            outputs = generate_remote_routed(engine, prompts, params, args.decode,
                                             fingerprint, args.timeout, args.max_inflight)
        else:
            prompts = [prompt]
            outputs = generate_remote(engine, prompts, params, args.host, args.port,
                                      fingerprint, args.timeout)
        for index, output in enumerate(outputs):
            print(f"Remote PD [{index}]:" if args.role == "routed" else "Remote PD:", output)
    finally:
        engine.exit()

    if args.compare:
        baseline = LLM(args.model, **options)
        try:
            expected = [baseline.generate([item], params, use_tqdm=False)[0]["token_ids"]
                        for item in prompts]
        finally:
            baseline.exit()
        for request_index, (output, reference) in enumerate(zip(outputs, expected)):
            actual = output["token_ids"]
            if actual != reference:
                raise SystemExit(f"Request {request_index}: token mismatch: "
                                 f"baseline={reference}, PD={actual}")
            print(f"Matched all {len(actual)} generated tokens for request {request_index}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
