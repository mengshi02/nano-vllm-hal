from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import SimpleQueue, Empty
from threading import Event
import hashlib
import json
import math
import select
import socket
import struct
from time import monotonic, sleep
from dataclasses import asdict
from itertools import cycle
from pathlib import Path
from uuid import UUID, uuid4

import torch

from nanovllm.engine.pd import KVHandoff
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.parallel import layer_range


_PREFIX = struct.Struct("!4sIQ")  # versioned magic, JSON length, raw KV length
_MAGIC = b"PD01"
_MAX_HEADER = 1 << 20
_MAX_PAYLOAD = 1 << 31
_CHUNK = 1 << 20
_DTYPES = {str(dtype): dtype for dtype in (torch.float16, torch.bfloat16, torch.float32)}


def model_fingerprint(model: str) -> str:
    """Identify actual weights and tokenizer, independently of their local paths."""
    root = Path(model)
    weights = sorted(root.glob("*.safetensors"))
    if not weights or not (root / "config.json").is_file():
        raise ValueError("Model fingerprint needs config.json and safetensors weights")
    # Hash files actually consumed by this engine. Distribution-specific files
    # (e.g. ModelScope configuration.json or generation_config.json) are not
    # loaded here and must not reject identical model weights across hubs.
    tokenizer_files = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                       "added_tokens.json", "vocab.json")
    if not (root / "tokenizer.json").is_file():
        tokenizer_files += ("merges.txt", "tokenizer.model")
    paths = [root / "config.json", *(root / name for name in tokenizer_files
                                       if (root / name).is_file()), *weights]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8") + b"\0")
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(_CHUNK), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _read_exact(connection: socket.socket, size: int) -> bytearray:
    data = bytearray(size)
    view = memoryview(data)
    position = 0
    while position < size:
        count = connection.recv_into(view[position:position + _CHUNK])
        if count == 0:
            raise EOFError(f"Peer closed connection after {position} of {size} bytes")
        position += count
    return data


def _send_frame(connection: socket.socket, header: dict, payload: memoryview | None = None):
    payload_size = len(payload) if payload is not None else 0
    if payload_size > _MAX_PAYLOAD:
        raise ValueError("KV payload exceeds the transport limit")
    if payload is not None:
        header = {**header, "sha256": hashlib.sha256(payload).hexdigest()}
    encoded = json.dumps(header, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(encoded) > _MAX_HEADER:
        raise ValueError("PD header exceeds the transport limit")
    connection.sendall(_PREFIX.pack(_MAGIC, len(encoded), payload_size))
    connection.sendall(encoded)
    if payload is not None:
        connection.sendall(payload)


def _recv_frame(connection: socket.socket, validate, max_payload: int):
    magic, header_size, payload_size = _PREFIX.unpack(_read_exact(connection, _PREFIX.size))
    if magic != _MAGIC or not 0 < header_size <= _MAX_HEADER or payload_size > min(max_payload, _MAX_PAYLOAD):
        raise ValueError("Invalid PD frame header or payload size")
    header = json.loads(_read_exact(connection, header_size))
    if not isinstance(header, dict):
        raise ValueError("Invalid PD message")
    validate(header, payload_size)  # Reject incompatible KV before allocating its buffer.
    payload = _read_exact(connection, payload_size)
    if payload_size and hashlib.sha256(payload).hexdigest() != header.get("sha256"):
        raise ValueError("PD KV checksum mismatch")
    return header, payload


def _expected_kv(engine, num_tokens: int):
    config = engine.config.hf_config
    block_size = engine.config.kvcache_block_size
    tp_size = getattr(engine.config, "tensor_parallel_size", 1)
    pp_size = getattr(engine.config, "pipeline_parallel_size", 1)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    shapes = [
        (2, len(layer_range(config.num_hidden_layers, pp_size, rank // tp_size)),
         math.ceil(num_tokens / block_size), block_size,
         config.num_key_value_heads // tp_size, head_dim)
        for rank in range(tp_size * pp_size)
    ]
    dtype = getattr(config, "dtype", None) or getattr(config, "torch_dtype")
    return shapes, dtype


@dataclass
class KVLocation:
    """Request metadata; physical block IDs refer to P, never to D."""

    request_id: str
    prompt_token_ids: list[int]
    first_token_id: int
    sampling_params: SamplingParams
    block_ids: list[int]
    shape: tuple[int, ...]
    dtype: torch.dtype
    model: str
    block_size: int
    rank_shapes: tuple[tuple[int, ...], ...] = ()
    tp_size: int = 1
    pp_size: int = 1


@dataclass
class PulledKV:
    """The Future retains this object; clear its tensor when the engine takes it."""

    location: KVLocation
    blocks: torch.Tensor | list[torch.Tensor] | None
    reused_blocks: int = 0

    def take(self) -> torch.Tensor | list[torch.Tensor]:
        if self.blocks is None:
            raise RuntimeError("KV has already been consumed")
        blocks, self.blocks = self.blocks, None
        return blocks


@dataclass
class PullPlan:
    location: KVLocation
    ready: Event
    reused_blocks: int = 0
    error: Exception | None = None


@dataclass
class PrefixProbe:
    tokens: list[int]
    ready: Event
    cached_blocks: int = 0
    error: Exception | None = None


def send_metadata(connection: socket.socket, lease, fingerprint: str, request_id: str):
    seq = lease.sequence
    shapes, dtype = _expected_kv(lease.engine, seq.num_prompt_tokens)
    _send_frame(connection, {
        "kind": "location", "request_id": request_id, "model_sha256": fingerprint,
        "prompt": seq.prompt_token_ids, "first_token": lease.first_token_id,
        "sampling": asdict(lease.sampling_params),
        "block_size": lease.engine.config.kvcache_block_size,
        "block_ids": list(seq.block_table), "shape": list(shapes[0]), "dtype": str(dtype),
        "rank_shapes": [list(shape) for shape in shapes],
        "tp_size": getattr(lease.engine.config, "tensor_parallel_size", 1),
        "pp_size": getattr(lease.engine.config, "pipeline_parallel_size", 1),
    })


def receive_metadata(connection: socket.socket, engine, fingerprint: str,
                     allow_probe: bool = False) -> KVLocation | dict:
    def validate(header, size):
        if allow_probe and header.get("kind") == "probe":
            tokens = header.get("prompt")
            if (size or header.get("model_sha256") != fingerprint or
                    header.get("block_size") != engine.config.kvcache_block_size or
                    not isinstance(tokens, list) or not tokens or
                    len(tokens) > engine.config.max_model_len or
                    not all(type(token) is int and 0 <= token < engine.config.hf_config.vocab_size
                            for token in tokens)):
                raise ValueError("Invalid PD prefix probe")
            return
        if size or header.get("kind") != "location":
            raise ValueError("Invalid PD location")
        _validate_location(header, engine, fingerprint)
        ids = header.get("block_ids")
        if (not isinstance(ids, list) or len(ids) != header["shape"][2] or
                not all(type(i) is int and i >= 0 for i in ids) or
                len(set(ids)) != len(ids)):
            raise ValueError("Invalid producer KV block IDs")
        try:
            uuid4_str = str(UUID(header.get("request_id", "")))
        except (TypeError, ValueError, AttributeError):
            raise ValueError("Invalid PD request ID") from None
        if uuid4_str != header["request_id"]:
            raise ValueError("Invalid PD request ID")
        if any(math.prod(shape) * _DTYPES[header["dtype"]].itemsize > _MAX_PAYLOAD
               for shape in header.get("rank_shapes", [header["shape"]])):
            raise ValueError("PD KV exceeds the transport limit")

    header, _ = _recv_frame(connection, validate, 0)
    if header["kind"] == "probe":
        return header
    return KVLocation(header["request_id"], header["prompt"], header["first_token"],
                      SamplingParams(**header["sampling"]), header["block_ids"],
                      tuple(header["shape"]), _DTYPES[header["dtype"]],
                      engine.config.model, engine.config.kvcache_block_size,
                      tuple(tuple(shape) for shape in header.get("rank_shapes", [header["shape"]])),
                      header.get("tp_size", 1), header.get("pp_size", 1))


def send_pull(connection: socket.socket, location: KVLocation, reused_blocks: int = 0):
    _send_frame(connection, {"kind": "pull", "request_id": location.request_id,
                             "block_ids": location.block_ids[reused_blocks:],
                             "reused_blocks": reused_blocks})


def receive_pull(connection: socket.socket, lease, request_id: str) -> int:
    def validate(header, size):
        reused = header.get("reused_blocks", 0)
        max_reused = lease.sequence.num_prompt_tokens // lease.engine.config.kvcache_block_size
        if header.get("kind") == "error":
            raise RuntimeError(f"Decode server: {header.get('message', 'unknown error')}")
        if (size or header.get("kind") != "pull" or header.get("request_id") != request_id
                or type(reused) is not int or not 0 <= reused <= max_reused
                or header.get("block_ids") != lease.sequence.block_table[reused:]):
            raise ValueError("Invalid PD KV pull request")

    header, _ = _recv_frame(connection, validate, 0)
    return header.get("reused_blocks", 0)


def send_kv_blocks(connection: socket.socket, request_id: str, blocks: torch.Tensor,
                   rank: int | None = None):
    if blocks.device.type != "cpu" or blocks.dtype not in _DTYPES.values():
        raise ValueError("KV transfer needs supported CPU tensors")
    raw = memoryview(blocks.contiguous().view(torch.uint8).flatten().numpy())
    header = {"kind": "kv", "request_id": request_id}
    if rank is not None:
        header["rank"] = rank
    _send_frame(connection, header, raw)


def receive_kv_blocks(connection: socket.socket, location: KVLocation, rank: int = 0,
                      reused_blocks: int = 0) -> torch.Tensor:
    shape = (*location.rank_shapes[rank][:2],
             location.rank_shapes[rank][2] - reused_blocks,
             *location.rank_shapes[rank][3:])
    expected_size = math.prod(shape) * location.dtype.itemsize

    def validate(header, size):
        if (header.get("kind") != "kv" or header.get("request_id") != location.request_id
                or size != expected_size or
                (len(location.rank_shapes) > 1 and header.get("rank") != rank) or
                (len(location.rank_shapes) == 1 and header.get("rank", 0) != 0)):
            raise ValueError("Invalid PD KV response")

    _, buffer = _recv_frame(connection, validate, expected_size)
    return torch.frombuffer(buffer, dtype=torch.uint8).view(location.dtype).reshape(shape)


def send_ack(connection: socket.socket, request_id: str):
    _send_frame(connection, {"kind": "read_done", "request_id": request_id})


def receive_ack(connection: socket.socket, request_id: str):
    def validate(header, size):
        if size:
            raise ValueError("Invalid PD read acknowledgment")
        if header.get("kind") == "error":
            raise RuntimeError(f"Decode server: {header.get('message', 'unknown error')}")
        if header.get("kind") != "read_done" or header.get("request_id") != request_id:
            raise ValueError("Invalid PD read acknowledgment")

    _recv_frame(connection, validate, 0)


def _validate_location(header, engine, fingerprint):
    if header.get("model_sha256") != fingerprint:
        raise ValueError("PD model weights or tokenizer differ")
    prompt = header.get("prompt")
    if (not isinstance(prompt, list) or not prompt or
            len(prompt) > engine.config.max_model_len or
            not all(type(token) is int and 0 <= token < engine.config.hf_config.vocab_size
                    for token in prompt)):
        raise ValueError("Invalid PD prompt")
    shapes, dtype = _expected_kv(engine, len(prompt))
    if math.ceil(len(prompt) / engine.config.kvcache_block_size) > engine.config.num_kvcache_blocks:
        raise ValueError("PD prompt exceeds decode KV capacity")
    if (header.get("block_size") != engine.config.kvcache_block_size or
            header.get("tp_size", 1) != getattr(engine.config, "tensor_parallel_size", 1) or
            header.get("pp_size", 1) != getattr(engine.config, "pipeline_parallel_size", 1) or
            header.get("shape") != list(shapes[0]) or
            header.get("rank_shapes", [header.get("shape")]) != [list(shape) for shape in shapes] or
            header.get("dtype") != str(dtype)):
        raise ValueError("Incompatible PD TP/PP layout, KV shape, dtype, or block size")
    if (type(header.get("first_token")) is not int or
            not 0 <= header["first_token"] < engine.config.hf_config.vocab_size):
        raise ValueError("Invalid PD first token")
    sampling = header.get("sampling")
    if not isinstance(sampling, dict) or set(sampling) != {"temperature", "max_tokens", "ignore_eos"}:
        raise ValueError("Invalid PD sampling parameters")
    if (type(sampling["max_tokens"]) is not int or sampling["max_tokens"] < 1 or
            not isinstance(sampling["ignore_eos"], bool) or
            type(sampling["temperature"]) not in (int, float) or
            not math.isfinite(sampling["temperature"]) or sampling["temperature"] <= 1e-10):
        raise ValueError("Invalid PD sampling parameters")
    if len(prompt) + sampling["max_tokens"] - 1 > engine.config.max_model_len:
        raise ValueError("PD request exceeds decode model length")
    if (math.ceil((len(prompt) + sampling["max_tokens"] - 1) /
                  engine.config.kvcache_block_size) > engine.config.num_kvcache_blocks):
        raise ValueError("PD request exceeds decode KV capacity")

def send_result(connection: socket.socket, token_ids: list[int]):
    _send_frame(connection, {"kind": "result", "token_ids": token_ids})


def receive_result(connection: socket.socket) -> list[int]:
    def validate(header, size):
        if size:
            raise ValueError("Unexpected PD response payload")

    header, _ = _recv_frame(connection, validate, 0)
    if header.get("kind") == "error":
        raise RuntimeError(f"Decode server: {header.get('message', 'unknown error')}")
    tokens = header.get("token_ids")
    if (header.get("kind") != "result" or not isinstance(tokens, list) or
            not all(type(t) is int and t >= 0 for t in tokens)):
        raise ValueError("Invalid PD response")
    return tokens


def _handoff_remote(engine, prompt, params, host, port, fingerprint, timeout):
    """The P engine has one owner; return the socket only after D installed KV."""
    with engine.prefill_for_remote(prompt, params) as lease:
        # A bounded lease also covers a D that accepts but never requests KV.
        deadline = monotonic() + timeout
        connection = socket.create_connection((host, port), timeout=timeout)
        try:
            connection.settimeout(timeout)
            request_id = str(uuid4())
            send_metadata(connection, lease, fingerprint, request_id)
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError("PD KV lease expired before D pulled its blocks")
            connection.settimeout(remaining)
            reused_blocks = receive_pull(connection, lease, request_id)
            if monotonic() >= deadline:
                raise TimeoutError("PD KV lease expired before D pulled its blocks")
            connection.settimeout(timeout)
            sent_bytes = 0
            if reused_blocks < len(lease.sequence.block_table):
                blocks = lease.read(reused_blocks)
                if isinstance(blocks, list):
                    if len(blocks) != (getattr(engine.config, "tensor_parallel_size", 1) *
                                       getattr(engine.config, "pipeline_parallel_size", 1)):
                        raise ValueError("Prefill did not export every KV rank")
                    for rank, shard in enumerate(blocks):
                        send_kv_blocks(connection, request_id, shard, rank)
                        sent_bytes += shard.numel() * shard.element_size()
                else:
                    send_kv_blocks(connection, request_id, blocks)
                    sent_bytes = blocks.numel() * blocks.element_size()
                del blocks
            receive_ack(connection, request_id)
            stats = getattr(engine, "pd_transfer_stats", None)
            if stats is None:
                stats = engine.pd_transfer_stats = dict(requests=0, reused_blocks=0, kv_bytes=0)
            stats["requests"] += 1
            stats["reused_blocks"] += reused_blocks
            stats["kv_bytes"] += sent_bytes
            first_token_id = lease.first_token_id
            # The P worker can prefill another request as soon as D owns KV.
            lease.release()
            return connection, first_token_id
        except BaseException:
            connection.close()
            raise


def _read_remote_output(engine, connection, params, first_token_id):
    token_ids = receive_result(connection)
    if not 1 <= len(token_ids) <= params.max_tokens or token_ids[0] != first_token_id:
        raise ValueError("Invalid PD response token count or first token")
    return {"text": engine.tokenizer.decode(token_ids), "token_ids": token_ids}


def _normalize_requests(prompts, sampling_params):
    if not isinstance(sampling_params, list):
        sampling_params = [sampling_params] * len(prompts)
    if len(prompts) != len(sampling_params):
        raise ValueError("Every prompt needs sampling parameters")
    return sampling_params


def _probe_prefix(host, port, prompt, fingerprint, block_size, timeout):
    with socket.create_connection((host, port), timeout=timeout) as connection:
        connection.settimeout(timeout)
        _send_frame(connection, {"kind": "probe", "model_sha256": fingerprint,
                                 "block_size": block_size, "prompt": prompt})
        def validate(header, size):
            if (size or header.get("kind") != "probe_result" or
                    type(header.get("cached_blocks")) is not int or
                    not 0 <= header["cached_blocks"] <= len(prompt) // block_size):
                raise ValueError("Invalid PD prefix probe result")
        header, _ = _recv_frame(connection, validate, 0)
        return header["cached_blocks"]


def generate_remote(engine, prompts, sampling_params, host: str, port: int,
                    fingerprint: str, timeout: float = 300.0) -> list[dict]:
    sampling_params = _normalize_requests(prompts, sampling_params)
    outputs = []
    for prompt, params in zip(prompts, sampling_params):
        connection, first_token_id = _handoff_remote(
            engine, prompt, params, host, port, fingerprint, timeout)
        with connection:
            outputs.append(_read_remote_output(engine, connection, params, first_token_id))
    return outputs


def generate_remote_routed(engine, prompts, sampling_params, endpoints: list[tuple[str, int]],
                           fingerprint: str, timeout: float = 300.0,
                           max_inflight: int = 2) -> list[dict]:
    """Route to a D with a resident prefix, falling back to round-robin.

    Only this thread touches the model worker. Each socket belongs to P until
    its result arrives; an ACK transfers KV ownership, not result ownership.
    """
    sampling_params = _normalize_requests(prompts, sampling_params)
    if (not endpoints or any(not isinstance(host, str) or not host or
                             type(port) is not int or not 1 <= port <= 65535
                             for host, port in endpoints)):
        raise ValueError("endpoints must contain valid (host, port) pairs")
    if type(max_inflight) is not int or max_inflight < 1 or timeout <= 0:
        raise ValueError("max_inflight and timeout must be positive")
    outputs = [None] * len(prompts)
    routes = cycle(endpoints)
    pending = {}  # socket -> (output index, params, first token, result deadline)

    def collect_one():
        remaining = min(deadline for _, _, _, deadline in pending.values()) - monotonic()
        if remaining <= 0:
            raise TimeoutError("Timed out waiting for a PD decode result")
        readable, _, _ = select.select(list(pending), [], [], remaining)
        if not readable:
            raise TimeoutError("Timed out waiting for a PD decode result")
        for connection in readable:
            index, params, first_token_id, deadline = pending.pop(connection)
            with connection:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out reading a PD decode result")
                connection.settimeout(remaining)
                outputs[index] = _read_remote_output(engine, connection, params,
                                                      first_token_id)

    try:
        for index, (prompt, params) in enumerate(zip(prompts, sampling_params)):
            while len(pending) >= max_inflight:
                collect_one()
            preferred = next(routes)
            candidates = [preferred, *(endpoint for endpoint in endpoints if endpoint != preferred)]
            if len(endpoints) > 1:
                tokens = engine.tokenizer.encode(prompt) if isinstance(prompt, str) else prompt
                # Rotate tie priority so a cold cache still distributes work.
                offset = endpoints.index(preferred)
                candidates = endpoints[offset:] + endpoints[:offset]
                scores = {}
                for candidate_host, candidate_port in candidates:
                    try:
                        scores[(candidate_host, candidate_port)] = _probe_prefix(
                            candidate_host, candidate_port, tokens, fingerprint,
                            engine.config.kvcache_block_size, min(timeout, 2.0))
                    except (OSError, ValueError, TimeoutError, EOFError):
                        scores[(candidate_host, candidate_port)] = -1
                # A reachable cold D beats an unreachable preferred D. Probe
                # results are hints: the selected D reserves again on handoff.
                candidates.sort(key=lambda endpoint: scores[endpoint], reverse=True)
            for attempt, (host, port) in enumerate(candidates):
                try:
                    connection, first_token_id = _handoff_remote(
                        engine, prompt, params, host, port, fingerprint, timeout)
                    break
                except (OSError, TimeoutError, EOFError):
                    if attempt == len(candidates) - 1:
                        raise
            pending[connection] = (index, params, first_token_id, monotonic() + timeout)
        while pending:
            collect_one()
        return outputs
    finally:
        for connection in pending:
            connection.close()


def _pull_into_cpu(connection, engine, fingerprint, planning: SimpleQueue | None = None):
    """Network-only D task: the model worker is exclusively owned by the main thread."""
    location = receive_metadata(connection, engine, fingerprint, allow_probe=planning is not None)
    if isinstance(location, dict):
        probe = PrefixProbe(location["prompt"], Event())
        planning.put((connection, probe))
        if not probe.ready.wait(connection.gettimeout()):
            raise TimeoutError("Timed out waiting for D prefix lookup")
        if probe.error is not None:
            raise probe.error
        _send_frame(connection, {"kind": "probe_result", "cached_blocks": probe.cached_blocks})
        return None
    reused = 0
    if planning is not None:
        plan = PullPlan(location, Event())
        planning.put((connection, plan))
        if not plan.ready.wait(connection.gettimeout()):
            raise TimeoutError("Timed out waiting for D to reserve KV pages")
        if plan.error is not None:
            raise plan.error
        reused = plan.reused_blocks
    send_pull(connection, location, reused)
    if reused == location.shape[2]:
        blocks = []
    elif len(location.rank_shapes) == 1:
        blocks = receive_kv_blocks(connection, location, reused_blocks=reused)
    else:
        blocks = [receive_kv_blocks(connection, location, rank, reused)
                  for rank in range(len(location.rank_shapes))]
    return PulledKV(location, blocks, reused)


def _send_error(connection, exc):
    try:
        _send_frame(connection, {"kind": "error", "message": str(exc)[:4096]})
    except (OSError, ValueError):
        pass


def serve_decode(engine, bind: str, port: int, fingerprint: str, timeout: float = 300.0):
    engine.scheduler.decode_only = True
    planning = SimpleQueue()
    reservations = {}  # connection -> (sequence, reused blocks), main thread only
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener, ThreadPoolExecutor(max_workers=2) as transfers:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((bind, port))
        listener.listen(2)
        listener.settimeout(0.05)
        print(f"PD decode listening on {bind}:{port}", flush=True)
        pending = {}  # future -> connection; network reads can overlap decode.
        active = {}  # sequence ID -> connection; only this thread touches the engine.
        try:
            while True:
                if len(pending) < 2:
                    # Never stall an in-flight decode step waiting for a new client.
                    listener.settimeout(0 if active else 0.05)
                    try:
                        connection, _ = listener.accept()
                    except (socket.timeout, BlockingIOError):
                        pass
                    else:
                        connection.settimeout(timeout)
                        pending[transfers.submit(_pull_into_cpu, connection, engine, fingerprint, planning)] = connection
                else:
                    sleep(0.01)

                while True:
                    try:
                        connection, plan = planning.get_nowait()
                    except Empty:
                        break
                    try:
                        if isinstance(plan, PrefixProbe):
                            plan.cached_blocks = len(engine.scheduler.block_manager.cached_blocks(plan.tokens))
                        else:
                            reserved = engine.reserve_handoff(plan.location.prompt_token_ids,
                                                              plan.location.sampling_params)
                            reservations[connection] = reserved
                            plan.reused_blocks = reserved[1]
                    except Exception as exc:
                        plan.error = exc
                    finally:
                        plan.ready.set()

                for future, connection in list(pending.items()):
                    if not future.done():
                        continue
                    del pending[future]
                    seq = None
                    reserved = reservations.pop(connection, None)
                    try:
                        received = future.result()
                        if received is None:  # Probe only; no KV ownership changed.
                            connection.close()
                            continue
                        location = received.location
                        handoff = KVHandoff(location.prompt_token_ids, location.first_token_id,
                                            location.sampling_params, received.take(),
                                            location.model, location.block_size)
                        try:
                            seq = engine.resume_from_handoff(handoff, reserved=reserved)
                            reserved = None
                        finally:
                            # Sending torch CPU tensors to a model worker can
                            # share their storage through /dev/shm. A Future's
                            # result and this loop must not pin it until decode ends.
                            del handoff
                        if not seq.is_finished:
                            active[seq.seq_id] = connection
                        send_ack(connection, location.request_id)
                        if seq.is_finished:
                            send_result(connection, seq.completion_token_ids)
                            connection.close()
                    except Exception as exc:
                        if reserved is not None and reserved[0].block_table:
                            engine.scheduler.block_manager.deallocate(reserved[0])
                        if seq is not None and active.get(seq.seq_id) is connection:
                            active.pop(seq.seq_id)
                            engine.scheduler.running.remove(seq)
                            engine.scheduler.block_manager.deallocate(seq)
                        _send_error(connection, exc)
                        connection.close()
                        if getattr(engine, "_closed", False):
                            raise

                if active:
                    completed, _, _ = engine.step()
                    for seq_id, token_ids in completed:
                        connection = active.pop(seq_id, None)
                        if connection is not None:
                            try:
                                send_result(connection, token_ids)
                            except OSError:
                                pass  # P may have timed out after D already imported KV.
                            finally:
                                connection.close()
        finally:
            for reserved in reservations.values():
                engine.scheduler.block_manager.deallocate(reserved[0])
            for connection in (*pending.values(), *active.values()):
                connection.close()
