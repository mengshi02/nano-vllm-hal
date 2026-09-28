"""NVIDIA CUDA platform -- the reference backend.

This is a pure move of the ``torch.cuda`` / ``nccl`` calls that currently live
inlined in ``ModelRunner``.  Behaviour is intentionally identical so that the
HAL refactor can be validated with a CUDA smoke run before any other vendor is
wired in.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from nanovllm.platforms.interface import Platform


class CUDAPlatform(Platform):
    backend_name = "nvidia"
    device_name = "cuda"
    dispatch_key = "CUDA"

    @property
    def device_module(self):
        return torch.cuda

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def device_count(self) -> int:
        return torch.cuda.device_count()

    def current_device(self) -> int:
        return torch.cuda.current_device()

    def set_device(self, rank: int) -> None:
        torch.cuda.set_device(rank)

    def set_default_device(self) -> None:
        torch.set_default_device("cuda")

    def mem_get_info(self) -> tuple[int, int]:
        return torch.cuda.mem_get_info()

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def reset_peak_memory_stats(self) -> None:
        torch.cuda.reset_peak_memory_stats()

    def memory_stats(self) -> dict[str, Any]:
        return torch.cuda.memory_stats()

    def synchronize(self) -> None:
        torch.cuda.synchronize()

    def init_process_group(self, world_size: int, rank: int) -> None:
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=world_size, rank=rank)

    def destroy_process_group(self) -> None:
        dist.destroy_process_group()

    def graph_class(self) -> Any:
        return torch.cuda.CUDAGraph

    def graph_capture_context(self, graph: Any, pool: Any):
        return torch.cuda.graph(graph, pool)