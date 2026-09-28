
from __future__ import annotations

from typing import Any

from nanovllm.platforms.cuda import CUDAPlatform
from nanovllm.platforms.interface import Platform

class ROCmPlatform(CUDAPlatform):
    backend_name = "rocm"
    device_name = "cuda"
    dispatch_key = "ROCM"

    def is_available(self) -> bool:
        import torch

        return torch.version.hip is not None

    def init_process_group(self, world_size: int, rank: int) -> None:
        import torch.distributed as dist

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=world_size, rank=rank)

class AscendPlatform(Platform):
    backend_name = "ascend"
    device_name = "npu"
    dispatch_key = "NPU"

    @property
    def device_module(self) -> Any:
        import torch_npu

        return torch_npu

    def is_available(self) -> bool:
        try:
            import torch_npu

            return torch_npu.npu.is_available()
        except ImportError:
            return False

    def device_count(self) -> int:
        return self.device_module.npu.device_count()

    def current_device(self) -> int:
        return self.device_module.npu.current_device()

    def set_device(self, rank: int) -> None:
        import torch_npu

        torch_npu.npu.set_device(rank)

    def set_default_device(self) -> None:
        import torch_npu

        torch_npu.set_device("npu")

    def mem_get_info(self) -> tuple[int, int]:
        return self.device_module.npu.mem_get_info()

    def empty_cache(self) -> None:
        self.device_module.npu.empty_cache()

    def reset_peak_memory_stats(self) -> None:
        self.device_module.npu.reset_peak_memory_stats()

    def memory_stats(self) -> dict[str, Any]:
        return self.device_module.npu.memory_stats()

    def synchronize(self) -> None:
        self.device_module.npu.synchronize()

    def init_process_group(self, world_size: int, rank: int) -> None:
        import torch.distributed as dist

        dist.init_process_group("hccl", "tcp://localhost:2333", world_size=world_size, rank=rank)

    def destroy_process_group(self) -> None:
        import torch.distributed as dist

        dist.destroy_process_group()

    def is_capture_supported(self) -> bool:
        return False

    def graph_class(self) -> Any:
        raise NotImplementedError

    def graph_capture_context(self, graph: Any, pool: Any) -> Any:
        raise NotImplementedError

class MUSAPlatform(CUDAPlatform):
    backend_name = "musa"
    device_name = "musa"
    dispatch_key = "MUSA"

    @property
    def device_module(self) -> Any:
        import torch_musa

        return torch_musa

    def is_available(self) -> bool:
        try:
            import torch_musa

            return torch_musa.is_available()
        except ImportError:
            return False

    def init_process_group(self, world_size: int, rank: int) -> None:
        import torch.distributed as dist

        dist.init_process_group("mccl", "tcp://localhost:2333", world_size=world_size, rank=rank)
