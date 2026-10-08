from __future__ import annotations

from typing import Any

from nanovllm.platforms.cuda import CUDAPlatform
from nanovllm.platforms.interface import Platform

class ROCmPlatform(CUDAPlatform):
    backend_name = "rocm"
    device_name = "cuda"

    def is_available(self) -> bool:
        import torch

        return torch.version.hip is not None

    def init_process_group(self, world_size: int, rank: int) -> None:
        import torch.distributed as dist

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=world_size, rank=rank)

_HYGON_ARCHS = ("gfx926", "gfx928", "gfx936", "gfx938")

class HygonPlatform(Platform):
    backend_name = "hygon"
    device_name = "cuda"

    @property
    def device_module(self) -> Any:
        import torch

        return torch.cuda

    @staticmethod
    def _is_hygon_dcu() -> bool:
        import os

        import torch

        try:
            props = torch.cuda.get_device_properties(0)
        except Exception:
            return False
        name = (props.name or "").upper()
        if "AMD" in name or "RADEON" in name or "INSTINCT" in name:
            return False
        if os.path.exists("/dev/mkfd") or os.path.exists("/opt/hyhal"):
            return True
        arch = str(getattr(props, "gcnArchName", "")).split(":")[0]
        return arch in _HYGON_ARCHS

    def is_available(self) -> bool:
        import torch

        if torch.version.hip is None:
            return False
        if not torch.cuda.is_available():
            return False
        return self._is_hygon_dcu()

    def device_count(self) -> int:
        import torch

        return torch.cuda.device_count()

    def device(self) -> Any:
        import torch

        return torch.device("cuda")

    def current_device(self) -> int:
        import torch

        return torch.cuda.current_device()

    def set_device(self, rank: int) -> None:
        import torch

        torch.cuda.set_device(rank)

    def set_default_device(self) -> None:
        import torch

        torch.set_default_device("cuda")

    def mem_get_info(self) -> tuple[int, int]:
        import torch

        return torch.cuda.mem_get_info()

    def empty_cache(self) -> None:
        import torch

        torch.cuda.empty_cache()

    def reset_peak_memory_stats(self) -> None:
        import torch

        torch.cuda.reset_peak_memory_stats()

    def memory_stats(self) -> dict[str, Any]:
        import torch

        return torch.cuda.memory_stats()

    def synchronize(self) -> None:
        import torch

        torch.cuda.synchronize()

    def init_process_group(self, world_size: int, rank: int) -> None:
        import torch.distributed as dist

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=world_size, rank=rank)

    def destroy_process_group(self) -> None:
        import torch.distributed as dist

        dist.destroy_process_group()

    def is_capture_supported(self) -> bool:
        return True

    def graph_class(self) -> Any:
        import torch

        return torch.cuda.CUDAGraph

    def graph_capture_context(self, graph: Any, pool: Any) -> Any:
        import torch

        return torch.cuda.graph(graph, pool)

class AscendPlatform(Platform):
    backend_name = "ascend"
    device_name = "npu"

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

    def device(self) -> Any:
        import torch

        return torch.device("npu")

    def current_device(self) -> int:
        return self.device_module.npu.current_device()

    def set_device(self, rank: int) -> None:
        import torch_npu

        torch_npu.npu.set_device(rank)

    def set_default_device(self) -> None:
        import torch

        torch.set_default_device("npu")

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

    def is_capture_supported(self) -> bool:
        return False

    def set_device(self, rank: int) -> None:
        import torch_musa

        torch_musa.set_device(rank)

    def set_default_device(self) -> None:
        import torch

        torch.set_default_device("musa")

    def device(self) -> Any:
        import torch

        return torch.device("musa")

    def mem_get_info(self) -> tuple[int, int]:
        return self.device_module.mem_get_info()

    def empty_cache(self) -> None:
        self.device_module.empty_cache()

    def reset_peak_memory_stats(self) -> None:
        self.device_module.reset_peak_memory_stats()

    def memory_stats(self) -> dict[str, Any]:
        return self.device_module.memory_stats()

    def synchronize(self) -> None:
        self.device_module.synchronize()
