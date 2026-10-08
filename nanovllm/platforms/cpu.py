from __future__ import annotations

from typing import Any

from nanovllm.platforms.interface import Platform

class CPUPlatform(Platform):
    backend_name = "cpu"
    device_name = "cpu"

    @property
    def device_module(self) -> Any:
        return None

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 0

    def device(self) -> Any:
        import torch

        return torch.device("cpu")

    def current_device(self) -> int:
        return 0

    def set_device(self, rank: int) -> None:
        return None

    def set_default_device(self) -> None:
        return None

    def mem_get_info(self) -> tuple[int, int]:
        return (0, 0)

    def empty_cache(self) -> None:
        return None

    def reset_peak_memory_stats(self) -> None:
        return None

    def memory_stats(self) -> dict[str, Any]:
        return {}

    def synchronize(self) -> None:
        return None

    def init_process_group(self, world_size: int, rank: int) -> None:
        raise NotImplementedError

    def destroy_process_group(self) -> None:
        raise NotImplementedError

    def is_capture_supported(self) -> bool:
        return False

    def graph_class(self) -> Any:
        raise NotImplementedError

    def graph_capture_context(self, graph: Any, pool: Any) -> Any:
        raise NotImplementedError
