
from __future__ import annotations

from typing import Any

class Platform:
    backend_name: str = ""
    device_name: str = ""
    dispatch_key: str = ""

    @property
    def device_module(self) -> Any:
        raise NotImplementedError

    def is_available(self) -> bool:
        raise NotImplementedError

    def device_count(self) -> int:
        raise NotImplementedError

    def device(self) -> Any:
        raise NotImplementedError

    def current_device(self) -> int:
        raise NotImplementedError

    def set_device(self, rank: int) -> None:
        raise NotImplementedError

    def set_default_device(self) -> None:
        raise NotImplementedError

    def mem_get_info(self) -> tuple[int, int]:
        raise NotImplementedError

    def empty_cache(self) -> None:
        raise NotImplementedError

    def reset_peak_memory_stats(self) -> None:
        raise NotImplementedError

    def memory_stats(self) -> dict[str, Any]:
        raise NotImplementedError

    def synchronize(self) -> None:
        raise NotImplementedError

    def init_process_group(self, world_size: int, rank: int) -> None:
        raise NotImplementedError

    def destroy_process_group(self) -> None:
        raise NotImplementedError

    def is_capture_supported(self) -> bool:
        return True

    def graph_class(self) -> Any:
        raise NotImplementedError

    def graph_capture_context(self, graph: Any, pool: Any) -> Any:
        raise NotImplementedError
