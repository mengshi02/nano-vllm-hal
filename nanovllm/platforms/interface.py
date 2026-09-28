"""Platform abstraction: one class per hardware backend.

The idea mirrors vLLM's ``Platform``/``Hardware`` split: the engine only ever
talks to ``current_platform``, and every backend-specific difference (device
torch module, memory info, communication backend, graph capture, dispatch
context, and custom op families) is isolated behind this interface.  Adding a
new vendor means adding one module under ``nanovllm/platforms`` and (optionally)
one family entry under ``nanovllm/ops`` -- the engine and the model code never
change.

``backend_name`` is the stable key used both to select a platform and to select
operator families in ``nanovllm.ops``.  Keep it lowercase and vendor-stable:
``"nvidia"``, ``"rocm"``, ``"ascend"``, ``"musa"``.
"""

from __future__ import annotations

from typing import Any


class Platform:
    """One hardware backend = one subclass of this class.

    Subclasses override the class attributes (``backend_name`` / ``device_name`` /
    ``dispatch_key``) and the methods below.  Instances are stateless singletons.
    """

    backend_name: str = ""
    device_name: str = ""
    dispatch_key: str = ""

    # --- device torch module / context -------------------------------------
    @property
    def device_module(self) -> Any:
        """The vendor torch device module, e.g. ``torch.cuda`` / ``torch_npu``."""
        raise NotImplementedError

    def is_available(self) -> bool:
        raise NotImplementedError

    def device_count(self) -> int:
        raise NotImplementedError

    def current_device(self) -> int:
        raise NotImplementedError

    def set_device(self, rank: int) -> None:
        raise NotImplementedError

    def set_default_device(self) -> None:
        raise NotImplementedError

    # --- memory ------------------------------------------------------------
    def mem_get_info(self) -> tuple[int, int]:
        """Return ``(free_bytes, total_bytes)``."""
        raise NotImplementedError

    def empty_cache(self) -> None:
        raise NotImplementedError

    def reset_peak_memory_stats(self) -> None:
        raise NotImplementedError

    def memory_stats(self) -> dict[str, Any]:
        raise NotImplementedError

    def synchronize(self) -> None:
        raise NotImplementedError

    # --- distributed communication ----------------------------------------
    def init_process_group(self, world_size: int, rank: int) -> None:
        raise NotImplementedError

    def destroy_process_group(self) -> None:
        raise NotImplementedError

    # --- graph capture -----------------------------------------------------
    def is_capture_supported(self) -> bool:
        return True

    def graph_class(self) -> Any:
        raise NotImplementedError

    def graph_capture_context(self, graph: Any, pool: Any) -> Any:
        raise NotImplementedError