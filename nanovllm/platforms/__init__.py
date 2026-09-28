"""Platform resolution: pick the platform matching the hardware in this process.

The engine (and operator dispatch) only ever imports ``current_platform``, never
a specific vendor module.  Resolution walks the known platforms in priority
order and returns the first whose ``is_available()`` reports true, falling back to
CPU so the process can still import cleanly on a machine with no accelerator.
"""

from __future__ import annotations

from nanovllm.platforms.cuda import CUDAPlatform
from nanovllm.platforms.interface import Platform
from nanovllm.platforms.vendors import AscendPlatform, MUSAPlatform, ROCmPlatform

# Order matters: a Hygon (ROCm) / Moore (MUSA) machine must be matched before the
# generic CUDA fallback, which would otherwise also report "available" when the
# vendor runtime is merely a CUDA translation layer.
_PLATFORMS: list[type[Platform]] = [
    AscendPlatform,
    MUSAPlatform,
    ROCmPlatform,
    CUDAPlatform,
]


def _resolve() -> Platform:
    for cls in _PLATFORMS:
        if cls().is_available():
            return cls()
    # Fallback: a CPU-only process still needs a platform object to import.
    from nanovllm.platforms.cpu import CPUPlatform

    return CPUPlatform()


current_platform: Platform = _resolve()