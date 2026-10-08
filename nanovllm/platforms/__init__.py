from __future__ import annotations

from nanovllm.platforms.cuda import CUDAPlatform
from nanovllm.platforms.interface import Platform
from nanovllm.platforms.vendors import AscendPlatform, HygonPlatform, MUSAPlatform, ROCmPlatform

_PLATFORMS: list[type[Platform]] = [
    AscendPlatform,
    MUSAPlatform,
    HygonPlatform,
    ROCmPlatform,
    CUDAPlatform,
]

def _resolve() -> Platform:
    for cls in _PLATFORMS:
        if cls().is_available():
            return cls()
    from nanovllm.platforms.cpu import CPUPlatform

    return CPUPlatform()

current_platform: Platform = _resolve()
