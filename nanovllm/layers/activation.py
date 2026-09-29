import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.platforms import current_platform


def _maybe_compile(fn):
    if current_platform.backend_name == "ascend":
        return fn
    return torch.compile(fn)


class SiluAndMul(nn.Module):

    @_maybe_compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
