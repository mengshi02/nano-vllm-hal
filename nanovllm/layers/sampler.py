import torch
from torch import nn

from nanovllm.platforms import current_platform


def _maybe_compile(fn):
    if current_platform.backend_name == "ascend":
        return fn
    return torch.compile(fn)


class Sampler(nn.Module):

    @_maybe_compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
