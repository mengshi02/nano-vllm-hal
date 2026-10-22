from torch import nn

from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.olmoe import OlmoeForCausalLM


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "qwen3": Qwen3ForCausalLM,
    "olmoe": OlmoeForCausalLM,
}


def get_model_class(model_type: str) -> type[nn.Module]:
    try:
        return MODEL_REGISTRY[model_type]
    except KeyError as exc:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unsupported model type {model_type!r}; supported types: {supported}"
        ) from exc