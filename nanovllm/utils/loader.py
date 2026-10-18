import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    local_parameters = dict(model.named_parameters())
    tied_groups = getattr(model, "tied_weight_groups", lambda: ())()
    loaded_weight_names = set()
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                packed = next(((k, v, shard_id) for k, (v, shard_id) in packed_modules_mapping.items()
                               if k in weight_name), None)
                param_name = weight_name.replace(packed[0], packed[1]) if packed else weight_name
                if param_name not in local_parameters:
                    param_name = next((name for group in tied_groups if weight_name in group
                                       for name in group if name in local_parameters), None)
                if param_name is None or param_name not in local_parameters:
                    continue
                param = local_parameters[param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if packed:
                    weight_loader(param, f.get_tensor(weight_name), packed[2])
                else:
                    weight_loader(param, f.get_tensor(weight_name))
                loaded_weight_names.add(weight_name)
    if hasattr(model, "expected_weight_names"):
        for names in tied_groups:
            if loaded_weight_names.intersection(names):
                loaded_weight_names.update(names)
        missing = model.expected_weight_names() - loaded_weight_names
        if missing:
            preview = ", ".join(sorted(missing)[:10])
            raise ValueError(f"Missing {len(missing)} model weights: {preview}")
