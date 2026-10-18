import torch.distributed as dist


_tp_group = None


def init_tensor_parallel_group(tp_size: int, pp_size: int, rank: int):
    global _tp_group
    if pp_size == 1:
        _tp_group = dist.group.WORLD
        return
    # Every rank must create the groups in the same order.
    for stage in range(pp_size):
        group = dist.new_group(ranks=list(range(stage * tp_size, (stage + 1) * tp_size)))
        if rank // tp_size == stage:
            _tp_group = group


def get_tp_group():
    return _tp_group


def get_tp_size():
    return dist.get_world_size(group=_tp_group)


def get_tp_rank():
    return dist.get_rank(group=_tp_group)


def get_tp_leader():
    return dist.get_rank() - get_tp_rank()


def layer_range(num_layers: int, pp_size: int, pp_rank: int) -> range:
    start = num_layers * pp_rank // pp_size
    end = num_layers * (pp_rank + 1) // pp_size
    return range(start, end)
