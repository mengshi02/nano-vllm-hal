import torch.distributed as dist


_tp_group = None
_tp_size = 1
_dp_size = 1
_ep_size = 1
_pp_size = 1
_ep_rank = 0
_expert_tensor_parallel_size = 1
_expert_tensor_parallel_rank = 0


def init_tensor_parallel_group(tp_size: int, pp_size: int, rank: int,
                               dp_size: int = 1):
    global _tp_group, _tp_size
    _tp_size = tp_size
    # A TP=1 worker needs no TP communicator. In particular, do not create
    # singleton NCCL process groups: some CUDA/NCCL builds leave the second
    # worker's default stream blocked after those groups are initialized.
    if tp_size == 1:
        _tp_group = dist.group.WORLD
        return
    # Every rank must create the groups in the same order.
    model_group_size = tp_size * pp_size
    for replica in range(dp_size):
        base = replica * model_group_size
        for stage in range(pp_size):
            ranks = list(range(base + stage * tp_size,
                              base + (stage + 1) * tp_size))
            group = dist.new_group(ranks=ranks)
            if (base <= rank < base + model_group_size
                    and (rank - base) // tp_size == stage):
                _tp_group = group


def init_parallel_groups(tp_size: int, pp_size: int, rank: int,
                         dp_size: int = 1, ep_size: int = 1):
    """Initialize TP groups and the MoE expert-placement metadata.

    EP is a sub-dimension of TP: expert groups are placed across the TP ranks
    of each pipeline stage, and each expert is optionally sharded across the
    remaining expert-tensor-parallel ranks.
    """
    global _dp_size, _ep_size, _pp_size, _ep_rank
    global _expert_tensor_parallel_size, _expert_tensor_parallel_rank
    _dp_size = dp_size
    _ep_size = ep_size
    _pp_size = pp_size
    _ep_rank = 0
    _expert_tensor_parallel_size = 1
    _expert_tensor_parallel_rank = 0
    init_tensor_parallel_group(tp_size, pp_size, rank, dp_size)

    model_group_size = tp_size * pp_size
    rank_in_model_group = rank % model_group_size
    tp_rank = rank_in_model_group % tp_size
    # EP divides TP.  Experts are grouped by EP rank, then each expert is
    # optionally sharded across the remaining TP ranks.
    _expert_tensor_parallel_size = tp_size // ep_size
    _expert_tensor_parallel_rank = tp_rank % _expert_tensor_parallel_size
    _ep_rank = tp_rank // _expert_tensor_parallel_size


def get_tp_group():
    return _tp_group


def get_tp_size():
    return _tp_size


def get_tp_rank():
    return dist.get_rank(group=_tp_group) if _tp_size > 1 else 0


def get_tp_leader():
    return dist.get_rank() - get_tp_rank()


def get_dp_size():
    return _dp_size


def get_dp_rank():
    return dist.get_rank() // (get_tp_size() * _pp_size)


def get_ep_size():
    return _ep_size


def get_local_ep_rank():
    """EP rank without consulting a collective group."""
    return _ep_rank


def get_expert_tensor_parallel_size():
    return _expert_tensor_parallel_size


def get_expert_tensor_parallel_rank():
    return _expert_tensor_parallel_rank


def layer_range(num_layers: int, pp_size: int, pp_rank: int) -> range:
    start = num_layers * pp_rank // pp_size
    end = num_layers * (pp_rank + 1) // pp_size
    return range(start, end)
